use anyhow::{Context, Result};
use glam::Mat4;
use image::RgbaImage;
use wgpu::util::DeviceExt;

use crate::camera::Camera;
use crate::cloud::SplatCloud;
use crate::sort::{self, SortMethod};

/// Settings for a single render.
#[derive(Clone, Debug)]
pub struct RenderSettings {
    pub width: u32,
    pub height: u32,
    /// Clear color, RGBA in 0..1.
    pub background: [f32; 4],
    /// Model transform placing the cloud in the world.
    pub model: Mat4,
    /// Convert sRGB to linear in the fragment shader.
    pub srgb_to_linear: bool,
    /// Depth sort method.
    pub sort: SortMethod,
}

impl Default for RenderSettings {
    fn default() -> Self {
        Self {
            width: 1024,
            height: 768,
            background: [0.0, 0.0, 0.0, 1.0],
            model: Mat4::IDENTITY,
            srgb_to_linear: false,
            sort: SortMethod::Gpu,
        }
    }
}

/// GPU-side splat: f16 position/scale/rotation plus rgba8 color, 24 bytes
/// (vs 64 for the f32 `GpuSplat`). Unpacked in WGSL with `unpack2x16float`.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct PackedSplat {
    position_xy: u32,
    position_z_scale_x: u32,
    scale_yz: u32,
    rotation_xy: u32,
    rotation_zw: u32,
    color: u32,
}

fn pack_f16_pair(low: f32, high: f32) -> u32 {
    half::f16::from_f32(low).to_bits() as u32 | (half::f16::from_f32(high).to_bits() as u32) << 16
}

fn pack_unorm8(value: f32) -> u32 {
    (value.clamp(0.0, 1.0) * 255.0).round() as u32
}

impl From<&crate::cloud::GpuSplat> for PackedSplat {
    fn from(s: &crate::cloud::GpuSplat) -> Self {
        PackedSplat {
            position_xy: pack_f16_pair(s.position[0], s.position[1]),
            position_z_scale_x: pack_f16_pair(s.position[2], s.scale[0]),
            scale_yz: pack_f16_pair(s.scale[1], s.scale[2]),
            rotation_xy: pack_f16_pair(s.rotation[0], s.rotation[1]),
            rotation_zw: pack_f16_pair(s.rotation[2], s.rotation[3]),
            color: pack_unorm8(s.color[0])
                | pack_unorm8(s.color[1]) << 8
                | pack_unorm8(s.color[2]) << 16
                | pack_unorm8(s.color[3]) << 24,
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    view: [f32; 16],
    projection: [f32; 16],
    model: [f32; 16],
    camera_pos: [f32; 4],
    drawable_size: [f32; 2],
    sh_degree: u32,
    srgb_to_linear: u32,
}

/// A reusable renderer that owns a `wgpu` device. Create once, render many.
pub struct Renderer {
    device: wgpu::Device,
    queue: wgpu::Queue,
    bind_group_layout: wgpu::BindGroupLayout,
    pipeline: wgpu::RenderPipeline,
    sort_pipelines: sort::SortPipelines,
}

const TARGET_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;
impl Renderer {
    /// Create a renderer on the default high-performance adapter.
    ///
    /// # Errors
    ///
    /// Fails when no suitable GPU adapter is found or the device request
    /// is denied.
    pub fn new() -> Result<Self> {
        pollster::block_on(Self::new_async())
    }

    async fn new_async() -> Result<Self> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                ..Default::default()
            })
            .await
            .context("no suitable GPU adapter")?;
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("splat-device"),
                required_features: wgpu::Features::empty(),
                required_limits: adapter.limits(),
                memory_hints: wgpu::MemoryHints::Performance,
                ..Default::default()
            })
            .await
            .context("request device")?;

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("splat-shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bgl"),
            entries: &[
                uniform_entry(0),
                storage_entry(1),
                storage_entry(2),
                storage_entry(3),
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("pl"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        // Premultiplied "over" blending.
        let blend_component = wgpu::BlendComponent {
            src_factor: wgpu::BlendFactor::One,
            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
            operation: wgpu::BlendOperation::Add,
        };
        let blend = wgpu::BlendState {
            color: blend_component,
            alpha: blend_component,
        };

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("splat-pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: TARGET_FORMAT,
                    blend: Some(blend),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        let sort_pipelines = sort::SortPipelines::new(&device);

        Ok(Self {
            device,
            queue,
            bind_group_layout,
            pipeline,
            sort_pipelines,
        })
    }

    /// Render a splat cloud to an RGBA image.
    ///
    /// # Errors
    ///
    /// Fails when the splat count exceeds the GPU sort limit (2^31) or the
    /// readback from the GPU fails.
    pub fn render(
        &self,
        cloud: &SplatCloud,
        camera: &Camera,
        settings: &RenderSettings,
    ) -> Result<RgbaImage> {
        pollster::block_on(self.render_async(cloud, camera, settings))
    }

    async fn render_async(
        &self,
        cloud: &SplatCloud,
        camera: &Camera,
        settings: &RenderSettings,
    ) -> Result<RgbaImage> {
        let device = &self.device;
        let queue = &self.queue;
        let width = settings.width;
        let height = settings.height;

        // Buffers. Splats are packed to 24 bytes (f16 + rgba8) for the GPU.
        let packed: Vec<PackedSplat> = cloud.splats().iter().map(PackedSplat::from).collect();
        let splat_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("splats"),
            contents: bytemuck::cast_slice(&packed),
            usage: wgpu::BufferUsages::STORAGE,
        });
        drop(packed);

        // SH buffer must be non-empty for the binding even at degree 0.
        let sh_data: &[f32] = if cloud.sh_coefficients().is_empty() {
            &[0.0]
        } else {
            cloud.sh_coefficients()
        };
        let sh_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("sh"),
            contents: bytemuck::cast_slice(sh_data),
            usage: wgpu::BufferUsages::STORAGE,
        });

        // Sorted draw order: farthest-first splat indices.
        let count = cloud.splats().len() as u32;
        let index_buf = match settings.sort {
            SortMethod::Cpu => {
                let indices = sort::cpu_sort(cloud.splats(), settings.model, camera.view);
                device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("sorted-indices"),
                    contents: bytemuck::cast_slice(&indices),
                    usage: wgpu::BufferUsages::STORAGE,
                })
            }
            SortMethod::Gpu => sort::gpu_sort(
                device,
                queue,
                &self.sort_pipelines,
                &splat_buf,
                count,
                camera.view * settings.model,
            )?,
        };

        let uniforms = Uniforms {
            view: camera.view.to_cols_array(),
            projection: camera.projection.to_cols_array(),
            model: settings.model.to_cols_array(),
            camera_pos: [camera.position.x, camera.position.y, camera.position.z, 1.0],
            drawable_size: [width as f32, height as f32],
            sh_degree: cloud.sh_degree(),
            srgb_to_linear: if settings.srgb_to_linear { 1 } else { 0 },
        };
        let uniform_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("uniforms"),
            contents: bytemuck::bytes_of(&uniforms),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        // sRGB target: the reference renders into a bgra8Unorm_srgb framebuffer, so
        // blending happens in linear space and values are sRGB-encoded on store.
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("target"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: TARGET_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bg"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: splat_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: sh_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: index_buf.as_entire_binding(),
                },
            ],
        });

        // Readback buffer with 256-byte row alignment.
        let bytes_per_pixel = 4u32;
        let unpadded = width * bytes_per_pixel;
        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let padded = unpadded.div_ceil(align) * align;
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: (padded * height) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("enc") });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("splat-pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    depth_slice: None,
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: settings.background[0] as f64,
                            g: settings.background[1] as f64,
                            b: settings.background[2] as f64,
                            a: settings.background[3] as f64,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.draw(0..4, 0..count);
        }

        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded),
                    rows_per_image: Some(height),
                },
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );

        queue.submit(Some(encoder.finish()));

        // Map and read back.
        let slice = readback.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .context("device poll")?;
        rx.recv().context("map channel")?.context("map readback")?;

        let data = slice.get_mapped_range().context("mapped range")?;
        // Strip per-row alignment padding with row-sized copies.
        let mut pixels = vec![0u8; (unpadded * height) as usize];
        for (y, out_row) in pixels.chunks_exact_mut(unpadded as usize).enumerate() {
            let start = y * padded as usize;
            out_row.copy_from_slice(&data[start..start + unpadded as usize]);
        }
        let img = RgbaImage::from_raw(width, height, pixels).context("readback size mismatch")?;
        drop(data);
        readback.unmap();

        Ok(img)
    }
}

fn uniform_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn storage_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}
