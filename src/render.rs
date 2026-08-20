use anyhow::{Context, Result};
use image::RgbaImage;
use wgpu::util::DeviceExt;

use crate::camera::Camera;
use crate::ply::{floats_per_splat, SplatCloud};
use crate::sort::{self, SortMethod};
use crate::Args;

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

pub fn render(
    cloud: &SplatCloud,
    cam: &Camera,
    args: &Args,
    background: [f32; 4],
) -> Result<RgbaImage> {
    pollster::block_on(render_async(cloud, cam, args, background))
}

async fn render_async(
    cloud: &SplatCloud,
    cam: &Camera,
    args: &Args,
    background: [f32; 4],
) -> Result<RgbaImage> {
    let width = args.width;
    let height = args.height;

    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        })
        .await
        .context("no suitable GPU adapter")?;

    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: Some("splat-device"),
                required_features: wgpu::Features::empty(),
                required_limits: adapter.limits(),
                memory_hints: wgpu::MemoryHints::Performance,
            },
            None,
        )
        .await
        .context("request device")?;

    // Buffers.
    let splat_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("splats"),
        contents: bytemuck::cast_slice(&cloud.splats),
        usage: wgpu::BufferUsages::STORAGE,
    });

    // SH buffer must be non-empty for the binding even at degree 0.
    let sh_data: Vec<f32> = if cloud.sh_coefficients.is_empty() {
        vec![0.0]
    } else {
        cloud.sh_coefficients.clone()
    };
    let sh_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("sh"),
        contents: bytemuck::cast_slice(&sh_data),
        usage: wgpu::BufferUsages::STORAGE,
    });

    // Sorted draw order: farthest-first splat indices.
    let count = cloud.splats.len() as u32;
    let index_buf = match args.sort {
        SortMethod::Cpu => {
            let indices = sort::cpu_sort(&cloud.splats, cam.model, cam.view);
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("sorted-indices"),
                contents: bytemuck::cast_slice(&indices),
                usage: wgpu::BufferUsages::STORAGE,
            })
        }
        SortMethod::Gpu => {
            sort::gpu_sort(&device, &queue, &splat_buf, count, cam.view * cam.model)
        }
    };

    let _ = floats_per_splat; // used by ply/tests

    let uniforms = Uniforms {
        view: cam.view.to_cols_array(),
        projection: cam.projection.to_cols_array(),
        model: cam.model.to_cols_array(),
        camera_pos: [cam.position.x, cam.position.y, cam.position.z, 1.0],
        drawable_size: [width as f32, height as f32],
        sh_degree: cloud.sh_degree,
        srgb_to_linear: if args.srgb_to_linear { 1 } else { 0 },
    };
    let uniform_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("uniforms"),
        contents: bytemuck::bytes_of(&uniforms),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    // Render target.
    // sRGB target: the reference renders into a bgra8Unorm_srgb framebuffer, so
    // blending happens in linear space and values are sRGB-encoded on store.
    let format = wgpu::TextureFormat::Rgba8UnormSrgb;
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
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

    // Pipeline.
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

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("bg"),
        layout: &bind_group_layout,
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

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("pl"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    // Premultiplied "over" blending.
    let blend = wgpu::BlendState {
        color: wgpu::BlendComponent {
            src_factor: wgpu::BlendFactor::One,
            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
            operation: wgpu::BlendOperation::Add,
        },
        alpha: wgpu::BlendComponent {
            src_factor: wgpu::BlendFactor::One,
            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
            operation: wgpu::BlendOperation::Add,
        },
    };

    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("splat-pipeline"),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: "vs_main",
            buffers: &[],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: "fs_main",
            targets: &[Some(wgpu::ColorTargetState {
                format,
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
        multiview: None,
        cache: None,
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
                view: &view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: background[0] as f64,
                        g: background[1] as f64,
                        b: background[2] as f64,
                        a: background[3] as f64,
                    }),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.draw(0..4, 0..cloud.splats.len() as u32);
    }

    encoder.copy_texture_to_buffer(
        wgpu::ImageCopyTexture {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::ImageCopyBuffer {
            buffer: &readback,
            layout: wgpu::ImageDataLayout {
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
    device.poll(wgpu::Maintain::Wait);
    rx.recv().context("map channel")?.context("map readback")?;

    let data = slice.get_mapped_range();
    let mut img = RgbaImage::new(width, height);
    for y in 0..height {
        let row = &data[(y * padded) as usize..(y * padded + unpadded) as usize];
        for x in 0..width {
            let i = (x * 4) as usize;
            img.put_pixel(
                x,
                y,
                image::Rgba([row[i], row[i + 1], row[i + 2], row[i + 3]]),
            );
        }
    }
    drop(data);
    readback.unmap();

    Ok(img)
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
