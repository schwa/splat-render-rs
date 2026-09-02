use glam::{Mat4, Vec3};

use crate::cloud::GpuSplat;

/// How to order splats back-to-front before drawing.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum SortMethod {
    /// CPU sort (bring-up fallback).
    Cpu,
    /// GPU bitonic sort.
    Gpu,
}

/// Back-to-front order of splat indices by view-space depth.
///
/// Farthest (most negative view z / largest positive depth) first, so the
/// premultiplied "over" blend composites correctly.
pub fn cpu_sort(splats: &[GpuSplat], model: Mat4, view: Mat4) -> Vec<u32> {
    let mv = view * model;
    let mut order: Vec<(f32, u32)> = splats
        .iter()
        .enumerate()
        .map(|(i, s)| {
            let p = Vec3::new(s.position[0], s.position[1], s.position[2]);
            // View-space depth = -z (positive in front of camera).
            let depth = -(mv.transform_point3(p).z);
            (depth, i as u32)
        })
        .collect();
    // Farthest first: descending depth.
    order.sort_by(|a, b| b.0.total_cmp(&a.0));
    order.into_iter().map(|(_, i)| i).collect()
}

// MARK: - GPU bitonic sort

use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct BuildParams {
    model_view: [f32; 16],
    count: u32,
    padded: u32,
    _pad: [u32; 2],
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct StepParams {
    k: u32,
    j: u32,
    _pad: [u32; 2],
}
/// Cached compute pipelines for the GPU bitonic sort. Create once per device.
pub(crate) struct SortPipelines {
    layout: wgpu::BindGroupLayout,
    build_pipeline: wgpu::ComputePipeline,
    step_pipeline: wgpu::ComputePipeline,
    extract_pipeline: wgpu::ComputePipeline,
}

impl SortPipelines {
    pub(crate) fn new(device: &wgpu::Device) -> Self {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("sort-shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("sort.wgsl").into()),
        });

        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("sort-bgl"),
            entries: &[
                storage_bgl_entry(0, true),
                storage_bgl_entry(1, false),
                uniform_bgl_entry(2, false),
                uniform_bgl_entry(3, true),
                storage_bgl_entry(4, false),
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("sort-pl"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let make = |entry: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&pipeline_layout),
                module: &module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        Self {
            build_pipeline: make("build_records"),
            step_pipeline: make("bitonic_step"),
            extract_pipeline: make("extract_indices"),
            layout,
        }
    }
}

/// Sort splats back-to-front on the GPU (bitonic) and return an index buffer of
/// `count` splat indices in draw order.
pub(crate) fn gpu_sort(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    pipelines: &SortPipelines,
    splat_buf: &wgpu::Buffer,
    count: u32,
    model_view: Mat4,
) -> anyhow::Result<wgpu::Buffer> {
    let padded = padded_count(count)
        .ok_or_else(|| anyhow::anyhow!("too many splats to sort on GPU: {count}"))?;

    let records = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("sort-records"),
        size: (padded as u64) * 8,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });

    let index_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("sorted-indices"),
        size: (count.max(1) as u64) * 4,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });

    let build = BuildParams {
        model_view: model_view.to_cols_array(),
        count,
        padded,
        _pad: [0, 0],
    };
    let build_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("sort-build-params"),
        contents: bytemuck::bytes_of(&build),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    // Bitonic passes: (k, j) pairs. Pack each into a dynamic-offset uniform slot.
    let mut steps: Vec<StepParams> = Vec::new();
    let mut k = 2u32;
    while k <= padded {
        let mut j = k / 2;
        while j >= 1 {
            steps.push(StepParams { k, j, _pad: [0, 0] });
            j /= 2;
        }
        k *= 2;
    }

    let align = device.limits().min_uniform_buffer_offset_alignment.max(256) as u64;
    let mut step_data = vec![0u8; steps.len() * align as usize];
    for (i, s) in steps.iter().enumerate() {
        let off = i * align as usize;
        step_data[off..off + std::mem::size_of::<StepParams>()]
            .copy_from_slice(bytemuck::bytes_of(s));
    }
    let step_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("sort-step-params"),
        contents: &step_data,
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("sort-bg"),
        layout: &pipelines.layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: splat_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: records.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: build_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &step_buf,
                    offset: 0,
                    size: wgpu::BufferSize::new(std::mem::size_of::<StepParams>() as u64),
                }),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: index_buf.as_entire_binding(),
            },
        ],
    });

    let groups = |n: u32| n.div_ceil(256);

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("sort-enc"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("sort-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipelines.build_pipeline);
        pass.set_bind_group(0, &bind_group, &[0]);
        pass.dispatch_workgroups(groups(padded), 1, 1);

        pass.set_pipeline(&pipelines.step_pipeline);
        for i in 0..steps.len() {
            let offset = (i as u64 * align) as u32;
            pass.set_bind_group(0, &bind_group, &[offset]);
            pass.dispatch_workgroups(groups(padded), 1, 1);
        }

        pass.set_pipeline(&pipelines.extract_pipeline);
        pass.set_bind_group(0, &bind_group, &[0]);
        pass.dispatch_workgroups(groups(count), 1, 1);
    }
    queue.submit(Some(encoder.finish()));

    Ok(index_buf)
}

/// Bitonic working size: next power of two above `count`, at least 2.
/// `None` when the padded size would overflow `u32`.
fn padded_count(count: u32) -> Option<u32> {
    count.checked_next_power_of_two().map(|p| p.max(2))
}

fn storage_bgl_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn uniform_bgl_entry(binding: u32, dynamic: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: dynamic,
            min_binding_size: None,
        },
        count: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn padded_count_rounds_up_and_rejects_overflow() {
        assert_eq!(padded_count(0), Some(2));
        assert_eq!(padded_count(1), Some(2));
        assert_eq!(padded_count(3), Some(4));
        assert_eq!(padded_count(1 << 31), Some(1 << 31));
        // Anything above 2^31 has no u32 power-of-two ceiling.
        assert_eq!(padded_count((1 << 31) + 1), None);
        assert_eq!(padded_count(u32::MAX), None);
    }
}
