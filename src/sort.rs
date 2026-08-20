use clap::ValueEnum;
use glam::{Mat4, Vec3};

use crate::ply::GpuSplat;

#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
pub enum SortMethod {
    /// CPU sort (bring-up fallback).
    Cpu,
    /// GPU radix sort compute passes.
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
    order.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    order.into_iter().map(|(_, i)| i).collect()
}
