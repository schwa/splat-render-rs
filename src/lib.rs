//! Offscreen 3D Gaussian Splat renderer built on `wgpu`.
//!
//! Ports the GPU/Spark renderer from MetalSprocketsGaussianSplats to Rust + WGSL.
//!
//! ```no_run
//! use glam::Vec3;
//! use splat_render_rs::{Camera, Perspective, RenderSettings, Renderer, SplatCloud};
//!
//! let cloud = SplatCloud::from_ply_path("scene.ply")?;
//! let settings = RenderSettings::default();
//! let aspect = settings.width as f32 / settings.height as f32;
//! let camera = Camera::look_at(
//!     Vec3::new(0.0, 0.0, 4.0),
//!     Vec3::ZERO,
//!     Vec3::Y,
//!     Perspective::new(60f32.to_radians(), aspect, 0.1, 100.0),
//! );
//! let renderer = Renderer::new()?;
//! let image = renderer.render(&cloud, &camera, &settings)?;
//! image.save("scene.png")?;
//! # Ok::<(), anyhow::Error>(())
//! ```

mod camera;
mod cloud;
mod render;
mod sort;

pub use camera::{Camera, Perspective};
pub use cloud::{floats_per_splat, GpuSplat, SplatCloud};
pub use render::{RenderSettings, Renderer};
pub use sort::SortMethod;
