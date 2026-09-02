# splat-render-rs

A CLI tool that renders 3D Gaussian Splat files to PNG using `wgpu`. It is a
Rust + WGSL port of the GPU/Spark renderer from
[MetalSprocketsGaussianSplats](https://github.com/schwa/MetalSprocketsGaussianSplats)
(`--renderer spark --sort gpu`).

## Usage

```sh
splat-render-rs <file.ply> [options]
```

Renders a `.ply` Gaussian splat file to `output.png`.

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--output <path>` | `output.png` | Output PNG path |
| `--width <n>` | `1024` | Image width |
| `--height <n>` | `768` | Image height |
| `--background r,g,b,a` | `0,0,0,1` | Clear color |
| `--camera-position x,y,z` | `0,0,1.5` | Eye position |
| `--camera-lookat x,y,z` | — | Look-at target |
| `--camera-matrix <16>` | — | Column-major camera-to-world; overrides the camera flags |
| `--model-position x,y,z` | `0,0,0` | Cloud placement |
| `--projection-fov <deg>` | `60` | Vertical field of view |
| `--near <f>` / `--far <f>` | `0.1` / `100` | Clip planes |
| `--srgb-to-linear` | off | Convert sRGB→linear in the fragment shader |
| `--sort cpu\|gpu` | `gpu` | Depth sort: GPU bitonic or CPU fallback |

Example:

```sh
splat-render-rs scene.ply --camera-position 0,0,4 --camera-lookat 0,0,0 --output scene.png
```

## Library

The CLI is a thin front-end over the `splat_render_rs` crate:

```rust
use glam::Vec3;
use splat_render_rs::{Camera, Perspective, RenderSettings, Renderer, SplatCloud};

let cloud = SplatCloud::from_ply_path("scene.ply")?;
let settings = RenderSettings::default();
let aspect = settings.width as f32 / settings.height as f32;
let camera = Camera::look_at(
    Vec3::new(0.0, 0.0, 4.0),
    Vec3::ZERO,
    Vec3::Y,
    Perspective::new(60f32.to_radians(), aspect, 0.1, 100.0),
);
let renderer = Renderer::new()?;                 // owns the wgpu device; reuse it
let image = renderer.render(&cloud, &camera, &settings)?;
image.save("scene.png")?;
```

Key types: `SplatCloud` (`from_ply_path` / `from_ply_bytes` / `from_ply_reader`),
`Camera` (`look_at` / `from_camera_to_world`), `Perspective`, `RenderSettings`,
`Renderer`, and `SortMethod`.
