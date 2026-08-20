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

## How it works

1. **PLY load** (`src/ply.rs`): reads positions, scales (`exp`), rotations
   (normalized quaternions), colors (`f_dc` through `SH_C0`), opacity
   (`sigmoid`), and spherical-harmonics coefficients (degrees 1–3).
2. **GPU sort** (`src/sort.wgsl`): a bitonic sort orders splats farthest-first
   by view-space depth for correct back-to-front alpha blending.
3. **Render** (`src/shader.wgsl`): each splat projects its 3D covariance to a 2D
   ellipse, expands to a quad, applies view-dependent SH color, and blends with
   premultiplied alpha into an sRGB framebuffer.

## Testing

```sh
cargo test
```

`tests/golden.rs` renders known clouds and compares (RMSE) against goldens
produced by the reference renderer, for both the CPU and GPU sort paths. It
skips gracefully when the (external) source `.ply` files are absent.

## Scope

PLY input, the Spark render path, and a GPU bitonic sort. See `ISSUES.md` for
tracked follow-ups (other input formats, f16 packing, a tile radix sort, and the
remaining reference CLI flags).
