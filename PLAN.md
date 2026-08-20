# splat-rs — PLAN

A CLI tool that loads a 3D Gaussian Splat file and renders it offscreen to a PNG
using `wgpu`. Ports the **GPU/Spark renderer** from
`~/Projects/Current/MetalSprocketsGaussianSplats` (Metal) to Rust + WGSL.

## Scope (lazy)

- One binary: `splat-rs <splat.ply> [flags]` → writes a PNG. No windowing, no
  interactivity, no framework crate.
- Input: **PLY only** (via `ply-rs`). Reference also does .splat/.spz/.sog — skip;
  add later if needed.
- Renderer: the single Spark-equivalent path (project + Gaussian quad + alpha
  blend). Skip the reference's other four renderers (point/tile/stochastic/debug).
- Sort: **GPU radix sort** on view-space depth (port `SplatGPUSort.metal` /
  `GPUSplatSortComputePass`: cull + key/index build + radix sort compute passes,
  feeding an indirect draw). CPU radix sort as a first-step fallback while
  bringing up the pipeline, then switch to GPU.
- SH: full **spherical harmonics degrees 0–3** (port `evaluateSH` from
  `SparkSplatSupport.h`), view-dependent color evaluated in the vertex shader.

## Dependencies

- `wgpu` — Metal backend on this machine, offscreen render to a texture.
- `ply-rs` — parse ASCII + binary_little_endian PLY.
- `clap` (derive) — CLI, mirrors the reference flags.
- `glam` — vec/mat/quat math (Mat4, Quat, Vec3).
- `image` — encode the readback buffer to PNG.
- `pollster` — block on `wgpu` async on the main thread.
- `half` — pack splats as f16 to match the reference's 32-byte layout (or just
  use f32 and skip; f16 packing is `ponytail:` optional).
- `bytemuck` — cast structs to GPU bytes.

## CLI (subset of MetalSprocketsGaussianSplats)

```
splat-rs <SPLAT>
  --output <path>            default output.png
  --width <n>                default 1024
  --height <n>               default 768
  --background r,g,b,a       default 0,0,0,1
  --camera-position x,y,z    default 0,0,1.5
  --camera-lookat x,y,z      default 0,0,0
  --camera-matrix <16 vals>  column-major c2w; overrides the camera flags
  --model-position x,y,z     default 0,0,0
  --projection-fov <deg>     default 60 (vertical FOV)
  --near <f>                 default 0.1
  --far <f>                  default 100
  --srgb-to-linear           flag (fragment pow 2.2)
```
Skip for now: `--config` JSON, `--renderer`, `--sort`, `--statistics`,
`--warmup/--frames`, `--label`, `--reveal`, `--capture`, camera-rotation quat.
Camera priority: `--camera-matrix` > (`--camera-lookat` + `--camera-position`) >
`--camera-position`. Same rule as the reference.

## Data flow

1. **Parse PLY** → for each vertex read:
   - position: `x,y,z`
   - scale: `scale_0..2` → **`exp()`** (activation)
   - rotation: quaternion `rot_0=w, rot_1=x, rot_2=y, rot_3=z`, normalized
   - color: `f_dc_0..2` → `dc * 0.2820947918 + 0.5`, clamp 0..1
   - opacity: `opacity` → **sigmoid** `1/(1+exp(-o))`, clamp 0..1
   - SH: `f_dc_0..2` is the DC term (→ base color above). `f_rest_*` holds the
     higher-order coefficients. Detect degree from the highest `f_rest_N`
     present (`f_rest_8`→deg1, `f_rest_23`→deg2, `f_rest_44`→deg3). PLY stores
     `f_rest` **planar** (all R, then all G, then all B); de-interleave to
     `[[R,G,B], ...]` per basis function, matching `PLYSplatReader`.
   (These activations are copied verbatim from `PLYSplatReader.swift` /
   `GenericSplat`. Getting them wrong = blank/garbage output.)
2. **Upload** two GPU buffers:
   - splats: struct { position f32x3, scale f32x3, rotation f32x4, color u8x4 }.
   - SH coefficients: a separate storage buffer, interleaved
     `[c0_r,c0_g,c0_b, c1_r,...]` per splat (9/24/45 floats for deg 1/2/3),
     indexed as `splatIndex * floatsPerSplat` — same layout `evaluateSH` expects.
3. **Per frame (once):**
   - Build `view = inverse(camera_c2w)`, `proj = perspective(fov, aspect, near, far)`,
     `model = translate(model_position)`.
   - **GPU sort** (port `GPUSplatSortComputePass` / `SplatGPUSort.metal`): a
     cull+key pass writes `(depthKey, splatIndex)` for visible splats and an
     atomic visible-count, then a radix sort (4 passes of 8-bit digits over the
     32-bit key) orders them back-to-front. The visible count drives an indirect
     draw. Keep a CPU radix-sort fallback for bring-up.
   - Draw: instanced quad (4 verts, triangle-strip), one instance per visible
     splat, indexed through the sorted `(splatIndex)` buffer via indirect draw.
4. **Vertex shader (WGSL port of `SparkSplatRenderShader::vertex_main`):**
   - world = model * center; view = view * world; cull behind camera / frustum.
   - **SH color** (port `evaluateSH`): if degree > 0, viewDir = normalize(world -
     cameraPosition), add the SH contribution to the DC base color, clamp 0..1.
     Evaluated per-splat here so the fragment shader stays flat-shaded.
   - RS = scaleQuaternionToMatrix(scale, quat);
     viewRS = rot3(view) * rot3(model) * RS.
   - cov3D = viewRS * viewRSᵀ.
   - focal = 0.5 * drawableSize * (proj[0][0], proj[1][1]).
   - J = projection Jacobian at view pos; cov2D = Jᵀ cov3D J.
   - Dilate cov2D by BLUR_AMOUNT (0.05), rescale alpha by sqrt(detOrig/det).
   - eigendecompose2D → major/minor axes (× MAX_STD_DEV=2.5, clamp
     MAX_PIXEL_RADIUS=512).
   - quad vertex = ndcCenter + (2/size) * (qx*major + qy*minor) offset.
   - Pass `splatUv = quadPos * MAX_STD_DEV` and rgba to fragment.
   All helper math is in `SparkSplatSupport.h` — port each inline fn to WGSL.
5. **Fragment shader (port of `fragment_main`):**
   - optional sRGB→linear (`pow(rgb, 2.2)`).
   - `z = dot(uv,uv)`; discard if `z > MAX_STD_DEV²`.
   - `alpha *= exp(-0.5*z)`; discard if `< MIN_ALPHA (2/255)`.
   - output **premultiplied**: `vec4(rgb*alpha, alpha)`.
6. **Blend state:** premultiplied over. src = ONE, dst = ONE_MINUS_SRC_ALPHA,
   for both color and alpha. No depth test (order comes from the GPU sort).
7. **Readback** the RGBA8 texture → `image` → PNG. Handle wgpu's 256-byte row
   alignment on copy.

## Constants (from the Metal shader — keep identical)

`CLIP_XY=1.4`, `MAX_PIXEL_RADIUS=512`, `MAX_STD_DEV=2.5`, `MIN_ALPHA=2/255`,
`BLUR_AMOUNT=0.05`, `SH_C0=0.28209479177387814`.

## Coordinate-system watch-outs

- Metal NDC z is [0,1]; wgpu NDC z is also [0,1] → perspective matrix matches
  (use `glam::Mat4::perspective_rh` / build to match reference depth mode).
- Metal uses column-major simd matrices; glam is column-major too — but verify
  the Jacobian/covariance index order when porting (`cov2D[0][0]` etc.).
- Texture is written top-left origin; confirm PNG isn't flipped (flip on readback
  if needed). `ponytail: flip row order in readback if image is upside down`.

## File layout

```
splat-rs/
  Cargo.toml
  PLAN.md
  src/
    main.rs        # clap args, orchestration
    ply.rs         # PLY -> Vec<Splat> + SH coeffs, with activations
    camera.rs      # arg parsing + view/proj/model matrices
    render.rs      # wgpu device, buffers, pipeline, draw, readback
    sort.rs        # GPU radix sort passes (+ CPU fallback)
    shader.wgsl    # render vertex + fragment (Spark port)
    sort.wgsl      # cull/key + radix sort compute (SplatGPUSort port)
```

## Milestones

1. Scaffold + `Cargo.toml` + clap args parsing; print a summary. Build.
2. `ply.rs`: parse a real 3DGS PLY, apply activations, assert splat count and a
   sample splat's values in a `#[test]`. **Check: parse the sample .ply from the
   reference `Samples/` dir and compare count.**
3. `render.rs`: wgpu init, clear to background color, readback → PNG. Verify a
   solid-color image writes.
4. Wire the splat buffer + quad draw + `shader.wgsl` with a **CPU** back-to-front
   sort (fallback path). DC color only. Render one splat, then the full cloud;
   eyeball against the reference PNG.
5. Add **SH degrees 1–3**: upload the coefficient buffer, port `evaluateSH` into
   the vertex shader. Compare a view-dependent file against the reference.
6. Add the **GPU radix sort** (`sort.wgsl` + `sort.rs`): cull/key pass, radix
   sort, indirect draw. Swap it in for the CPU sort; verify identical output.
7. Camera flags (lookat, matrix), fov, near/far, srgb-to-linear.
8. (Later) f16 packing; more input formats (.splat/.spz/.sog).

## Verification

- One runnable check: `cargo test` parses a known PLY and asserts count +
  activation on a fixed vertex (catches the exp/sigmoid/SH_C0 mistakes).
- Visual: render the same file with both tools and diff the PNGs by eye. The
  reference lives at `~/Projects/Current/MetalSprocketsGaussianSplats`; grab a
  sample `.ply` from its `Samples/` dir.
```
skipped: stats/label/config flags, non-PLY formats, f16 packing. add when the
base render matches the reference.
```
