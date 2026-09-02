# ISSUES.md

File format: <https://github.com/schwa/issues-format>

---

## 1: GPU radix sort not implemented; renderer falls back to CPU sort

+++
status: closed
priority: medium
kind: none
labels: renderer
created: 2026-08-20T16:58:49Z
updated: 2026-08-20T17:04:15Z
closed: 2026-08-20T17:04:15Z
+++

The --sort gpu path prints a note and uses the CPU back-to-front sort instead. The reference GPU renderer (SplatGPUSort.metal / GPUSplatSortComputePass) does a cull+key pass, a 4x8-bit radix sort, and an indirect draw entirely on the GPU. Port this so large clouds don't pay the per-frame CPU sort cost.

- `2026-08-20T17:04:15Z`: Implemented as a GPU bitonic sort (src/sort.wgsl + gpu_sort in sort.rs): build_records computes a depth key per splat, log^2(n) bitonic_step passes order them farthest-first, extract_indices fills the draw index buffer. Not the reference's tile radix, but fully on-GPU and no atomics/subgroups. Golden test covers both sort paths (RMSE < 0.006 vs reference). Left a ponytail note to swap in a radix if sort time dominates.

---

## 2: Only PLY input is supported; reference also reads .splat/.spz/.sog

+++
status: open
priority: low
kind: feature
labels: io, effort:l
created: 2026-08-20T16:58:49Z
updated: 2026-09-02T19:34:42Z
+++

load_ply handles PLY (ascii + binary_little_endian) only. The reference SplatLoader reads .splat, .spz, and .sog (the latter GPU-decoded). Add these formats.

---

## 3: Splats stored as f32 (64 bytes), not f16-packed like the reference

+++
status: open
priority: low
kind: enhancement
labels: perf, effort:l
created: 2026-08-20T16:58:49Z
updated: 2026-09-02T19:34:42Z
+++

GpuSplat is 64 bytes of f32. The reference SparkSplat is a 32-byte half-precision struct. Packing to f16 halves buffer size and bandwidth. Verify visual parity after packing.

---

## 4: Unported reference CLI flags: config JSON, statistics, label, camera-rotation, capture, reveal

+++
status: open
priority: low
kind: feature
labels: cli, effort:l
created: 2026-08-20T16:58:49Z
updated: 2026-09-02T19:34:42Z
+++

splat-rs implements core render flags. Reference also has --config, --statistics/--stats, --label, --camera-rotation, --capture, --reveal, --renderer selection, point tuning.

---

## 5: SH_C0 constant has excessive f32 precision (clippy warning)

+++
status: open
priority: low
kind: task
labels: lint, effort:xs
created: 2026-09-02T18:59:04Z
updated: 2026-09-02T19:34:42Z
+++

src/cloud.rs:8: clippy::excessive_precision fires on SH_C0 = 0.28209479177387814; the extra digits are silently dropped by f32.

---

## 6: Manual div_ceil in gpu_sort workgroup count (clippy warning)

+++
status: open
priority: low
kind: task
labels: lint, effort:xs
created: 2026-09-02T18:59:04Z
updated: 2026-09-02T19:34:42Z
+++

src/sort.rs:172: clippy::manual_div_ceil on '(n + 255) / 256'. render.rs already uses div_ceil for row padding, so the codebase is inconsistent.

---

## 7: Renderer rebuilds shader modules and pipelines on every render

+++
status: closed
priority: high
kind: enhancement
labels: perf, effort:m
created: 2026-09-02T18:59:04Z
updated: 2026-09-02T19:40:46Z
closed: 2026-09-02T19:40:46Z
+++

Renderer is documented as 'create once, render many' but render_async recreates the shader module, bind group layout, pipeline layout, and render pipeline per call. gpu_sort likewise compiles its WGSL and three compute pipelines on every sort. Pipeline compilation dominates per-frame cost for repeated renders.

- `2026-09-02T19:40:46Z`: Render pipeline, bind group layout, and sort compute pipelines now built once in Renderer::new and reused per render. Added tests/render_smoke.rs covering repeated renders and cpu/gpu sort parity.

---

## 8: SH coefficient vector is cloned on every render

+++
status: closed
priority: medium
kind: enhancement
labels: perf, effort:xs
created: 2026-09-02T18:59:04Z
updated: 2026-09-02T19:40:59Z
closed: 2026-09-02T19:40:59Z
+++

render.rs render_async clones cloud.sh_coefficients (up to 45 floats per splat, i.e. hundreds of MB for large clouds) just to substitute a 1-element placeholder when empty. The buffer upload could borrow the slice instead.

- `2026-09-02T19:40:59Z`: SH upload now borrows cloud.sh_coefficients (or a static 1-float placeholder) instead of cloning the whole vector each render.

---

## 9: PLY loader allocates format! key strings per coefficient per splat

+++
status: open
priority: medium
kind: enhancement
labels: perf, effort:s
created: 2026-09-02T18:59:04Z
updated: 2026-09-02T19:34:42Z
+++

cloud.rs cloud_from_ply builds f_rest_N property names with format! inside the per-vertex loop: 3 string allocations per coefficient, ~135M allocations for a 1M-splat degree-3 cloud. Key names depend only on the degree and could be computed once.

---

## 10: SplatCloud public fields allow breaking the SH length invariant

+++
status: open
priority: medium
kind: enhancement
labels: api, effort:s
created: 2026-09-02T18:59:04Z
updated: 2026-09-02T19:34:42Z
+++

SplatCloud::with_sh validates sh_coefficients.len() == splats.len() * floats_per_splat(sh_degree), but all fields are pub so callers can mutate them afterwards and violate the invariant the shader relies on.

---

## 11: cpu_sort uses partial_cmp with NaN fallback instead of total_cmp

+++
status: open
priority: low
kind: task
labels: cleanup, effort:xs
created: 2026-09-02T18:59:04Z
updated: 2026-09-02T19:34:42Z
+++

sort.rs cpu_sort compares depths via partial_cmp(..).unwrap_or(Equal), which silently treats NaN as equal and is more code than f32::total_cmp. Also uses allocating stable sort_by where order of equal keys may not matter.

---

## 12: gpu_sort can overflow next_power_of_two for very large splat counts

+++
status: open
priority: low
kind: bug
labels: bug, effort:xs
created: 2026-09-02T18:59:14Z
updated: 2026-09-02T19:34:42Z
+++

sort.rs gpu_sort calls count.next_power_of_two(), which panics in debug (wraps to 0 in release) for counts above 2^31. No documented limit or guard on splat count.

---

## 13: Unit test parses_three_point_sample fails hard when external sample file is missing

+++
status: closed
priority: low
kind: bug
labels: test, effort:xs
created: 2026-09-02T18:59:14Z
updated: 2026-09-02T19:39:02Z
closed: 2026-09-02T19:39:02Z
+++

cloud.rs test uses a hardcoded absolute path under /Users/schwa/Projects/Scratch/Splats and expect()s the load, so the test suite fails on any machine without that file. The golden test skips gracefully; this one does not.

- `2026-09-02T19:39:02Z`: Test now builds an in-memory ASCII PLY fixture (degree-3 SH) and parses via from_ply_bytes; no external files needed.

---

## 14: Public fallible APIs lack # Errors doc sections

+++
status: open
priority: low
kind: documentation
labels: docs, effort:s
created: 2026-09-02T18:59:14Z
updated: 2026-09-02T19:34:42Z
+++

Renderer::new, Renderer::render, SplatCloud::from_ply_* and with_sh return Result but their doc comments have no # Errors sections describing failure conditions.

---

## 15: Release profile missing lto and codegen-units tuning

+++
status: open
priority: low
kind: task
labels: perf, effort:xs
created: 2026-09-02T18:59:14Z
updated: 2026-09-02T19:34:42Z
+++

Cargo.toml release profile only sets opt-level = 3; no lto or codegen-units = 1, leaving performance on the table for a render-heavy binary.

---

## 16: Readback copies pixels one at a time with put_pixel

+++
status: open
priority: low
kind: enhancement
labels: perf, effort:s
created: 2026-09-02T18:59:14Z
updated: 2026-09-02T19:34:42Z
+++

render.rs converts the mapped readback buffer to an RgbaImage via per-pixel put_pixel in a nested loop. Rows could be memcpy'd into the image buffer instead. Minor at small sizes, scales with resolution.

---

## 17: ply-rs dependency pulls future-incompatible block v0.1.6

+++
status: open
priority: low
kind: task
labels: deps, effort:s
created: 2026-09-02T18:59:14Z
updated: 2026-09-02T19:34:42Z
+++

cargo reports 'block v0.1.6 contains code that will be rejected by a future version of Rust', pulled in transitively via ply-rs 0.1.3. Upstream issue; may eventually break builds.

---

## 18: CLI vector/matrix flags are strings parsed after arg parsing

+++
status: open
priority: low
kind: enhancement
labels: cli, effort:s
created: 2026-09-02T18:59:14Z
updated: 2026-09-02T19:34:42Z
+++

background, camera-position, camera-lookat, camera-matrix, and model-position are String args parsed manually in main/build_camera, so bad values produce runtime errors instead of clap usage errors at parse time.

---
