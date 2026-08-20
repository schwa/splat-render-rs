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
status: new
priority: low
kind: none
labels: io
created: 2026-08-20T16:58:49Z
+++

load_ply handles PLY (ascii + binary_little_endian) only. The reference SplatLoader reads .splat, .spz, and .sog (the latter GPU-decoded). Add these formats.

---

## 3: Splats stored as f32 (64 bytes), not f16-packed like the reference

+++
status: new
priority: low
kind: none
labels: perf
created: 2026-08-20T16:58:49Z
+++

GpuSplat is 64 bytes of f32. The reference SparkSplat is a 32-byte half-precision struct. Packing to f16 halves buffer size and bandwidth. Verify visual parity after packing.

---

## 4: Unported reference CLI flags: config JSON, statistics, label, camera-rotation, capture, reveal

+++
status: new
priority: low
kind: none
labels: cli
created: 2026-08-20T16:58:49Z
+++

splat-rs implements core render flags. Reference also has --config, --statistics/--stats, --label, --camera-rotation, --capture, --reveal, --renderer selection, point tuning.

---
