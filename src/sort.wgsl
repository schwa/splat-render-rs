// GPU depth sort via bitonic sort.
//
// Not the reference's tile radix (SplatGPUSort.metal) — a bitonic sort is far
// simpler and needs no atomics or subgroup ops, which WGSL lacks portably. It
// sorts (key, splat_index) records by view-space depth so an ascending sort
// yields farthest-first (back-to-front) draw order.
//
// All three entry points share one bind group. The step params binding is
// dynamic-offset so each bitonic pass supplies its own (k, j).

struct Record {
    key: u32,
    index: u32,
};

struct BuildParams {
    model_view: mat4x4<f32>,
    count: u32,
    padded: u32,
    _pad0: u32,
    _pad1: u32,
};

struct StepParams {
    k: u32,
    j: u32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<storage, read> splats_words: array<u32>;
@group(0) @binding(1) var<storage, read_write> records: array<Record>;
@group(0) @binding(2) var<uniform> build: BuildParams;
@group(0) @binding(3) var<uniform> step: StepParams;
@group(0) @binding(4) var<storage, read_write> indices: array<u32>;

// Map a float to a u32 whose unsigned order matches ascending float order.
fn depth_key(vz: f32) -> u32 {
    let b = bitcast<u32>(vz);
    if ((b & 0x80000000u) != 0u) {
        return ~b;
    }
    return b | 0x80000000u;
}

// splats_words aliases the packed Splat buffer (24-byte stride = 6 u32 words):
// word 0 holds f16 x|y, word 1's low half holds f16 z.
@compute @workgroup_size(256)
fn build_records(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= build.padded) {
        return;
    }
    if (i >= build.count) {
        // Padding sorts to the end (nearest side) and is never drawn.
        records[i] = Record(0xFFFFFFFFu, 0u);
        return;
    }
    let xy = unpack2x16float(splats_words[i * 6u]);
    let z = unpack2x16float(splats_words[i * 6u + 1u]).x;
    let view_pos = build.model_view * vec4<f32>(xy, z, 1.0);
    records[i] = Record(depth_key(view_pos.z), i);
}

@compute @workgroup_size(256)
fn bitonic_step(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let partner = i ^ step.j;
    // Each compare handled once, by the lower index.
    if (partner <= i) {
        return;
    }
    let a = records[i];
    let b = records[partner];
    // Ascending block when the k-bit of i is 0, descending otherwise.
    let ascending = (i & step.k) == 0u;
    let swap = select((a.key < b.key), (a.key > b.key), ascending);
    if (swap) {
        records[i] = b;
        records[partner] = a;
    }
}

@compute @workgroup_size(256)
fn extract_indices(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= build.count) {
        return;
    }
    indices[i] = records[i].index;
}
