// Spark-style Gaussian splat renderer, ported from SparkSplatRenderShader.metal
// and SparkSplatSupport.h.

struct Uniforms {
    view: mat4x4<f32>,
    projection: mat4x4<f32>,
    model: mat4x4<f32>,
    camera_pos: vec4<f32>,
    drawable_size: vec2<f32>,
    sh_degree: u32,
    srgb_to_linear: u32,
};

struct Splat {
    position: vec4<f32>,
    scale: vec4<f32>,
    color: vec4<f32>,
    rotation: vec4<f32>,  // (x, y, z, w)
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> splats: array<Splat>;
@group(0) @binding(2) var<storage, read> sh_coeffs: array<f32>;
@group(0) @binding(3) var<storage, read> sorted_indices: array<u32>;

const CLIP_XY: f32 = 1.4;
const MAX_PIXEL_RADIUS: f32 = 512.0;
const MAX_STD_DEV: f32 = 2.5;
const MIN_ALPHA: f32 = 0.00784313725;  // 2/255
const BLUR_AMOUNT: f32 = 0.05;

// Spherical harmonics constants.
const SH_C1: f32 = 0.4886025119029199;
const SH_C2_0: f32 = 1.0925484305920792;
const SH_C2_1: f32 = 0.31539156525252005;
const SH_C2_2: f32 = 0.5462742152960396;
const SH_C3_0: f32 = 0.5900435899266435;
const SH_C3_1: f32 = 2.890611442640554;
const SH_C3_2: f32 = 0.4570457994644658;
const SH_C3_3: f32 = 0.3731763325901154;

fn floats_per_splat(degree: u32) -> u32 {
    if (degree == 1u) { return 9u; }
    if (degree == 2u) { return 24u; }
    if (degree == 3u) { return 45u; }
    return 0u;
}

fn evaluate_sh(view_dir: vec3<f32>, splat_index: u32, degree: u32) -> vec3<f32> {
    var result = vec3<f32>(0.0);
    if (degree == 0u) {
        return result;
    }
    let x = view_dir.x;
    let y = view_dir.y;
    let z = view_dir.z;
    let offset = splat_index * floats_per_splat(degree);

    let b1_0 = SH_C1 * y;
    let b1_1 = SH_C1 * z;
    let b1_2 = SH_C1 * x;
    result += b1_0 * vec3<f32>(sh_coeffs[offset + 0u], sh_coeffs[offset + 1u], sh_coeffs[offset + 2u]);
    result += b1_1 * vec3<f32>(sh_coeffs[offset + 3u], sh_coeffs[offset + 4u], sh_coeffs[offset + 5u]);
    result += b1_2 * vec3<f32>(sh_coeffs[offset + 6u], sh_coeffs[offset + 7u], sh_coeffs[offset + 8u]);
    if (degree < 2u) { return result; }

    let xx = x * x; let yy = y * y; let zz = z * z;
    let xy = x * y; let yz = y * z; let xz = x * z;
    let b2_0 = SH_C2_0 * xy;
    let b2_1 = SH_C2_0 * yz;
    let b2_2 = SH_C2_1 * (3.0 * zz - 1.0);
    let b2_3 = SH_C2_0 * xz;
    let b2_4 = SH_C2_2 * (xx - yy);
    result += b2_0 * vec3<f32>(sh_coeffs[offset + 9u], sh_coeffs[offset + 10u], sh_coeffs[offset + 11u]);
    result += b2_1 * vec3<f32>(sh_coeffs[offset + 12u], sh_coeffs[offset + 13u], sh_coeffs[offset + 14u]);
    result += b2_2 * vec3<f32>(sh_coeffs[offset + 15u], sh_coeffs[offset + 16u], sh_coeffs[offset + 17u]);
    result += b2_3 * vec3<f32>(sh_coeffs[offset + 18u], sh_coeffs[offset + 19u], sh_coeffs[offset + 20u]);
    result += b2_4 * vec3<f32>(sh_coeffs[offset + 21u], sh_coeffs[offset + 22u], sh_coeffs[offset + 23u]);
    if (degree < 3u) { return result; }

    let b3_0 = SH_C3_0 * y * (3.0 * xx - yy);
    let b3_1 = SH_C3_1 * xy * z;
    let b3_2 = SH_C3_2 * y * (5.0 * zz - 1.0);
    let b3_3 = SH_C3_3 * z * (5.0 * zz - 3.0);
    let b3_4 = SH_C3_2 * x * (5.0 * zz - 1.0);
    let b3_5 = SH_C3_1 * z * (xx - yy);
    let b3_6 = SH_C3_0 * x * (xx - 3.0 * yy);
    result += b3_0 * vec3<f32>(sh_coeffs[offset + 24u], sh_coeffs[offset + 25u], sh_coeffs[offset + 26u]);
    result += b3_1 * vec3<f32>(sh_coeffs[offset + 27u], sh_coeffs[offset + 28u], sh_coeffs[offset + 29u]);
    result += b3_2 * vec3<f32>(sh_coeffs[offset + 30u], sh_coeffs[offset + 31u], sh_coeffs[offset + 32u]);
    result += b3_3 * vec3<f32>(sh_coeffs[offset + 33u], sh_coeffs[offset + 34u], sh_coeffs[offset + 35u]);
    result += b3_4 * vec3<f32>(sh_coeffs[offset + 36u], sh_coeffs[offset + 37u], sh_coeffs[offset + 38u]);
    result += b3_5 * vec3<f32>(sh_coeffs[offset + 39u], sh_coeffs[offset + 40u], sh_coeffs[offset + 41u]);
    result += b3_6 * vec3<f32>(sh_coeffs[offset + 42u], sh_coeffs[offset + 43u], sh_coeffs[offset + 44u]);
    return result;
}

fn scale_quat_to_matrix(s: vec3<f32>, q: vec4<f32>) -> mat3x3<f32> {
    // q = (x, y, z, w)
    let qx = q.x; let qy = q.y; let qz = q.z; let qw = q.w;
    return mat3x3<f32>(
        vec3<f32>(
            s.x * (1.0 - 2.0 * (qy * qy + qz * qz)),
            s.x * (2.0 * (qx * qy + qw * qz)),
            s.x * (2.0 * (qx * qz - qw * qy)),
        ),
        vec3<f32>(
            s.y * (2.0 * (qx * qy - qw * qz)),
            s.y * (1.0 - 2.0 * (qx * qx + qz * qz)),
            s.y * (2.0 * (qy * qz + qw * qx)),
        ),
        vec3<f32>(
            s.z * (2.0 * (qx * qz + qw * qy)),
            s.z * (2.0 * (qy * qz - qw * qx)),
            s.z * (1.0 - 2.0 * (qx * qx + qy * qy)),
        ),
    );
}

fn rot3(m: mat4x4<f32>) -> mat3x3<f32> {
    return mat3x3<f32>(m[0].xyz, m[1].xyz, m[2].xyz);
}

struct VertexOut {
    @builtin(position) position: vec4<f32>,
    @location(0) splat_uv: vec2<f32>,
    @location(1) rgba: vec4<f32>,
};

// The quad corners in [-1, 1], as a triangle strip.
const QUAD = array<vec2<f32>, 4>(
    vec2<f32>(-1.0, -1.0),
    vec2<f32>( 1.0, -1.0),
    vec2<f32>(-1.0,  1.0),
    vec2<f32>( 1.0,  1.0),
);

@vertex
fn vs_main(
    @builtin(vertex_index) vertex_index: u32,
    @builtin(instance_index) instance_index: u32,
) -> VertexOut {
    var out: VertexOut;
    // Default outside the frustum so an early-out discards the vertex.
    out.position = vec4<f32>(0.0, 0.0, 2.0, 1.0);
    out.splat_uv = vec2<f32>(0.0);
    out.rgba = vec4<f32>(0.0);

    var quad = QUAD;
    let quad_pos = quad[vertex_index];
    let splat_index = sorted_indices[instance_index];
    let splat = splats[splat_index];

    let center = splat.position.xyz;
    let scales = splat.scale.xyz;
    let quat = splat.rotation;
    var rgba = splat.color;

    if (rgba.a < MIN_ALPHA) { return out; }
    if (scales.x == 0.0 && scales.y == 0.0 && scales.z == 0.0) { return out; }

    let world_center = u.model * vec4<f32>(center, 1.0);

    // View-dependent color.
    if (u.sh_degree > 0u) {
        let view_dir = normalize(world_center.xyz - u.camera_pos.xyz);
        let sh = evaluate_sh(view_dir, splat_index, u.sh_degree);
        rgba = vec4<f32>(clamp(rgba.rgb + sh, vec3<f32>(0.0), vec3<f32>(1.0)), rgba.a);
    }

    let view_center4 = u.view * world_center;
    let view_center = view_center4.xyz;

    // Cull behind camera.
    if (view_center.z >= 0.0) { return out; }

    let clip_center = u.projection * vec4<f32>(view_center, 1.0);

    // Near/far cull (loose, matches reference).
    if (abs(clip_center.z) >= clip_center.w) { return out; }

    // XY frustum cull.
    let clip = CLIP_XY * clip_center.w;
    if (abs(clip_center.x) > clip || abs(clip_center.y) > clip) { return out; }

    let local_rs = scale_quat_to_matrix(scales, quat);
    let view_rs = rot3(u.view) * rot3(u.model) * local_rs;
    let cov3d = view_rs * transpose(view_rs);

    let focal = 0.5 * u.drawable_size * vec2<f32>(u.projection[0][0], u.projection[1][1]);

    // Projection Jacobian.
    let inv_z = 1.0 / view_center.z;
    let j1 = focal * inv_z;
    let j2 = -(j1 * view_center.xy) * inv_z;
    let jac = mat3x3<f32>(
        vec3<f32>(j1.x, 0.0, j2.x),
        vec3<f32>(0.0, j1.y, j2.y),
        vec3<f32>(0.0, 0.0, 0.0),
    );

    // cov2D = J^T cov3D J
    let cov = transpose(jac) * cov3d * jac;
    var a = cov[0][0];
    var b = cov[0][1];
    var d = cov[1][1];

    // Anti-aliasing covariance dilation.
    let det_orig = a * d - b * b;
    a += BLUR_AMOUNT;
    d += BLUR_AMOUNT;
    let det = a * d - b * b;
    let blur_adjust = sqrt(max(0.0, det_orig / det));
    rgba.a *= blur_adjust;
    if (rgba.a < MIN_ALPHA) { return out; }

    // Eigendecomposition of the 2D covariance.
    let eigen_avg = 0.5 * (a + d);
    let eigen_delta = sqrt(max(0.0, eigen_avg * eigen_avg - det));
    let eigen1 = eigen_avg + eigen_delta;
    let eigen2 = eigen_avg - eigen_delta;
    var ev1_x: f32;
    if (abs(b) < 0.001) { ev1_x = 1.0; } else { ev1_x = b; }
    let eigen_vec1 = normalize(vec2<f32>(ev1_x, eigen1 - a));
    let eigen_vec2 = vec2<f32>(eigen_vec1.y, -eigen_vec1.x);
    let scale1 = min(MAX_PIXEL_RADIUS, MAX_STD_DEV * sqrt(eigen1));
    let scale2 = min(MAX_PIXEL_RADIUS, MAX_STD_DEV * sqrt(eigen2));
    let major = eigen_vec1 * scale1;
    let minor = eigen_vec2 * scale2;

    let ndc_center = clip_center.xyz / clip_center.w;
    let pixel_offset = quad_pos.x * major + quad_pos.y * minor;
    let ndc_offset = (2.0 / u.drawable_size) * pixel_offset;
    let ndc = ndc_center.xy + ndc_offset;

    out.rgba = rgba;
    out.splat_uv = quad_pos * MAX_STD_DEV;
    out.position = vec4<f32>(ndc * clip_center.w, ndc_center.z * clip_center.w, clip_center.w);
    return out;
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    var rgba = in.rgba;

    if (u.srgb_to_linear != 0u) {
        rgba = vec4<f32>(pow(rgba.rgb, vec3<f32>(2.2)), rgba.a);
    }

    let z = dot(in.splat_uv, in.splat_uv);
    if (z > MAX_STD_DEV * MAX_STD_DEV) {
        discard;
    }

    rgba.a *= exp(-0.5 * z);
    if (rgba.a < MIN_ALPHA) {
        discard;
    }

    // Premultiplied alpha.
    return vec4<f32>(rgba.rgb * rgba.a, rgba.a);
}
