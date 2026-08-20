use anyhow::{Context, Result};
use ply_rs::parser::Parser;
use ply_rs::ply::{DefaultElement, Property};
use std::fs::File;
use std::io::BufReader;

/// SH DC-term normalization constant (l=0 basis function value).
const SH_C0: f32 = 0.28209479177387814;

/// A single Gaussian, laid out for the GPU (64 bytes, all vec4-aligned).
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuSplat {
    /// World-space center; w unused.
    pub position: [f32; 4],
    /// Per-axis scale (activated: exp); w unused.
    pub scale: [f32; 4],
    /// Base color rgb + opacity a, all 0..1.
    pub color: [f32; 4],
    /// Rotation quaternion in (x, y, z, w) order, normalized.
    pub rotation: [f32; 4],
}

pub struct SplatCloud {
    pub splats: Vec<GpuSplat>,
    /// SH degree, 0..3. 0 means no view-dependent color.
    pub sh_degree: u32,
    /// Interleaved SH coefficients [c0_r, c0_g, c0_b, c1_r, ...] per splat.
    /// Length = splats.len() * floats_per_splat(sh_degree). Empty if degree 0.
    pub sh_coefficients: Vec<f32>,
}

/// Floats of SH coefficient data per splat, excluding the DC term.
pub fn floats_per_splat(degree: u32) -> usize {
    match degree {
        1 => 9,
        2 => 24,
        3 => 45,
        _ => 0,
    }
}

/// Coefficients per channel above DC, for a given degree.
fn coeffs_per_channel(degree: u32) -> usize {
    match degree {
        1 => 3,
        2 => 8,
        3 => 15,
        _ => 0,
    }
}

fn prop_f32(el: &DefaultElement, key: &str) -> Option<f32> {
    match el.get(key)? {
        Property::Float(v) => Some(*v),
        Property::Double(v) => Some(*v as f32),
        Property::Int(v) => Some(*v as f32),
        Property::UInt(v) => Some(*v as f32),
        Property::UChar(v) => Some(*v as f32),
        _ => None,
    }
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

impl SplatCloud {
    /// Load a Gaussian splat cloud from a `.ply` file.
    pub fn from_ply_path(path: impl AsRef<std::path::Path>) -> Result<SplatCloud> {
        let path = path.as_ref();
        let file = File::open(path)
            .with_context(|| format!("open {}", path.display()))?;
        Self::from_ply_reader(BufReader::new(file))
    }

    /// Load a Gaussian splat cloud from `.ply` bytes.
    pub fn from_ply_bytes(bytes: &[u8]) -> Result<SplatCloud> {
        Self::from_ply_reader(std::io::Cursor::new(bytes))
    }

    /// Load a Gaussian splat cloud from any reader over `.ply` data.
    pub fn from_ply_reader<R: std::io::Read>(mut reader: R) -> Result<SplatCloud> {
        let parser = Parser::<DefaultElement>::new();
        let ply = parser.read_ply(&mut reader).context("parse PLY")?;
        Ok(cloud_from_ply(ply))
    }
}

fn cloud_from_ply(ply: ply_rs::ply::Ply<DefaultElement>) -> SplatCloud {

    // The vertex element holds the splats; pick the largest element as a fallback.
    let empty: Vec<DefaultElement> = Vec::new();
    let verts = ply
        .payload
        .get("vertex")
        .or_else(|| ply.payload.values().max_by_key(|v| v.len()))
        .unwrap_or(&empty);

    let degree = detect_sh_degree(verts.first());
    let fps = floats_per_splat(degree);
    let cpc = coeffs_per_channel(degree);

    let mut splats = Vec::with_capacity(verts.len());
    let mut sh = Vec::with_capacity(verts.len() * fps);

    for v in verts {
        let x = prop_f32(v, "x").unwrap_or(0.0);
        let y = prop_f32(v, "y").unwrap_or(0.0);
        let z = prop_f32(v, "z").unwrap_or(0.0);

        // Scale activation: exp.
        let sx = prop_f32(v, "scale_0").or_else(|| prop_f32(v, "sx")).unwrap_or(0.0);
        let sy = prop_f32(v, "scale_1").or_else(|| prop_f32(v, "sy")).unwrap_or(0.0);
        let sz = prop_f32(v, "scale_2").or_else(|| prop_f32(v, "sz")).unwrap_or(0.0);

        // Color: f_dc_* through SH_C0, or raw red/green/blue (0..255).
        let (cr, cg, cb) = if let (Some(a), Some(b), Some(c)) =
            (prop_f32(v, "f_dc_0"), prop_f32(v, "f_dc_1"), prop_f32(v, "f_dc_2"))
        {
            (
                (a * SH_C0 + 0.5).clamp(0.0, 1.0),
                (b * SH_C0 + 0.5).clamp(0.0, 1.0),
                (c * SH_C0 + 0.5).clamp(0.0, 1.0),
            )
        } else {
            (
                prop_f32(v, "red").unwrap_or(255.0) / 255.0,
                prop_f32(v, "green").unwrap_or(255.0) / 255.0,
                prop_f32(v, "blue").unwrap_or(255.0) / 255.0,
            )
        };

        // Opacity activation: sigmoid, or raw alpha.
        let opacity = if let Some(o) = prop_f32(v, "opacity") {
            sigmoid(o)
        } else {
            prop_f32(v, "alpha").unwrap_or(1.0)
        }
        .clamp(0.0, 1.0);

        // Rotation: rot_0=w, rot_1=x, rot_2=y, rot_3=z; normalize.
        let rw = prop_f32(v, "rot_0").unwrap_or(1.0);
        let rx = prop_f32(v, "rot_1").unwrap_or(0.0);
        let ry = prop_f32(v, "rot_2").unwrap_or(0.0);
        let rz = prop_f32(v, "rot_3").unwrap_or(0.0);
        let len = (rw * rw + rx * rx + ry * ry + rz * rz).sqrt();
        let (rx, ry, rz, rw) = if len > 0.0 {
            (rx / len, ry / len, rz / len, rw / len)
        } else {
            (0.0, 0.0, 0.0, 1.0)
        };

        splats.push(GpuSplat {
            position: [x, y, z, 1.0],
            scale: [sx.exp(), sy.exp(), sz.exp(), 0.0],
            color: [cr, cg, cb, opacity],
            rotation: [rx, ry, rz, rw],
        });

        // SH coefficients: PLY stores f_rest planar (all R, then G, then B).
        // Output interleaved [c0_r, c0_g, c0_b, c1_r, ...].
        for c in 0..cpc {
            let r = prop_f32(v, &format!("f_rest_{}", c)).unwrap_or(0.0);
            let g = prop_f32(v, &format!("f_rest_{}", cpc + c)).unwrap_or(0.0);
            let b = prop_f32(v, &format!("f_rest_{}", 2 * cpc + c)).unwrap_or(0.0);
            sh.push(r);
            sh.push(g);
            sh.push(b);
        }
    }

    SplatCloud {
        splats,
        sh_degree: degree,
        sh_coefficients: sh,
    }
}

/// Detect SH degree from the highest `f_rest_N` property present.
fn detect_sh_degree(sample: Option<&DefaultElement>) -> u32 {
    let Some(el) = sample else {
        return 0;
    };
    if el.contains_key("f_rest_44") {
        3
    } else if el.contains_key("f_rest_23") {
        2
    } else if el.contains_key("f_rest_8") {
        1
    } else {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str =
        "/Users/schwa/Projects/Scratch/Splats/inria/test-splat.3-points-from-train.ply";

    #[test]
    fn parses_three_point_sample() {
        let cloud = SplatCloud::from_ply_path(SAMPLE).expect("load sample");
        assert_eq!(cloud.splats.len(), 3);
        // This file carries full degree-3 SH.
        assert_eq!(cloud.sh_degree, 3);
        assert_eq!(cloud.sh_coefficients.len(), 3 * floats_per_splat(3));

        // Activations must produce valid ranges.
        for s in &cloud.splats {
            for ch in 0..4 {
                assert!(s.color[ch] >= 0.0 && s.color[ch] <= 1.0, "color in range");
            }
            for ax in 0..3 {
                assert!(s.scale[ax] > 0.0, "scale is exp() so strictly positive");
            }
            let q = s.rotation;
            let n = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
            assert!((n - 1.0).abs() < 1e-3, "rotation normalized");
        }
    }
}
