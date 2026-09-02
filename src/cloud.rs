use anyhow::{Context, Result};
use ply_rs::parser::Parser;
use ply_rs::ply::{DefaultElement, Property};
use std::fs::File;
use std::io::BufReader;

/// SH DC-term normalization constant (l=0 basis function value).
const SH_C0: f32 = 0.282_094_79;

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

impl GpuSplat {
    /// Build a splat from glam types. Values must already be in GPU form:
    /// `scale` activated (`exp`), `color` rgb + opacity all in 0..1, and
    /// `rotation` a normalized quaternion.
    pub fn new(
        position: glam::Vec3,
        scale: glam::Vec3,
        rotation: glam::Quat,
        color: glam::Vec4,
    ) -> Self {
        GpuSplat {
            position: [position.x, position.y, position.z, 1.0],
            scale: [scale.x, scale.y, scale.z, 0.0],
            color: color.to_array(),
            rotation: [rotation.x, rotation.y, rotation.z, rotation.w],
        }
    }
}

pub struct SplatCloud {
    splats: Vec<GpuSplat>,
    /// SH degree, 0..3. 0 means no view-dependent color.
    sh_degree: u32,
    /// Interleaved SH coefficients [c0_r, c0_g, c0_b, c1_r, ...] per splat.
    /// Length = splats.len() * floats_per_splat(sh_degree). Empty if degree 0.
    sh_coefficients: Vec<f32>,
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
    pub fn splats(&self) -> &[GpuSplat] {
        &self.splats
    }

    /// SH degree, 0..3. 0 means no view-dependent color.
    pub fn sh_degree(&self) -> u32 {
        self.sh_degree
    }

    /// Interleaved SH coefficients `[c0_r, c0_g, c0_b, c1_r, ...]` per splat.
    pub fn sh_coefficients(&self) -> &[f32] {
        &self.sh_coefficients
    }

    /// A cloud from GPU-ready splats, with no view-dependent color (SH degree 0).
    ///
    /// Each `GpuSplat` must already be in GPU form: scale activated (`exp`),
    /// opacity in 0..1, color rgb in 0..1, rotation a normalized xyzw quaternion.
    pub fn from_splats(splats: Vec<GpuSplat>) -> SplatCloud {
        SplatCloud {
            splats,
            sh_degree: 0,
            sh_coefficients: Vec::new(),
        }
    }

    /// A cloud from GPU-ready splats plus spherical-harmonics coefficients.
    ///
    /// `sh_degree` must be 1, 2, or 3, and `sh_coefficients` must be interleaved
    /// `[c0_r, c0_g, c0_b, c1_r, ...]` per splat with exactly
    /// `splats.len() * floats_per_splat(sh_degree)` entries.
    ///
    /// # Errors
    ///
    /// Fails when `sh_degree` is outside 1..=3 or `sh_coefficients` has the
    /// wrong length for the splat count and degree.
    pub fn with_sh(
        splats: Vec<GpuSplat>,
        sh_degree: u32,
        sh_coefficients: Vec<f32>,
    ) -> Result<SplatCloud> {
        anyhow::ensure!(
            (1..=3).contains(&sh_degree),
            "sh_degree must be 1, 2, or 3 (use from_splats for degree 0)"
        );
        let expected = splats.len() * floats_per_splat(sh_degree);
        anyhow::ensure!(
            sh_coefficients.len() == expected,
            "sh_coefficients has {} entries, expected {} ({} splats * {} per splat)",
            sh_coefficients.len(),
            expected,
            splats.len(),
            floats_per_splat(sh_degree),
        );
        Ok(SplatCloud {
            splats,
            sh_degree,
            sh_coefficients,
        })
    }

    /// Load a Gaussian splat cloud from a `.ply` file.
    ///
    /// # Errors
    ///
    /// Fails when the file cannot be opened or is not parseable PLY.
    pub fn from_ply_path(path: impl AsRef<std::path::Path>) -> Result<SplatCloud> {
        let path = path.as_ref();
        let file = File::open(path)
            .with_context(|| format!("open {}", path.display()))?;
        Self::from_ply_reader(BufReader::new(file))
    }

    /// Load a Gaussian splat cloud from `.ply` bytes.
    ///
    /// # Errors
    ///
    /// Fails when the bytes are not parseable PLY.
    pub fn from_ply_bytes(bytes: &[u8]) -> Result<SplatCloud> {
        Self::from_ply_reader(std::io::Cursor::new(bytes))
    }

    /// Load a Gaussian splat cloud from any reader over `.ply` data.
    ///
    /// # Errors
    ///
    /// Fails when reading fails or the data is not parseable PLY.
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

    // Property names depend only on the degree; build them once, not per vertex.
    let rest_keys: Vec<String> = (0..3 * cpc).map(|i| format!("f_rest_{i}")).collect();

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
            let r = prop_f32(v, &rest_keys[c]).unwrap_or(0.0);
            let g = prop_f32(v, &rest_keys[cpc + c]).unwrap_or(0.0);
            let b = prop_f32(v, &rest_keys[2 * cpc + c]).unwrap_or(0.0);
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

    /// ASCII PLY with 3 splats carrying full degree-3 SH (45 f_rest_* props).
    fn sample_ply() -> Vec<u8> {
        let mut header = String::from(
            "ply\nformat ascii 1.0\nelement vertex 3\n\
             property float x\nproperty float y\nproperty float z\n\
             property float scale_0\nproperty float scale_1\nproperty float scale_2\n\
             property float opacity\n\
             property float rot_0\nproperty float rot_1\nproperty float rot_2\nproperty float rot_3\n\
             property float f_dc_0\nproperty float f_dc_1\nproperty float f_dc_2\n",
        );
        for i in 0..45 {
            header.push_str(&format!("property float f_rest_{i}\n"));
        }
        header.push_str("end_header\n");
        for i in 0..3 {
            let mut row = format!(
                "{i} 0 0 -1 -2 -3 0.5 2 0 0 0 {} {} {}",
                i as f32 * 0.1,
                0.2,
                -0.3,
            );
            for c in 0..45 {
                row.push_str(&format!(" {}", c as f32 * 0.01));
            }
            row.push('\n');
            header.push_str(&row);
        }
        header.into_bytes()
    }

    #[test]
    fn parses_three_point_sample() {
        let cloud = SplatCloud::from_ply_bytes(&sample_ply()).expect("load sample");
        assert_eq!(cloud.splats.len(), 3);
        // The fixture carries full degree-3 SH.
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
        // SH is re-interleaved from planar; splat 0 channel data starts at coefficient 0.
        assert_eq!(cloud.sh_coefficients[0], 0.0);
    }

    #[test]
    fn constructs_from_splats() {
        let s = GpuSplat::new(
            glam::Vec3::ZERO,
            glam::Vec3::ONE,
            glam::Quat::IDENTITY,
            glam::Vec4::new(1.0, 0.0, 0.0, 1.0),
        );
        let cloud = SplatCloud::from_splats(vec![s; 4]);
        assert_eq!(cloud.splats.len(), 4);
        assert_eq!(cloud.sh_degree, 0);
        assert!(cloud.sh_coefficients.is_empty());
        // Padding slot conventions the shader relies on.
        assert_eq!(cloud.splats[0].position[3], 1.0);
    }

    #[test]
    fn with_sh_validates_length() {
        let splats = vec![GpuSplat::new(
            glam::Vec3::ZERO,
            glam::Vec3::ONE,
            glam::Quat::IDENTITY,
            glam::Vec4::ONE,
        ); 2];
        // degree 1 => 9 floats per splat => 18 total.
        assert!(SplatCloud::with_sh(splats.clone(), 1, vec![0.0; 18]).is_ok());
        assert!(SplatCloud::with_sh(splats.clone(), 1, vec![0.0; 17]).is_err());
        assert!(SplatCloud::with_sh(splats, 0, vec![]).is_err());
    }
}
