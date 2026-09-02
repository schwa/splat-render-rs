//! Antimatter15 `.splat` reader: 32-byte records, no SH.

use anyhow::Result;

use crate::cloud::{GpuSplat, SplatCloud};

const RECORD_SIZE: usize = 32;

/// Decode `.splat` bytes: per record position 3xf32, scale 3xf32 (linear,
/// already activated), rgba 4xu8, rotation 4xu8 quantized (w, x, y, z).
pub fn from_splat_bytes(bytes: &[u8]) -> Result<SplatCloud> {
    anyhow::ensure!(
        bytes.len().is_multiple_of(RECORD_SIZE),
        ".splat data length {} is not a multiple of {RECORD_SIZE}",
        bytes.len()
    );

    let f32_at = |record: &[u8], offset: usize| {
        f32::from_le_bytes(record[offset..offset + 4].try_into().unwrap())
    };
    let rot = |value: u8| (value as f32 - 128.0) / 128.0;

    let splats = bytes
        .chunks_exact(RECORD_SIZE)
        .map(|r| {
            let (x, y, z) = (f32_at(r, 0), f32_at(r, 4), f32_at(r, 8));
            let (sx, sy, sz) = (f32_at(r, 12), f32_at(r, 16), f32_at(r, 20));
            let color = [
                r[24] as f32 / 255.0,
                r[25] as f32 / 255.0,
                r[26] as f32 / 255.0,
                r[27] as f32 / 255.0,
            ];
            // File order is w, x, y, z.
            let q = [rot(r[29]), rot(r[30]), rot(r[31]), rot(r[28])];
            let len = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
            let q = if len > 0.0 {
                [q[0] / len, q[1] / len, q[2] / len, q[3] / len]
            } else {
                [0.0, 0.0, 0.0, 1.0]
            };
            GpuSplat {
                position: [x, y, z, 1.0],
                scale: [sx, sy, sz, 0.0],
                color,
                rotation: q,
            }
        })
        .collect();

    Ok(SplatCloud::from_splats(splats))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decodes_one_record() {
        let mut record = Vec::new();
        for v in [1.0f32, 2.0, 3.0, 0.5, 0.6, 0.7] {
            record.extend_from_slice(&v.to_le_bytes());
        }
        record.extend_from_slice(&[255, 128, 0, 200]); // rgba
        record.extend_from_slice(&[255, 128, 128, 128]); // w=~1, x=y=z=0

        let cloud = from_splat_bytes(&record).expect("decode");
        assert_eq!(cloud.splats().len(), 1);
        let s = &cloud.splats()[0];
        assert_eq!(s.position[..3], [1.0, 2.0, 3.0]);
        assert_eq!(s.scale[..3], [0.5, 0.6, 0.7]);
        assert!((s.color[0] - 1.0).abs() < 1e-6);
        assert!((s.color[3] - 200.0 / 255.0).abs() < 1e-6);
        assert!((s.rotation[3] - 1.0).abs() < 1e-6, "identity-ish quat");
        assert_eq!(cloud.sh_degree(), 0);
    }

    #[test]
    fn rejects_partial_record() {
        assert!(from_splat_bytes(&[0u8; 33]).is_err());
    }
}
