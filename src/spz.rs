//! SPZ reader: v2/v3 (gzip container) and v4 (zstd streams).

use std::io::Read;

use anyhow::{Context, Result};

use crate::cloud::{floats_per_splat, GpuSplat, SplatCloud};

const MAGIC: u32 = 0x5053_474e; // "NGSP"
const SH_C0: f32 = 0.282_094_8;

struct Container {
    payload: Vec<u8>,
    version: u32,
    splat_count: usize,
    sh_degree: u8,
    fractional_bits: u8,
    header_size: usize,
}

struct Layout {
    positions: usize,
    alphas: usize,
    colors: usize,
    scales: usize,
    rotations: usize,
    sh: usize,
    coefficients_per_channel: usize,
    rotation_bytes: usize,
}

fn u32_le(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap())
}

fn u64_le(bytes: &[u8], offset: usize) -> u64 {
    u64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap())
}

fn coefficients_per_channel(degree: u8) -> usize {
    [0, 3, 8, 15, 24][degree as usize]
}

impl Container {
    fn parse(data: &[u8]) -> Result<Container> {
        let container = if data.len() >= 4 && u32_le(data, 0) == MAGIC {
            Self::parse_v4(data)?
        } else {
            Self::parse_gzip(data)?
        };
        anyhow::ensure!(
            container.sh_degree <= 4,
            "invalid SH degree {}",
            container.sh_degree
        );
        anyhow::ensure!(
            container.fractional_bits <= 23,
            "invalid fractional bits {}",
            container.fractional_bits
        );
        Ok(container)
    }

    fn parse_gzip(data: &[u8]) -> Result<Container> {
        let mut payload = Vec::new();
        flate2::read::GzDecoder::new(data)
            .read_to_end(&mut payload)
            .context("gzip decompress SPZ")?;
        anyhow::ensure!(payload.len() >= 16, "SPZ header truncated");
        anyhow::ensure!(u32_le(&payload, 0) == MAGIC, "bad SPZ magic");
        let version = u32_le(&payload, 4);
        anyhow::ensure!(
            version == 2 || version == 3,
            "unsupported SPZ version {version}"
        );
        let splat_count = u32_le(&payload, 8) as usize;
        let sh_degree = payload[12];
        let fractional_bits = payload[13];
        Ok(Container {
            payload,
            version,
            splat_count,
            sh_degree,
            fractional_bits,
            header_size: 16,
        })
    }

    fn parse_v4(data: &[u8]) -> Result<Container> {
        anyhow::ensure!(data.len() >= 32, "SPZ v4 header truncated");
        let version = u32_le(data, 4);
        anyhow::ensure!(version == 4, "unsupported SPZ version {version}");
        let splat_count = u32_le(data, 8) as usize;
        let sh_degree = data[12];
        anyhow::ensure!(sh_degree <= 4, "invalid SH degree {sh_degree}");
        let stream_count = data[15] as usize;
        let coefficient_count = coefficients_per_channel(sh_degree);
        let expected_sizes: Vec<usize> = [
            splat_count * 9,
            splat_count,
            splat_count * 3,
            splat_count * 3,
            splat_count * 4,
            splat_count * coefficient_count * 3,
        ]
        .into_iter()
        .filter(|&s| s > 0)
        .collect();
        anyhow::ensure!(
            stream_count == expected_sizes.len(),
            "unexpected SPZ stream count {stream_count}"
        );
        let table_offset = u32_le(data, 16) as usize;
        anyhow::ensure!(
            table_offset >= 32 && table_offset + stream_count * 16 <= data.len(),
            "bad SPZ stream table"
        );

        let mut source_offset = table_offset + stream_count * 16;
        let mut payload = Vec::new();
        for (index, &expected) in expected_sizes.iter().enumerate() {
            let entry = table_offset + index * 16;
            let compressed = usize::try_from(u64_le(data, entry)).context("stream size")?;
            let uncompressed = usize::try_from(u64_le(data, entry + 8)).context("stream size")?;
            anyhow::ensure!(uncompressed == expected, "SPZ stream size mismatch");
            anyhow::ensure!(
                source_offset + compressed <= data.len(),
                "SPZ stream out of bounds"
            );
            let mut decoder = ruzstd::decoding::StreamingDecoder::new(
                &data[source_offset..source_offset + compressed],
            )
            .context("zstd stream")?;
            let start = payload.len();
            decoder
                .read_to_end(&mut payload)
                .context("zstd decompress SPZ stream")?;
            anyhow::ensure!(
                payload.len() - start == uncompressed,
                "SPZ stream decompressed to wrong size"
            );
            source_offset += compressed;
        }
        anyhow::ensure!(
            source_offset == data.len(),
            "trailing bytes after SPZ streams"
        );
        Ok(Container {
            payload,
            version,
            splat_count,
            sh_degree,
            fractional_bits: data[13],
            header_size: 0,
        })
    }
}

impl Layout {
    fn validated(container: &Container) -> Result<Layout> {
        let n = container.splat_count;
        let coefficients_per_channel = coefficients_per_channel(container.sh_degree);
        let rotation_bytes = if container.version >= 3 { 4 } else { 3 };
        let positions = container.header_size;
        let alphas = positions + n * 9;
        let colors = alphas + n;
        let scales = colors + n * 3;
        let rotations = scales + n * 3;
        let sh = rotations + n * rotation_bytes;
        anyhow::ensure!(
            container.payload.len() >= sh + n * coefficients_per_channel * 3,
            "SPZ payload truncated"
        );
        Ok(Layout {
            positions,
            alphas,
            colors,
            scales,
            rotations,
            sh,
            coefficients_per_channel,
            rotation_bytes,
        })
    }
}

fn unpack_rotation_v2(bytes: &[u8]) -> [f32; 4] {
    let x = bytes[0] as f32 / 127.5 - 1.0;
    let y = bytes[1] as f32 / 127.5 - 1.0;
    let z = bytes[2] as f32 / 127.5 - 1.0;
    let w = (1.0 - (x * x + y * y + z * z)).max(0.0).sqrt();
    [x, y, z, w]
}

fn unpack_rotation_v3(bytes: &[u8]) -> [f32; 4] {
    let mut packed = u32_le(bytes, 0);
    let largest = (packed >> 30) as usize;
    let mut rotation = [0.0f32; 4];
    let mut sum = 0.0f32;
    for component in (0..4).rev() {
        if component == largest {
            continue;
        }
        let magnitude = (packed & 511) as f32;
        let sign = if (packed >> 9) & 1 == 0 { 1.0 } else { -1.0 };
        let value = std::f32::consts::FRAC_1_SQRT_2 * magnitude / 511.0 * sign;
        rotation[component] = value;
        sum += value * value;
        packed >>= 10;
    }
    rotation[largest] = (1.0 - sum).max(0.0).sqrt();
    rotation
}

/// Decode SPZ bytes (v2/v3 gzip container or v4 zstd streams).
pub fn from_spz_bytes(data: &[u8]) -> Result<SplatCloud> {
    let container = Container::parse(data)?;
    let layout = Layout::validated(&container)?;
    anyhow::ensure!(
        container.sh_degree <= 3,
        "SH degree {} not supported by this renderer",
        container.sh_degree
    );

    let position_scale = (1u32 << container.fractional_bits) as f32;
    let payload = &container.payload;
    let n = container.splat_count;
    let coefficients = layout.coefficients_per_channel;

    let mut splats = Vec::with_capacity(n);
    let mut sh = Vec::with_capacity(n * coefficients * 3);
    for index in 0..n {
        let mut position = [0.0f32; 4];
        for (component, slot) in position.iter_mut().take(3).enumerate() {
            let offset = layout.positions + index * 9 + component * 3;
            let mut bits = payload[offset] as u32
                | (payload[offset + 1] as u32) << 8
                | (payload[offset + 2] as u32) << 16;
            if bits & 0x80_0000 != 0 {
                bits |= 0xff00_0000;
            }
            *slot = bits as i32 as f32 / position_scale;
        }
        position[3] = 1.0;

        let mut color = [0.0f32; 4];
        for component in 0..3 {
            let coefficient =
                (payload[layout.colors + index * 3 + component] as f32 / 255.0 - 0.5) / 0.15;
            color[component] = (coefficient * SH_C0 + 0.5).clamp(0.0, 1.0);
        }
        color[3] = payload[layout.alphas + index] as f32 / 255.0;

        let mut scale = [0.0f32; 4];
        for component in 0..3 {
            scale[component] =
                (payload[layout.scales + index * 3 + component] as f32 / 16.0 - 10.0).exp();
        }

        let rotation_offset = layout.rotations + index * layout.rotation_bytes;
        let rotation_bytes = &payload[rotation_offset..rotation_offset + layout.rotation_bytes];
        let rotation = if container.version >= 3 {
            unpack_rotation_v3(rotation_bytes)
        } else {
            unpack_rotation_v2(rotation_bytes)
        };

        splats.push(GpuSplat {
            position,
            scale,
            color,
            rotation,
        });

        // SPZ stores SH coefficient-major per splat, matching our interleaving.
        let base = layout.sh + index * coefficients * 3;
        for value_index in 0..coefficients * 3 {
            sh.push((payload[base + value_index] as f32 - 128.0) / 128.0);
        }
    }

    if coefficients == 0 {
        Ok(SplatCloud::from_splats(splats))
    } else {
        let degree = match coefficients {
            3 => 1,
            8 => 2,
            _ => 3,
        };
        debug_assert_eq!(sh.len(), splats.len() * floats_per_splat(degree));
        SplatCloud::with_sh(splats, degree, sh)
    }
}
