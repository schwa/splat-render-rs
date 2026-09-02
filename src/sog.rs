//! SOG reader: zip archive of meta.json plus WebP-coded attribute images.

use std::io::{Cursor, Read};

use anyhow::{Context, Result};
use serde::Deserialize;

use crate::cloud::{floats_per_splat, GpuSplat, SplatCloud};

const SH0_COLOR_SCALE: f32 = 0.282_094_8;
const ROTATION_SCALE: f32 = std::f32::consts::SQRT_2;
const SH_ENTRIES_PER_ROW: usize = 64;

#[derive(Deserialize)]
struct Metadata {
    count: usize,
    means: Means,
    scales: Codebook,
    quats: Files,
    sh0: Codebook,
    #[serde(rename = "shN")]
    sh_n: Option<HigherOrder>,
}

#[derive(Deserialize)]
struct Means {
    files: Vec<String>,
    mins: Vec<f32>,
    maxs: Vec<f32>,
}

#[derive(Deserialize)]
struct Codebook {
    files: Vec<String>,
    codebook: Vec<f32>,
}

#[derive(Deserialize)]
struct Files {
    files: Vec<String>,
}

#[derive(Deserialize)]
struct HigherOrder {
    bands: u32,
    codebook: Vec<f32>,
    files: Vec<String>,
}

struct Image {
    pixels: Vec<u8>,
    width: usize,
    height: usize,
}

impl Image {
    fn at(&self, index: usize, component: usize) -> u8 {
        self.pixels[index * 4 + component]
    }
}

fn read_entry(archive: &mut zip::ZipArchive<Cursor<&[u8]>>, name: &str) -> Result<Vec<u8>> {
    let mut file = archive
        .by_name(name)
        .with_context(|| format!("missing SOG entry {name}"))?;
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes)
        .with_context(|| format!("read SOG entry {name}"))?;
    Ok(bytes)
}

fn read_image(archive: &mut zip::ZipArchive<Cursor<&[u8]>>, name: &str) -> Result<Image> {
    let bytes = read_entry(archive, name)?;
    let decoded = image::load_from_memory(&bytes)
        .with_context(|| format!("decode SOG image {name}"))?
        .to_rgba8();
    let (width, height) = decoded.dimensions();
    Ok(Image {
        pixels: decoded.into_raw(),
        width: width as usize,
        height: height as usize,
    })
}

fn coefficient_count(bands: u32) -> usize {
    [0, 3, 8, 15][bands as usize]
}

fn decode_position(raw: u16, min: f32, max: f32) -> f32 {
    let value = min + raw as f32 / 65_535.0 * (max - min);
    let magnitude = value.abs().exp() - 1.0;
    if value < 0.0 {
        -magnitude
    } else {
        magnitude
    }
}

fn decode_rotation(image: &Image, index: usize) -> [f32; 4] {
    let component = |c: usize| (image.at(index, c) as f32 / 255.0 - 0.5) * ROTATION_SCALE;
    let (first, second, third) = (component(0), component(1), component(2));
    let remaining = (1.0 - first * first - second * second - third * third)
        .max(0.0)
        .sqrt();
    let order = image.at(index, 3) as i32 - 252;
    let q = [
        if order == 0 {
            first
        } else if order == 1 {
            remaining
        } else {
            second
        },
        if order <= 1 {
            second
        } else if order == 2 {
            remaining
        } else {
            third
        },
        if order <= 2 { third } else { remaining },
        if order == 0 { remaining } else { first },
    ];
    let len = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
    if len > 0.0 {
        [q[0] / len, q[1] / len, q[2] / len, q[3] / len]
    } else {
        [0.0, 0.0, 0.0, 1.0]
    }
}

/// Decode a SOG archive from bytes.
pub fn from_sog_bytes(data: &[u8]) -> Result<SplatCloud> {
    let mut archive = zip::ZipArchive::new(Cursor::new(data)).context("open SOG zip archive")?;

    let metadata: Metadata = serde_json::from_slice(&read_entry(&mut archive, "meta.json")?)
        .context("parse meta.json")?;
    anyhow::ensure!(
        metadata.means.files.len() >= 2
            && metadata.means.mins.len() == 3
            && metadata.means.maxs.len() == 3
            && !metadata.scales.files.is_empty()
            && !metadata.quats.files.is_empty()
            && !metadata.sh0.files.is_empty(),
        "invalid SOG metadata"
    );

    let means_low = read_image(&mut archive, &metadata.means.files[0])?;
    let means_high = read_image(&mut archive, &metadata.means.files[1])?;
    let scales = read_image(&mut archive, &metadata.scales.files[0])?;
    let quaternions = read_image(&mut archive, &metadata.quats.files[0])?;
    let sh0 = read_image(&mut archive, &metadata.sh0.files[0])?;
    for image in [&means_high, &scales, &quaternions, &sh0] {
        anyhow::ensure!(
            image.width == means_low.width && image.height == means_low.height,
            "SOG image dimensions differ"
        );
    }
    anyhow::ensure!(
        metadata.count <= means_low.width * means_low.height,
        "SOG count exceeds image capacity"
    );

    let higher = match &metadata.sh_n {
        Some(higher) => {
            anyhow::ensure!(
                (1..=3).contains(&higher.bands) && higher.files.len() >= 2,
                "invalid SOG SH metadata"
            );
            let centroids = read_image(&mut archive, &higher.files[0])?;
            let labels = read_image(&mut archive, &higher.files[1])?;
            anyhow::ensure!(
                labels.width == means_low.width && labels.height == means_low.height,
                "SOG label image dimensions differ"
            );
            Some((higher, centroids, labels))
        }
        None => None,
    };

    let coefficients = metadata
        .sh_n
        .as_ref()
        .map_or(0, |h| coefficient_count(h.bands));
    let mut splats = Vec::with_capacity(metadata.count);
    let mut sh = vec![0.0f32; metadata.count * coefficients * 3];
    for index in 0..metadata.count {
        let mut position = [0.0f32; 4];
        for (component, slot) in position.iter_mut().take(3).enumerate() {
            let raw = means_low.at(index, component) as u16
                | (means_high.at(index, component) as u16) << 8;
            *slot = decode_position(
                raw,
                metadata.means.mins[component],
                metadata.means.maxs[component],
            );
        }
        position[3] = 1.0;

        let mut scale = [0.0f32; 4];
        for (component, slot) in scale.iter_mut().take(3).enumerate() {
            let codebook_index = scales.at(index, component) as usize;
            let value = *metadata
                .scales
                .codebook
                .get(codebook_index)
                .context("scale codebook index out of range")?;
            *slot = value.exp();
        }

        let mut color = [0.0f32; 4];
        for (component, slot) in color.iter_mut().take(3).enumerate() {
            let codebook_index = sh0.at(index, component) as usize;
            let value = *metadata
                .sh0
                .codebook
                .get(codebook_index)
                .context("sh0 codebook index out of range")?;
            *slot = (value * SH0_COLOR_SCALE + 0.5).clamp(0.0, 1.0);
        }
        color[3] = sh0.at(index, 3) as f32 / 255.0;

        splats.push(GpuSplat {
            position,
            scale,
            color,
            rotation: decode_rotation(&quaternions, index),
        });

        if let Some((higher, centroids, labels)) = &higher {
            let palette_index = labels.at(index, 0) as usize | (labels.at(index, 1) as usize) << 8;
            let palette_x = palette_index % SH_ENTRIES_PER_ROW * coefficients;
            let palette_y = palette_index / SH_ENTRIES_PER_ROW;
            anyhow::ensure!(
                palette_x + coefficients <= centroids.width && palette_y < centroids.height,
                "SOG SH palette index out of range"
            );
            for coefficient in 0..coefficients {
                let texel = palette_y * centroids.width + palette_x + coefficient;
                for component in 0..3 {
                    let value = *higher
                        .codebook
                        .get(centroids.at(texel, component) as usize)
                        .context("SH codebook index out of range")?;
                    sh[index * coefficients * 3 + coefficient * 3 + component] = value;
                }
            }
        }
    }

    if coefficients == 0 {
        Ok(SplatCloud::from_splats(splats))
    } else {
        let degree = metadata.sh_n.as_ref().map_or(0, |h| h.bands);
        debug_assert_eq!(sh.len(), splats.len() * floats_per_splat(degree));
        SplatCloud::with_sh(splats, degree, sh)
    }
}
