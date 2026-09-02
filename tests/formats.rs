//! Format-reader tests against local sample files.
//!
//! Each test skips gracefully when the (external) sample directory is absent.

use std::path::{Path, PathBuf};

use splat_render_rs::SplatCloud;

const SAMPLES_DIR: &str = "/Users/schwa/Shared/Projects/Scratch/SplatSamples";

fn sample(relative: &str) -> Option<PathBuf> {
    let path = Path::new(SAMPLES_DIR).join(relative);
    if path.exists() {
        Some(path)
    } else {
        eprintln!("skip: {relative} not found");
        None
    }
}

fn assert_valid(cloud: &SplatCloud, name: &str) {
    assert!(!cloud.splats().is_empty(), "{name}: no splats");
    assert_eq!(
        cloud.sh_coefficients().len(),
        cloud.splats().len() * splat_render_rs::floats_per_splat(cloud.sh_degree()),
        "{name}: SH length mismatch"
    );
    for (index, splat) in cloud.splats().iter().enumerate() {
        for channel in 0..4 {
            assert!(
                (0.0..=1.0).contains(&splat.color[channel]),
                "{name}[{index}]: color out of range"
            );
        }
        for axis in 0..3 {
            assert!(
                splat.scale[axis] > 0.0 && splat.scale[axis].is_finite(),
                "{name}[{index}]: bad scale"
            );
            assert!(
                splat.position[axis].is_finite(),
                "{name}[{index}]: bad position"
            );
        }
        let q = splat.rotation;
        let norm = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
        assert!(
            (norm - 1.0).abs() < 0.02,
            "{name}[{index}]: rotation not normalized ({norm})"
        );
    }
}

#[test]
fn reads_antimatter15_splat() {
    let Some(path) = sample("Test Splats/6Splats.splat") else {
        return;
    };
    let cloud = SplatCloud::from_path(&path).expect("read .splat");
    assert_eq!(cloud.splats().len(), 6);
    assert_valid(&cloud, "6Splats.splat");
}

#[test]
fn reads_spz_v3() {
    let Some(path) = sample("Lion/lion.v3.spz") else {
        return;
    };
    let cloud = SplatCloud::from_path(&path).expect("read v3 spz");
    assert_valid(&cloud, "lion.v3.spz");
}

#[test]
fn reads_spz_v4() {
    let Some(path) = sample("Tomatoes/tomatoes.v4.spz") else {
        return;
    };
    let cloud = SplatCloud::from_path(&path).expect("read v4 spz");
    assert_valid(&cloud, "tomatoes.v4.spz");
}

#[test]
fn reads_sog() {
    let Some(path) = sample("Test Splats/RainbowRing.sog") else {
        return;
    };
    let cloud = SplatCloud::from_path(&path).expect("read sog");
    assert_valid(&cloud, "RainbowRing.sog");
}

/// RainbowRing exists in both formats; the decoded clouds should agree
/// loosely (both formats quantize differently).
#[test]
fn sog_and_splat_agree_on_rainbow_ring() {
    let (Some(sog_path), Some(splat_path)) = (
        sample("Test Splats/RainbowRing.sog"),
        sample("Test Splats/RainbowRing.splat"),
    ) else {
        return;
    };
    let sog = SplatCloud::from_path(&sog_path).expect("read sog");
    let splat = SplatCloud::from_path(&splat_path).expect("read splat");
    assert_eq!(
        sog.splats().len(),
        splat.splats().len(),
        "splat counts differ"
    );

    // Compare centroids of positions; quantization differs but the cloud
    // shape must match.
    let centroid = |cloud: &SplatCloud| {
        let n = cloud.splats().len() as f32;
        let mut sum = [0.0f32; 3];
        for s in cloud.splats() {
            for (axis, total) in sum.iter_mut().enumerate() {
                *total += s.position[axis];
            }
        }
        [sum[0] / n, sum[1] / n, sum[2] / n]
    };
    let a = centroid(&sog);
    let b = centroid(&splat);
    for axis in 0..3 {
        assert!(
            (a[axis] - b[axis]).abs() < 0.05,
            "centroid differs on axis {axis}: {a:?} vs {b:?}"
        );
    }
}
