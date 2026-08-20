//! Golden-image regression tests.
//!
//! Renders known splat files and compares against goldens produced by the
//! MetalSprocketsGaussianSplats reference renderer (--renderer spark --sort gpu).
//! Skips gracefully when the (large, external) source .ply files are absent.

use std::path::Path;
use std::process::Command;

const SPLATS_DIR: &str = "/Users/schwa/Projects/Scratch/Splats";

struct Case {
    ply: &'static str,
    camera_position: &'static str,
    camera_lookat: &'static str,
    golden: &'static str,
}

const CASES: &[Case] = &[
    Case {
        ply: "work/splat0.ply",
        camera_position: "0,0,3",
        camera_lookat: "0,0,0",
        golden: "tests/goldens/splat0.png",
    },
    Case {
        ply: "sparkjs/furry-logo-pedestal.ply",
        camera_position: "0,0,4",
        camera_lookat: "0,0,0",
        golden: "tests/goldens/furry.png",
    },
];

fn rmse(a: &image::RgbaImage, b: &image::RgbaImage) -> f64 {
    assert_eq!(a.dimensions(), b.dimensions(), "image sizes differ");
    let mut sum = 0.0f64;
    let mut n = 0.0f64;
    for (pa, pb) in a.pixels().zip(b.pixels()) {
        for c in 0..4 {
            let d = pa.0[c] as f64 - pb.0[c] as f64;
            sum += d * d;
            n += 1.0;
        }
    }
    (sum / n).sqrt() / 255.0
}

#[test]
fn goldens_match_reference() {
    let bin = env!("CARGO_BIN_EXE_splat-rs");
    let mut ran = 0;

    for case in CASES {
        let ply = format!("{SPLATS_DIR}/{}", case.ply);
        if !Path::new(&ply).exists() {
            eprintln!("skip {}: source not found", case.ply);
            continue;
        }
        let golden_path = case.golden;
        assert!(Path::new(golden_path).exists(), "missing golden {golden_path}");

        let out = std::env::temp_dir().join(format!(
            "splat-rs-test-{}.png",
            case.ply.replace('/', "_")
        ));
        let status = Command::new(bin)
            .args([
                &ply,
                "--camera-position",
                case.camera_position,
                "--camera-lookat",
                case.camera_lookat,
                "--width",
                "256",
                "--height",
                "192",
                "--sort",
                "cpu",
                "--output",
                out.to_str().unwrap(),
            ])
            .status()
            .expect("run splat-rs");
        assert!(status.success(), "render failed for {}", case.ply);

        let mine = image::open(&out).expect("open render").to_rgba8();
        let golden = image::open(golden_path).expect("open golden").to_rgba8();
        let err = rmse(&mine, &golden);
        eprintln!("{}: RMSE {err:.5}", case.ply);
        assert!(err < 0.02, "{} diverged from golden: RMSE {err}", case.ply);
        ran += 1;
    }

    if ran == 0 {
        eprintln!("no source .ply files present; golden test skipped");
    }
}
