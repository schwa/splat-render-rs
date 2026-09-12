//! CLI front-end for the `splat_render_rs` library.

use std::fs::File;
use std::io::{BufWriter, Write as _};
use std::path::{Path, PathBuf};
use std::str::FromStr;

use anyhow::{Context, Result};
use clap::{Parser, ValueEnum};
use glam::{Mat4, Vec3};
use serde::{Deserialize, Serialize};
use splat_render_rs::{Camera, Perspective, RenderSettings, Renderer, SortMethod, SplatCloud};

#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
enum SortArg {
    Cpu,
    Gpu,
}

impl From<SortArg> for SortMethod {
    fn from(s: SortArg) -> Self {
        match s {
            SortArg::Cpu => SortMethod::Cpu,
            SortArg::Gpu => SortMethod::Gpu,
        }
    }
}

fn parse_floats(s: &str, n: usize) -> Result<Vec<f32>, String> {
    let v: Vec<f32> = s
        .split(',')
        .map(|p| p.trim().parse::<f32>())
        .collect::<Result<_, _>>()
        .map_err(|_| format!("expected {n} comma-separated numbers"))?;
    if v.len() != n {
        return Err(format!("expected {n} values, got {}", v.len()));
    }
    Ok(v)
}

/// `x,y,z` vector CLI argument.
#[derive(Copy, Clone, Debug)]
struct Vec3Arg(Vec3);

impl FromStr for Vec3Arg {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        let v = parse_floats(s, 3)?;
        Ok(Vec3Arg(Vec3::new(v[0], v[1], v[2])))
    }
}

/// 16 comma-separated column-major values.
#[derive(Copy, Clone, Debug)]
struct Mat4Arg(Mat4);

impl FromStr for Mat4Arg {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        let v = parse_floats(s, 16)?;
        Ok(Mat4Arg(Mat4::from_cols_slice(&v)))
    }
}

/// `r,g,b,a` color in 0..1.
#[derive(Copy, Clone, Debug)]
struct RgbaArg([f32; 4]);

impl FromStr for RgbaArg {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        let v = parse_floats(s, 4)?;
        Ok(RgbaArg([v[0], v[1], v[2], v[3]]))
    }
}

/// Render a 3D Gaussian Splat file to a PNG.
#[derive(Parser, Debug)]
#[command(
    name = "splat-render-rs",
    about = "Render Gaussian splat files to PNG images"
)]
struct Args {
    /// Path to the splat file (.ply, .splat, .spz, .sog) to render.
    splat: String,

    /// Output PNG file path.
    #[arg(long, default_value = "output.png")]
    output: String,

    #[arg(long, default_value_t = 1024)]
    width: u32,

    #[arg(long, default_value_t = 768)]
    height: u32,

    /// Background color in RGBA (e.g. 0,0,0,1).
    #[arg(long, default_value = "0,0,0,1")]
    background: RgbaArg,

    /// Camera position in x,y,z.
    #[arg(long)]
    camera_position: Option<Vec3Arg>,

    /// Camera look-at target in x,y,z.
    #[arg(long)]
    camera_lookat: Option<Vec3Arg>,

    /// Camera-to-world matrix, 16 comma-separated column-major values. Overrides other camera flags.
    #[arg(long, conflicts_with_all = ["camera_position", "camera_lookat"])]
    camera_matrix: Option<Mat4Arg>,

    /// Model position in x,y,z.
    #[arg(long)]
    model_position: Option<Vec3Arg>,

    /// Vertical field of view in degrees.
    #[arg(long, default_value_t = 60.0)]
    projection_fov: f32,

    #[arg(long, default_value_t = 0.1)]
    near: f32,

    #[arg(long, default_value_t = 100.0)]
    far: f32,

    /// Convert sRGB to linear in the fragment shader.
    #[arg(long)]
    srgb_to_linear: bool,

    /// Depth sort method.
    #[arg(long, default_value = "gpu")]
    sort: SortArg,

    /// Batch mode: JSON file with an array of camera poses to render.
    #[arg(long, conflicts_with_all = ["camera_position", "camera_lookat", "camera_matrix", "output"])]
    poses: Option<PathBuf>,

    /// Batch mode: directory for output PNGs (default: current directory).
    #[arg(long, requires = "poses")]
    output_dir: Option<PathBuf>,

    /// Batch mode: write per-pose alpha stats as JSON lines to this path.
    #[arg(long, requires = "poses")]
    stats: Option<PathBuf>,

    /// Write splat positions as raw little-endian f32 x,y,z triples and exit.
    #[arg(long)]
    dump_positions: Option<PathBuf>,
}

fn build_camera(args: &Args) -> Camera {
    let aspect = args.width as f32 / args.height as f32;
    let projection = Perspective::new(
        args.projection_fov.to_radians(),
        aspect,
        args.near,
        args.far,
    );
    let up = Vec3::Y;

    // Priority: matrix > lookat > position > default.
    if let Some(Mat4Arg(c2w)) = args.camera_matrix {
        return Camera::from_camera_to_world(c2w, projection);
    }
    if let Some(Vec3Arg(target)) = args.camera_lookat {
        let eye = match args.camera_position {
            Some(Vec3Arg(p)) => p,
            None => Vec3::new(0.0, 0.0, 1.5),
        };
        return Camera::look_at(eye, target, up, projection);
    }
    if let Some(Vec3Arg(eye)) = args.camera_position {
        return Camera::from_camera_to_world(Mat4::from_translation(eye), projection);
    }
    Camera::look_at(Vec3::new(0.0, 0.0, 1.5), Vec3::ZERO, up, projection)
}

/// One camera pose in a batch `--poses` file.
#[derive(Deserialize, Debug)]
struct Pose {
    /// Output file stem; defaults to the pose index.
    name: Option<String>,
    /// Camera-to-world matrix, 16 column-major values. Overrides position/lookat.
    matrix: Option<[f32; 16]>,
    position: Option<[f32; 3]>,
    lookat: Option<[f32; 3]>,
    up: Option<[f32; 3]>,
    /// Vertical field of view in degrees; falls back to --projection-fov.
    fov: Option<f32>,
}

#[derive(Serialize, Debug)]
struct PoseStats {
    name: String,
    /// Fraction of pixels with alpha > 0.5.
    coverage: f32,
    /// Mean alpha over all pixels (0..1).
    mean_alpha: f32,
}

fn pose_camera(pose: &Pose, args: &Args) -> Result<Camera> {
    let aspect = args.width as f32 / args.height as f32;
    let fov = pose.fov.unwrap_or(args.projection_fov);
    let projection = Perspective::new(fov.to_radians(), aspect, args.near, args.far);
    if let Some(m) = pose.matrix {
        return Ok(Camera::from_camera_to_world(Mat4::from_cols_slice(&m), projection));
    }
    let position = pose.position.context("pose needs matrix or position")?;
    let eye = Vec3::from_array(position);
    let target = pose.lookat.map_or(Vec3::ZERO, Vec3::from_array);
    let up = pose.up.map_or(Vec3::Y, Vec3::from_array);
    Ok(Camera::look_at(eye, target, up, projection))
}

fn alpha_stats(image: &image::RgbaImage) -> (f32, f32) {
    let mut covered = 0u64;
    let mut total_alpha = 0u64;
    for pixel in image.pixels() {
        let alpha = pixel[3];
        if alpha > 127 {
            covered += 1;
        }
        total_alpha += u64::from(alpha);
    }
    let count = (image.width() * image.height()) as f32;
    (covered as f32 / count, total_alpha as f32 / (count * 255.0))
}

fn run_batch(args: &Args, poses_path: &Path) -> Result<()> {
    let poses: Vec<Pose> = serde_json::from_reader(
        File::open(poses_path).with_context(|| format!("opening {}", poses_path.display()))?,
    )
    .context("parsing poses JSON")?;

    let output_dir = args.output_dir.clone().unwrap_or_else(|| PathBuf::from("."));
    std::fs::create_dir_all(&output_dir)?;
    let mut stats_out = args
        .stats
        .as_ref()
        .map(|p| File::create(p).map(BufWriter::new))
        .transpose()?;

    let cloud =
        SplatCloud::from_path(&args.splat).with_context(|| format!("loading {}", args.splat))?;
    eprintln!(
        "Loaded {} splats, SH degree {}",
        cloud.splats().len(),
        cloud.sh_degree()
    );
    let renderer = Renderer::new()?;

    if args.stats.is_some() && args.background.0[3] >= 1.0 {
        eprintln!("warning: --stats with an opaque background; use --background 0,0,0,0 so alpha reflects splat coverage");
    }

    let settings = RenderSettings {
        width: args.width,
        height: args.height,
        background: args.background.0,
        model: match args.model_position {
            Some(Vec3Arg(p)) => Mat4::from_translation(p),
            None => Mat4::IDENTITY,
        },
        srgb_to_linear: args.srgb_to_linear,
        sort: args.sort.into(),
    };

    for (index, pose) in poses.iter().enumerate() {
        let name = pose
            .name
            .clone()
            .unwrap_or_else(|| format!("{index:05}"));
        let camera = pose_camera(pose, args).with_context(|| format!("pose {name}"))?;
        let image = renderer.render(&cloud, &camera, &settings)?;
        let path = output_dir.join(format!("{name}.png"));
        image
            .save(&path)
            .with_context(|| format!("writing {}", path.display()))?;
        if let Some(out) = stats_out.as_mut() {
            let (coverage, mean_alpha) = alpha_stats(&image);
            let record = PoseStats {
                name: name.clone(),
                coverage,
                mean_alpha,
            };
            serde_json::to_writer(&mut *out, &record)?;
            out.write_all(b"\n")?;
        }
        eprintln!("[{}/{}] {}", index + 1, poses.len(), path.display());
    }
    if let Some(mut out) = stats_out {
        out.flush()?;
    }
    Ok(())
}

fn main() -> Result<()> {
    let args = Args::parse();

    if let Some(path) = args.dump_positions.clone() {
        let cloud = SplatCloud::from_path(&args.splat)
            .with_context(|| format!("loading {}", args.splat))?;
        let mut out = BufWriter::new(File::create(&path)?);
        for splat in cloud.splats() {
            out.write_all(bytemuck::cast_slice(&splat.position[..3]))?;
        }
        out.flush()?;
        eprintln!("Wrote {} positions to {}", cloud.splats().len(), path.display());
        return Ok(());
    }

    if let Some(poses_path) = args.poses.clone() {
        return run_batch(&args, &poses_path);
    }

    let camera = build_camera(&args);
    let model = match args.model_position {
        Some(Vec3Arg(p)) => Mat4::from_translation(p),
        None => Mat4::IDENTITY,
    };

    let cloud =
        SplatCloud::from_path(&args.splat).with_context(|| format!("loading {}", args.splat))?;
    eprintln!(
        "Loaded {} splats, SH degree {}",
        cloud.splats().len(),
        cloud.sh_degree()
    );

    let settings = RenderSettings {
        width: args.width,
        height: args.height,
        background: args.background.0,
        model,
        srgb_to_linear: args.srgb_to_linear,
        sort: args.sort.into(),
    };

    let renderer = Renderer::new()?;
    let image = renderer.render(&cloud, &camera, &settings)?;
    image
        .save(&args.output)
        .with_context(|| format!("writing {}", args.output))?;
    eprintln!("Wrote {}", args.output);
    Ok(())
}
