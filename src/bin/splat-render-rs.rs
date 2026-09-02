//! CLI front-end for the `splat_render_rs` library.

use std::str::FromStr;

use anyhow::{Context, Result};
use clap::{Parser, ValueEnum};
use glam::{Mat4, Vec3};
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

fn main() -> Result<()> {
    let args = Args::parse();

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
