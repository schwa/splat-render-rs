mod camera;
mod ply;
mod render;
mod sort;

use anyhow::{Context, Result};
use clap::Parser;

/// Render a 3D Gaussian Splat file to a PNG.
#[derive(Parser, Debug)]
#[command(name = "splat-render-rs", about = "Render Gaussian splat files to PNG images")]
pub struct Args {
    /// Path to the splat file (.ply) to render.
    pub splat: String,

    /// Output PNG file path.
    #[arg(long, default_value = "output.png")]
    pub output: String,

    /// Width of the output image.
    #[arg(long, default_value_t = 1024)]
    pub width: u32,

    /// Height of the output image.
    #[arg(long, default_value_t = 768)]
    pub height: u32,

    /// Background color in RGBA (e.g. 0,0,0,1).
    #[arg(long, default_value = "0,0,0,1")]
    pub background: String,

    /// Camera position in x,y,z.
    #[arg(long)]
    pub camera_position: Option<String>,

    /// Camera look-at target in x,y,z.
    #[arg(long)]
    pub camera_lookat: Option<String>,

    /// Camera-to-world matrix, 16 comma-separated column-major values. Overrides other camera flags.
    #[arg(long)]
    pub camera_matrix: Option<String>,

    /// Model position in x,y,z.
    #[arg(long)]
    pub model_position: Option<String>,

    /// Vertical field of view in degrees.
    #[arg(long, default_value_t = 60.0)]
    pub projection_fov: f32,

    /// Near clipping plane.
    #[arg(long, default_value_t = 0.1)]
    pub near: f32,

    /// Far clipping plane.
    #[arg(long, default_value_t = 100.0)]
    pub far: f32,

    /// Convert sRGB to linear in the fragment shader.
    #[arg(long)]
    pub srgb_to_linear: bool,

    /// Sort method: cpu (fallback) or gpu.
    #[arg(long, default_value = "gpu")]
    pub sort: sort::SortMethod,
}

fn parse_floats(s: &str, n: usize, label: &str) -> Result<Vec<f32>> {
    let v: Vec<f32> = s
        .split(',')
        .map(|p| p.trim().parse::<f32>())
        .collect::<Result<_, _>>()
        .with_context(|| format!("{label} must be {n} comma-separated numbers"))?;
    anyhow::ensure!(v.len() == n, "{label} must have {n} values, got {}", v.len());
    Ok(v)
}

fn main() -> Result<()> {
    let args = Args::parse();

    let background = parse_floats(&args.background, 4, "--background")?;
    let cam = camera::Camera::from_args(&args)?;

    let cloud = ply::load_ply(&args.splat)
        .with_context(|| format!("loading {}", args.splat))?;
    eprintln!(
        "Loaded {} splats, SH degree {}",
        cloud.splats.len(),
        cloud.sh_degree
    );

    let image = render::render(&cloud, &cam, &args, [background[0], background[1], background[2], background[3]])?;
    image.save(&args.output).with_context(|| format!("writing {}", args.output))?;
    eprintln!("Wrote {}", args.output);
    Ok(())
}
