//! CLI front-end for the `splat_render_rs` library.

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

/// Render a 3D Gaussian Splat file to a PNG.
#[derive(Parser, Debug)]
#[command(name = "splat-render-rs", about = "Render Gaussian splat files to PNG images")]
struct Args {
    /// Path to the splat file (.ply) to render.
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
    background: String,

    /// Camera position in x,y,z.
    #[arg(long)]
    camera_position: Option<String>,

    /// Camera look-at target in x,y,z.
    #[arg(long)]
    camera_lookat: Option<String>,

    /// Camera-to-world matrix, 16 comma-separated column-major values. Overrides other camera flags.
    #[arg(long)]
    camera_matrix: Option<String>,

    /// Model position in x,y,z.
    #[arg(long)]
    model_position: Option<String>,

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

fn parse_floats(s: &str, n: usize, label: &str) -> Result<Vec<f32>> {
    let v: Vec<f32> = s
        .split(',')
        .map(|p| p.trim().parse::<f32>())
        .collect::<Result<_, _>>()
        .with_context(|| format!("{label} must be {n} comma-separated numbers"))?;
    anyhow::ensure!(v.len() == n, "{label} must have {n} values, got {}", v.len());
    Ok(v)
}

fn parse_vec3(s: &str, label: &str) -> Result<Vec3> {
    let v = parse_floats(s, 3, label)?;
    Ok(Vec3::new(v[0], v[1], v[2]))
}

fn build_camera(args: &Args) -> Result<Camera> {
    let aspect = args.width as f32 / args.height as f32;
    let projection = Perspective::new(args.projection_fov.to_radians(), aspect, args.near, args.far);
    let up = Vec3::Y;

    // Priority: matrix > lookat > position > default.
    if let Some(m) = &args.camera_matrix {
        anyhow::ensure!(
            args.camera_position.is_none() && args.camera_lookat.is_none(),
            "--camera-matrix cannot be combined with --camera-position/--camera-lookat"
        );
        let vals = parse_floats(m, 16, "--camera-matrix")?;
        let c2w = Mat4::from_cols_slice(&vals);
        return Ok(Camera::from_camera_to_world(c2w, projection));
    }
    if let Some(target) = &args.camera_lookat {
        let target = parse_vec3(target, "--camera-lookat")?;
        let eye = match &args.camera_position {
            Some(p) => parse_vec3(p, "--camera-position")?,
            None => Vec3::new(0.0, 0.0, 1.5),
        };
        return Ok(Camera::look_at(eye, target, up, projection));
    }
    if let Some(p) = &args.camera_position {
        let eye = parse_vec3(p, "--camera-position")?;
        return Ok(Camera::from_camera_to_world(Mat4::from_translation(eye), projection));
    }
    Ok(Camera::look_at(
        Vec3::new(0.0, 0.0, 1.5),
        Vec3::ZERO,
        up,
        projection,
    ))
}

fn main() -> Result<()> {
    let args = Args::parse();

    let bg = parse_floats(&args.background, 4, "--background")?;
    let camera = build_camera(&args)?;
    let model = match &args.model_position {
        Some(p) => Mat4::from_translation(parse_vec3(p, "--model-position")?),
        None => Mat4::IDENTITY,
    };

    let cloud = SplatCloud::from_ply_path(&args.splat)
        .with_context(|| format!("loading {}", args.splat))?;
    eprintln!("Loaded {} splats, SH degree {}", cloud.splats().len(), cloud.sh_degree());

    let settings = RenderSettings {
        width: args.width,
        height: args.height,
        background: [bg[0], bg[1], bg[2], bg[3]],
        model,
        srgb_to_linear: args.srgb_to_linear,
        sort: args.sort.into(),
    };

    let renderer = Renderer::new()?;
    let image = renderer.render(&cloud, &camera, &settings)?;
    image.save(&args.output).with_context(|| format!("writing {}", args.output))?;
    eprintln!("Wrote {}", args.output);
    Ok(())
}
