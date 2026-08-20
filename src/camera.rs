use crate::Args;
use anyhow::{Context, Result};
use glam::{Mat4, Vec3};

/// Resolved camera + projection for a single render.
pub struct Camera {
    /// World-to-view matrix.
    pub view: Mat4,
    /// Perspective projection (view-to-clip), wgpu depth range [0, 1].
    pub projection: Mat4,
    /// Model transform (world placement of the cloud).
    pub model: Mat4,
    /// Camera position in world space.
    pub position: Vec3,
}

fn parse_vec3(s: &str, label: &str) -> Result<Vec3> {
    let v: Vec<f32> = s
        .split(',')
        .map(|p| p.trim().parse::<f32>())
        .collect::<Result<_, _>>()
        .with_context(|| format!("{label} must be x,y,z numbers"))?;
    anyhow::ensure!(v.len() == 3, "{label} must have 3 values");
    Ok(Vec3::new(v[0], v[1], v[2]))
}

impl Camera {
    pub fn from_args(args: &Args) -> Result<Self> {
        let up = Vec3::Y;

        // Camera-to-world and world position. Priority: matrix > lookat > position.
        let (c2w, position) = if let Some(m) = &args.camera_matrix {
            anyhow::ensure!(
                args.camera_position.is_none()
                    && args.camera_lookat.is_none(),
                "--camera-matrix cannot be combined with --camera-position/--camera-lookat"
            );
            let vals: Vec<f32> = m
                .split(',')
                .map(|p| p.trim().parse::<f32>())
                .collect::<Result<_, _>>()
                .context("--camera-matrix must be 16 numbers")?;
            anyhow::ensure!(vals.len() == 16, "--camera-matrix must have 16 values");
            let c2w = Mat4::from_cols_array(&[
                vals[0], vals[1], vals[2], vals[3], vals[4], vals[5], vals[6], vals[7], vals[8],
                vals[9], vals[10], vals[11], vals[12], vals[13], vals[14], vals[15],
            ]);
            let pos = c2w.col(3).truncate();
            (c2w, pos)
        } else if let Some(target_s) = &args.camera_lookat {
            let target = parse_vec3(target_s, "--camera-lookat")?;
            let pos = match &args.camera_position {
                Some(p) => parse_vec3(p, "--camera-position")?,
                None => Vec3::new(0.0, 0.0, 1.5),
            };
            // look_at_rh is world->view; invert for camera-to-world.
            let view = Mat4::look_at_rh(pos, target, up);
            (view.inverse(), pos)
        } else if let Some(p) = &args.camera_position {
            let pos = parse_vec3(p, "--camera-position")?;
            (Mat4::from_translation(pos), pos)
        } else {
            let pos = Vec3::new(0.0, 0.0, 1.5);
            let view = Mat4::look_at_rh(pos, Vec3::ZERO, up);
            (view.inverse(), pos)
        };

        let view = c2w.inverse();

        let aspect = args.width as f32 / args.height as f32;
        let projection =
            Mat4::perspective_rh(args.projection_fov.to_radians(), aspect, args.near, args.far);

        let model = match &args.model_position {
            Some(p) => Mat4::from_translation(parse_vec3(p, "--model-position")?),
            None => Mat4::IDENTITY,
        };

        Ok(Camera {
            view,
            projection,
            model,
            position,
        })
    }
}
