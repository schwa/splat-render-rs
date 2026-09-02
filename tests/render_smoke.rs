//! End-to-end render smoke test: no external files, tiny synthetic cloud.

use glam::{Quat, Vec3, Vec4};
use splat_render_rs::{
    Camera, GpuSplat, Perspective, RenderSettings, Renderer, SortMethod, SplatCloud,
};

fn test_cloud() -> SplatCloud {
    let splats = vec![
        GpuSplat::new(
            Vec3::ZERO,
            Vec3::splat(0.3),
            Quat::IDENTITY,
            Vec4::new(1.0, 0.0, 0.0, 1.0),
        ),
        GpuSplat::new(
            Vec3::new(0.2, 0.0, -0.5),
            Vec3::splat(0.3),
            Quat::IDENTITY,
            Vec4::new(0.0, 1.0, 0.0, 0.8),
        ),
        GpuSplat::new(
            Vec3::new(-0.2, 0.1, 0.5),
            Vec3::splat(0.3),
            Quat::IDENTITY,
            Vec4::new(0.0, 0.0, 1.0, 0.8),
        ),
    ];
    SplatCloud::from_splats(splats)
}

#[test]
fn renders_reuse_and_sorts_agree() {
    let renderer = Renderer::new().expect("create renderer");
    let cloud = test_cloud();
    let settings = RenderSettings {
        // Non-256-byte-aligned row width so readback row padding is exercised.
        width: 63,
        height: 48,
        ..RenderSettings::default()
    };
    let aspect = settings.width as f32 / settings.height as f32;
    let camera = Camera::look_at(
        Vec3::new(0.0, 0.0, 3.0),
        Vec3::ZERO,
        Vec3::Y,
        Perspective::new(60f32.to_radians(), aspect, 0.1, 100.0),
    );

    // Render twice with the same renderer (pipelines are cached), both sort paths.
    for sort in [SortMethod::Gpu, SortMethod::Cpu] {
        let settings = RenderSettings {
            sort,
            ..settings.clone()
        };
        let image = renderer.render(&cloud, &camera, &settings).expect("render");
        assert_eq!(image.dimensions(), (63, 48));
        let center = image.get_pixel(31, 24);
        assert!(
            center.0[0] > 0,
            "center pixel should show the red splat ({sort:?})"
        );
    }

    // CPU and GPU sorts must produce identical images for this scene.
    let cpu = renderer
        .render(
            &cloud,
            &camera,
            &RenderSettings {
                sort: SortMethod::Cpu,
                ..settings.clone()
            },
        )
        .expect("cpu render");
    let gpu = renderer
        .render(
            &cloud,
            &camera,
            &RenderSettings {
                sort: SortMethod::Gpu,
                ..settings
            },
        )
        .expect("gpu render");
    assert_eq!(cpu.as_raw(), gpu.as_raw(), "cpu vs gpu sort images differ");
}
