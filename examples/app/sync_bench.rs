//! Headless Bevy cell for the Blade / wgpu-core barrier-host protocol.
//!
//! Same cube+ground scene as `headless_renderer`, with pass multipliers turned
//! on: point-light shadows, GPU mesh preprocessing (engine default), a depth
//! and normal prepass, and SSAO compute. MSAA stays off. Emits
//! `# schema,blade-sync-bench-v1` on stdout. Host record/submit is timed at
//! the wgpu dispatch boundary (`CommandEncoder::finish` / `Queue::submit`);
//! GPU elapsed is wait-to-idle.

#![allow(clippy::print_stdout, clippy::print_stderr)]

use bevy::{
    app::{AppExit, ScheduleRunnerPlugin},
    camera::RenderTarget,
    core_pipeline::{
        prepass::{DepthPrepass, NormalPrepass},
        tonemapping::Tonemapping,
    },
    image::TextureFormatPixelInfo,
    pbr::{ScreenSpaceAmbientOcclusion, ScreenSpaceAmbientOcclusionQualityLevel},
    prelude::*,
    render::{
        dispatch_stats,
        render_asset::RenderAssets,
        render_resource::{
            Buffer, BufferDescriptor, BufferUsages, CommandEncoderDescriptor, Extent3d, MapMode,
            PollType, TexelCopyBufferInfo, TexelCopyBufferLayout, TextureFormat, TextureUsages,
        },
        renderer::{
            RenderAdapterInfo, RenderContext, RenderDevice, RenderGraph, RenderGraphSystems,
            RenderQueue,
        },
        view::Msaa,
        Extract, Render, RenderApp, RenderSystems,
    },
    window::ExitCondition,
    winit::WinitPlugin,
};
use crossbeam_channel::{Receiver, Sender};
use std::{
    env, process,
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        Arc,
    },
    time::Duration,
};

const WORKLOAD: &str = "bevy-headless";
const CLEAR_SRGB: [u8; 4] = [40, 40, 80, 255];

fn implementation() -> &'static str {
    if cfg!(feature = "blade") {
        "blade"
    } else {
        "wgpu"
    }
}

fn backend_name() -> &'static str {
    if cfg!(any(target_os = "macos", target_os = "ios")) {
        "metal"
    } else {
        "vulkan"
    }
}

fn default_policy() -> String {
    if cfg!(feature = "blade") {
        "automatic".into()
    } else {
        "tracked".into()
    }
}

struct Config {
    workload: String,
    policy: String,
    passes: u32,
    elements: u32,
    rounds: u32,
    width: u32,
    height: u32,
    warmups: u32,
    samples: u32,
    pre_roll: u32,
    validation: bool,
    gpu_timing: bool,
    allow_software: bool,
    output_image: std::path::PathBuf,
}

impl Config {
    fn parse() -> Result<Self, String> {
        let mut cfg = Self {
            workload: WORKLOAD.into(),
            policy: default_policy(),
            passes: 1,
            elements: 1,
            rounds: 1,
            width: 1280,
            height: 720,
            warmups: 8,
            samples: 5,
            pre_roll: 90,
            validation: false,
            gpu_timing: true,
            allow_software: false,
            output_image: std::path::PathBuf::from("bevy-sync-bench.png"),
        };
        let mut args = env::args().skip(1);
        while let Some(arg) = args.next() {
            let mut next = |flag: &str| {
                args.next()
                    .ok_or_else(|| format!("{flag} requires a value"))
            };
            match arg.as_str() {
                "--workload" => cfg.workload = next("--workload")?,
                "--policy" => cfg.policy = next("--policy")?,
                "--passes" => cfg.passes = next("--passes")?.parse().map_err(|e| format!("{e}"))?,
                "--elements" => {
                    cfg.elements = next("--elements")?.parse().map_err(|e| format!("{e}"))?
                }
                "--rounds" => cfg.rounds = next("--rounds")?.parse().map_err(|e| format!("{e}"))?,
                "--width" => cfg.width = next("--width")?.parse().map_err(|e| format!("{e}"))?,
                "--height" => cfg.height = next("--height")?.parse().map_err(|e| format!("{e}"))?,
                "--warmups" => {
                    cfg.warmups = next("--warmups")?.parse().map_err(|e| format!("{e}"))?
                }
                "--samples" => {
                    cfg.samples = next("--samples")?.parse().map_err(|e| format!("{e}"))?
                }
                "--pre-roll" => {
                    cfg.pre_roll = next("--pre-roll")?.parse().map_err(|e| format!("{e}"))?
                }
                "--output-image" => cfg.output_image = next("--output-image")?.into(),
                "--validation" => cfg.validation = true,
                "--no-gpu-timing" => cfg.gpu_timing = false,
                "--allow-software" => cfg.allow_software = true,
                "--device-id" => {
                    let _ = next("--device-id")?;
                }
                "--list-adapters" => {
                    return Err("adapter listing is not implemented for the Bevy cell".into());
                }
                "--help" | "-h" => {
                    eprintln!(
                        "Bevy headless sync-bench cell. Accepts the synthetic collector flags\n\
                         (--workload --policy --passes --width --height --warmups --samples ...)."
                    );
                    process::exit(0);
                }
                other => return Err(format!("unknown argument: {other}")),
            }
        }
        if cfg.workload != WORKLOAD {
            return Err(format!(
                "unsupported workload {}; only {WORKLOAD} is collected",
                cfg.workload
            ));
        }
        if cfg.width == 0 || cfg.height == 0 || cfg.samples == 0 {
            return Err("--width, --height, and --samples must be positive".into());
        }
        Ok(cfg)
    }
}

#[derive(Resource, Clone)]
struct BenchConfig {
    workload: String,
    policy: String,
    cli_passes: u32,
    elements: u32,
    rounds: u32,
    width: u32,
    height: u32,
    warmups: u32,
    samples: u32,
    pre_roll: u32,
    validation: bool,
    gpu_timing: bool,
    allow_software: bool,
    output_image: std::path::PathBuf,
}

#[derive(Clone, Copy, Debug)]
enum Phase {
    PreRoll(u32),
    Warmup(u32),
    Sample(u32),
    Capture,
    Done,
}

#[derive(Resource, Clone)]
struct BenchState {
    phase: Phase,
    sample_index: u32,
}

#[derive(Resource, Deref)]
struct MainWorldReceiver(Receiver<Vec<u8>>);

#[derive(Resource, Deref)]
struct RenderWorldSender(Sender<Vec<u8>>);

#[derive(Resource, Clone)]
struct LastPassCount(Arc<AtomicU64>);

fn main() {
    let config = Config::parse().unwrap_or_else(|error| {
        eprintln!("error: {error}");
        process::exit(2);
    });
    let last_pass = LastPassCount(Arc::new(AtomicU64::new(0)));
    let bench = BenchConfig {
        workload: config.workload,
        policy: config.policy,
        cli_passes: config.passes,
        elements: config.elements,
        rounds: config.rounds,
        width: config.width,
        height: config.height,
        warmups: config.warmups,
        samples: config.samples,
        pre_roll: config.pre_roll,
        validation: config.validation,
        gpu_timing: config.gpu_timing,
        allow_software: config.allow_software,
        output_image: config.output_image,
    };
    let state = BenchState {
        phase: Phase::PreRoll(bench.pre_roll.max(1)),
        sample_index: 0,
    };

    App::new()
        .insert_resource(ClearColor(Color::srgb_u8(CLEAR_SRGB[0], CLEAR_SRGB[1], CLEAR_SRGB[2])))
        .insert_resource(bench.clone())
        .insert_resource(state)
        .insert_resource(last_pass.clone())
        .add_plugins(
            DefaultPlugins
                .set(ImagePlugin::default_nearest())
                .set(WindowPlugin {
                    primary_window: None,
                    exit_condition: ExitCondition::DontExit,
                    ..default()
                })
                .disable::<WinitPlugin>(),
        )
        .add_plugins(ImageCopyPlugin)
        .add_plugins(CaptureFramePlugin)
        .add_plugins(ScheduleRunnerPlugin::run_loop(Duration::ZERO))
        .add_systems(Startup, setup)
        .run();
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut images: ResMut<Assets<Image>>,
    bench: Res<BenchConfig>,
    render_device: Res<RenderDevice>,
) {
    let render_target = setup_render_target(
        &mut commands,
        &mut images,
        &render_device,
        bench.width,
        bench.height,
    );

    commands.spawn((
        Mesh3d(meshes.add(Circle::new(4.0))),
        MeshMaterial3d(materials.add(Color::WHITE)),
        Transform::from_rotation(Quat::from_rotation_x(-std::f32::consts::FRAC_PI_2)),
    ));
    commands.spawn((
        Mesh3d(meshes.add(Cuboid::new(1.0, 1.0, 1.0))),
        MeshMaterial3d(materials.add(Color::srgb_u8(124, 144, 255))),
        Transform::from_xyz(0.0, 0.5, 0.0),
    ));
    commands.spawn((
        PointLight {
            shadow_maps_enabled: true,
            ..default()
        },
        Transform::from_xyz(4.0, 8.0, 4.0),
    ));
    commands.spawn((
        Camera3d::default(),
        render_target,
        Tonemapping::Linear,
        Msaa::Off,
        DepthPrepass,
        NormalPrepass,
        ScreenSpaceAmbientOcclusion {
            quality_level: ScreenSpaceAmbientOcclusionQualityLevel::High,
            ..default()
        },
        Transform::from_xyz(-2.5, 4.5, 9.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));
}

struct ImageCopyPlugin;
impl Plugin for ImageCopyPlugin {
    fn build(&self, app: &mut App) {
        let (s, r) = crossbeam_channel::unbounded();
        let render_app = app
            .insert_resource(MainWorldReceiver(r))
            .sub_app_mut(RenderApp);
        render_app
            .insert_resource(RenderWorldSender(s))
            .add_systems(ExtractSchedule, image_copy_extract)
            .add_systems(ExtractSchedule, extract_bench)
            .add_systems(
                RenderGraph,
                reset_dispatch_stats.in_set(RenderGraphSystems::Begin),
            )
            .add_systems(
                Render,
                sample_dispatch_stats.after(RenderSystems::Render),
            )
            .add_systems(
                Render,
                receive_image_from_buffer.after(RenderSystems::Render),
            )
            .add_systems(
                RenderGraph,
                image_copy_driver.after(RenderGraphSystems::Submit),
            );
    }

    fn finish(&self, app: &mut App) {
        let bench = app.world().resource::<BenchConfig>().clone();
        let last = app.world().resource::<LastPassCount>().clone();
        let render_app = app.sub_app_mut(RenderApp);
        render_app.insert_resource(bench);
        render_app.insert_resource(last);
        render_app.insert_resource(HeaderPrinted(false));
        render_app.insert_resource(BenchState {
            phase: Phase::PreRoll(1),
            sample_index: 0,
        });
    }
}

#[derive(Resource, Clone)]
struct HeaderPrinted(bool);

fn extract_bench(mut commands: Commands, state: Extract<Res<BenchState>>) {
    commands.insert_resource(state.clone());
}

fn reset_dispatch_stats() {
    dispatch_stats::reset();
    #[cfg(feature = "blade")]
    bevy::render::lifecycle_reset();
}

fn sample_dispatch_stats(
    adapter: Res<RenderAdapterInfo>,
    device: Res<RenderDevice>,
    config: Res<BenchConfig>,
    state: Res<BenchState>,
    last_pass: Res<LastPassCount>,
    mut header: ResMut<HeaderPrinted>,
) {
    if config.gpu_timing {
        let _ = device.poll(PollType::wait_indefinitely());
    }
    let stats = dispatch_stats::take();
    last_pass.0.store(stats.gpu_pass_count, Ordering::Relaxed);

    if !header.0 {
        if is_software_adapter(&adapter) && !config.allow_software {
            eprintln!(
                "error: {} is a software device; pass --allow-software for correctness-only runs",
                adapter.name
            );
            process::exit(2);
        }
        print_header(&adapter, &config);
        header.0 = true;
    }

    if let Phase::Sample(_) = state.phase {
        println!(
            "{},{},{},{},{},{},{},{},0,{},{},{},{},{}",
            state.sample_index,
            config.workload,
            config.policy,
            stats.gpu_pass_count,
            config.elements,
            config.rounds,
            config.width,
            config.height,
            stats.record_ns,
            stats.submit_ns,
            stats.wait_ns,
            stats.gpu_ns(),
            stats.gpu_pass_count,
        );
        #[cfg(feature = "blade")]
        {
            let lc = bevy::render::take_encoder_lifecycle();
            println!(
                "# encoder_lifecycle,create_n={},alloc_n={},recycle_n={},cb_n={},create_ns={},vk_submit_ns={},destroy_ns={}",
                lc.create_count,
                lc.alloc_count,
                lc.recycle_count,
                lc.cb_count,
                lc.create_ns,
                lc.vk_submit_ns,
                lc.destroy_ns,
            );
        }
    }
}

fn is_software_adapter(adapter: &RenderAdapterInfo) -> bool {
    format!("{:?}", adapter.device_type).contains("Cpu")
}

fn print_header(adapter: &RenderAdapterInfo, config: &BenchConfig) {
    println!("# schema,blade-sync-bench-v1");
    println!("# implementation,{}", implementation());
    println!("# backend,{}", backend_name());
    println!("# device_name,\"{}\"", adapter.name.replace('"', "\"\""));
    println!(
        "# driver_name,\"{}\"",
        adapter.driver.replace('"', "\"\"")
    );
    println!(
        "# driver_info,\"{}\"",
        adapter.driver_info.replace('"', "\"\"")
    );
    println!(
        "# software_emulated,{}",
        is_software_adapter(adapter)
    );
    println!("# validation,{}", config.validation);
    println!("# gpu_timing,{}", config.gpu_timing);
    println!("# gpu_timing_method,{}", dispatch_stats::GPU_TIMING_METHOD);
    println!(
        "# bevy_settings,shadows+depth_prepass+normal_prepass+ssao+gpu_preprocess,msaa=off"
    );
    println!("# cli_passes,{}", config.cli_passes);
    println!(
        "sample,workload,policy,passes,elements,rounds,width,height,start_ns,record_ns,submit_ns,wait_ns,gpu_ns,gpu_pass_count"
    );
}

fn setup_render_target(
    commands: &mut Commands,
    images: &mut ResMut<Assets<Image>>,
    render_device: &Res<RenderDevice>,
    width: u32,
    height: u32,
) -> RenderTarget {
    let size = Extent3d {
        width,
        height,
        ..Default::default()
    };
    let mut render_target_image =
        Image::new_target_texture(size.width, size.height, TextureFormat::Rgba8UnormSrgb, None);
    render_target_image.texture_descriptor.usage |= TextureUsages::COPY_SRC;
    let render_target_image_handle = images.add(render_target_image);
    let cpu_image =
        Image::new_target_texture(size.width, size.height, TextureFormat::Rgba8UnormSrgb, None);
    let cpu_image_handle = images.add(cpu_image);
    commands.spawn(ImageCopier::new(
        render_target_image_handle.clone(),
        size,
        render_device,
        false,
    ));
    commands.spawn(ImageToSave(cpu_image_handle));
    RenderTarget::Image(render_target_image_handle.into())
}

struct CaptureFramePlugin;
impl Plugin for CaptureFramePlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(PostUpdate, update);
    }
}

#[derive(Clone, Default, Resource, Deref, DerefMut)]
struct ImageCopiers(pub Vec<ImageCopier>);

#[derive(Clone, Component)]
struct ImageCopier {
    buffer: Buffer,
    enabled: Arc<AtomicBool>,
    src_image: Handle<Image>,
}

impl ImageCopier {
    pub fn new(
        src_image: Handle<Image>,
        size: Extent3d,
        render_device: &RenderDevice,
        enabled: bool,
    ) -> ImageCopier {
        let padded_bytes_per_row = RenderDevice::align_copy_bytes_per_row(size.width as usize * 4);
        let cpu_buffer = render_device.create_buffer(&BufferDescriptor {
            label: None,
            size: padded_bytes_per_row as u64 * size.height as u64,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        ImageCopier {
            buffer: cpu_buffer,
            src_image,
            enabled: Arc::new(AtomicBool::new(enabled)),
        }
    }

    pub fn enabled(&self) -> bool {
        self.enabled.load(Ordering::Relaxed)
    }
}

fn image_copy_extract(mut commands: Commands, image_copiers: Extract<Query<&ImageCopier>>) {
    commands.insert_resource(ImageCopiers(
        image_copiers.iter().cloned().collect::<Vec<ImageCopier>>(),
    ));
}

fn image_copy_driver(
    render_context: RenderContext,
    image_copiers: Res<ImageCopiers>,
    render_queue: Res<RenderQueue>,
    gpu_images: Res<RenderAssets<bevy::render::texture::GpuImage>>,
) {
    for image_copier in image_copiers.iter() {
        if !image_copier.enabled() {
            continue;
        }
        let src_image = gpu_images.get(&image_copier.src_image).unwrap();
        let mut encoder = render_context
            .render_device()
            .create_command_encoder(&CommandEncoderDescriptor::default());
        let block_dimensions = src_image.texture_descriptor.format.block_dimensions();
        let block_size = src_image
            .texture_descriptor
            .format
            .block_copy_size(None)
            .unwrap();
        let padded_bytes_per_row = RenderDevice::align_copy_bytes_per_row(
            (src_image.texture_descriptor.size.width as usize / block_dimensions.0 as usize)
                * block_size as usize,
        );
        encoder.copy_texture_to_buffer(
            src_image.texture.as_image_copy(),
            TexelCopyBufferInfo {
                buffer: &image_copier.buffer,
                layout: TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(
                        std::num::NonZero::<u32>::new(padded_bytes_per_row as u32)
                            .unwrap()
                            .into(),
                    ),
                    rows_per_image: None,
                },
            },
            src_image.texture_descriptor.size,
        );
        render_queue.submit(std::iter::once(encoder.finish()));
    }
}

fn receive_image_from_buffer(
    image_copiers: Res<ImageCopiers>,
    render_device: Res<RenderDevice>,
    sender: Res<RenderWorldSender>,
) {
    for image_copier in image_copiers.0.iter() {
        if !image_copier.enabled() {
            continue;
        }
        let buffer_slice = image_copier.buffer.slice(..);
        let (s, r) = crossbeam_channel::bounded(1);
        buffer_slice.map_async(MapMode::Read, move |res| match res {
            Ok(ok) => s.send(ok).expect("Failed to send map update"),
            Err(err) => panic!("Failed to map buffer {err}"),
        });
        render_device
            .poll(PollType::wait_indefinitely())
            .expect("Failed to poll device for map async");
        r.recv().expect("Failed to receive the map_async message");
        let _ = sender.send(buffer_slice.get_mapped_range().unwrap().to_vec());
        image_copier.buffer.unmap();
    }
}

#[derive(Component, Deref, DerefMut)]
struct ImageToSave(Handle<Image>);

fn fnv1a64(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

fn unique_pixels(pixels: &[u8]) -> usize {
    let mut seen = std::collections::BTreeSet::new();
    for chunk in pixels.chunks_exact(4) {
        seen.insert([chunk[0], chunk[1], chunk[2], chunk[3]]);
    }
    seen.len()
}

fn is_blank(pixels: &[u8]) -> bool {
    if pixels.is_empty() {
        return true;
    }
    let all_zero = pixels.iter().all(|&b| b == 0);
    let all_clear = pixels.chunks_exact(4).all(|p| p == CLEAR_SRGB);
    all_zero || all_clear
}

fn update(
    images_to_save: Query<&ImageToSave>,
    copiers: Query<&ImageCopier>,
    receiver: Res<MainWorldReceiver>,
    mut images: ResMut<Assets<Image>>,
    mut state: ResMut<BenchState>,
    config: Res<BenchConfig>,
    last_pass: Res<LastPassCount>,
    mut app_exit_writer: MessageWriter<AppExit>,
) {
    match state.phase {
        Phase::PreRoll(n) => {
            while receiver.try_recv().is_ok() {}
            if n <= 1 {
                state.phase = if config.warmups == 0 {
                    Phase::Sample(config.samples)
                } else {
                    Phase::Warmup(config.warmups)
                };
            } else {
                state.phase = Phase::PreRoll(n - 1);
            }
        }
        Phase::Warmup(n) => {
            while receiver.try_recv().is_ok() {}
            if n <= 1 {
                state.phase = Phase::Sample(config.samples);
                state.sample_index = 0;
            } else {
                state.phase = Phase::Warmup(n - 1);
            }
        }
        Phase::Sample(n) => {
            while receiver.try_recv().is_ok() {}
            if n <= 1 {
                for copier in copiers.iter() {
                    copier.enabled.store(true, Ordering::Relaxed);
                }
                state.phase = Phase::Capture;
            } else {
                state.sample_index += 1;
                state.phase = Phase::Sample(n - 1);
            }
        }
        Phase::Capture => {
            let mut image_data = Vec::new();
            while let Ok(data) = receiver.try_recv() {
                image_data = data;
            }
            if image_data.is_empty() {
                return;
            }
            for image in images_to_save.iter() {
                let mut img_bytes = images.get_mut(image.id()).unwrap();
                let row_bytes = img_bytes.width() as usize
                    * img_bytes.texture_descriptor.format.pixel_size().unwrap();
                let aligned_row_bytes = RenderDevice::align_copy_bytes_per_row(row_bytes);
                if row_bytes == aligned_row_bytes {
                    img_bytes.data.as_mut().unwrap().clone_from(&image_data);
                } else {
                    img_bytes.data = Some(
                        image_data
                            .chunks(aligned_row_bytes)
                            .take(img_bytes.height() as usize)
                            .flat_map(|row| &row[..row_bytes.min(row.len())])
                            .cloned()
                            .collect(),
                    );
                }
                let pixels = img_bytes.data.as_ref().unwrap().clone();
                if is_blank(&pixels) {
                    eprintln!("error: captured Bevy frame is blank or clear-color only");
                    process::exit(1);
                }
                if unique_pixels(&pixels) < 8 {
                    eprintln!(
                        "error: captured Bevy frame is not substantially filled (unique colors = {})",
                        unique_pixels(&pixels)
                    );
                    process::exit(1);
                }
                let pass_count = last_pass.0.load(Ordering::Relaxed);
                if pass_count <= 1 {
                    eprintln!(
                        "error: Bevy cell gpu_pass_count is {pass_count}, expected multiple graphics+compute passes"
                    );
                    process::exit(1);
                }
                let img = match img_bytes.clone().try_into_dynamic() {
                    Ok(img) => img.to_rgba8(),
                    Err(e) => panic!("Failed to create image buffer {e:?}"),
                };
                if let Some(parent) = config.output_image.parent() {
                    if !parent.as_os_str().is_empty() {
                        std::fs::create_dir_all(parent).unwrap();
                    }
                }
                if let Err(e) = img.save(&config.output_image) {
                    panic!("Failed to save image: {e}");
                }
                let hash = fnv1a64(&pixels);
                println!("# validation_hash,fnv1a64-standard:{hash:016x}");
                println!("# output_image,{}", config.output_image.display());
                println!("# unique_colors,{}", unique_pixels(&pixels));
            }
            state.phase = Phase::Done;
            app_exit_writer.write(AppExit::Success);
        }
        Phase::Done => {}
    }
}
