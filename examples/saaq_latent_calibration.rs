mod support;

use corinth_canal::{
    FunnelActivity, HeartbeatInjector, SaaqUpdateRule, SnnDualLatentCalibrator,
    SnnLatentCsvExporter, gpu::GpuAccelerator, model::Model,
};
use serde::Serialize;
use std::fs::{self, File};
use std::io::{BufWriter, Error, Write};
use std::path::PathBuf;
use std::time::Instant;
use support::{
    ResolvedTelemetry, RunConfig, TelemetrySource, ValidationModelSpec,
    default_spiking_model_config, heartbeat_gain, prompt_embedding_for_validation,
    telemetry_snapshot_for_tick,
};

#[derive(Debug, Serialize)]
struct ValidationManifest {
    model_slug: String,
    model_family: String,
    architecture: String,
    checkpoint_path: String,
    routing_tensor_name: String,
    synapse_source: String,
    checkpoint_format: &'static str,
    prompt_embedding_source: String,
    prompt_profile: String,
    prompt_text: String,
    ticks: usize,
    saaq_rule: &'static str,
    saaq_primary_rule: &'static str,
    saaq_dual_emit: bool,
    validation_status: &'static str,
    error: Option<String>,
    heartbeat_enabled: bool,
    heartbeat_amplitude: f32,
    heartbeat_period_ticks: usize,
    heartbeat_duty_cycle: f32,
    heartbeat_phase_offset_ticks: usize,
    telemetry_source: String,
    telemetry_csv_path: Option<String>,
    telemetry_row_count: Option<usize>,
    wraparound_enabled: bool,
    wraparound_loops: u64,
    ticks_effective: usize,
    run_id: String,
    run_dir: String,
    output_root: String,
    repeat_idx: usize,
    repeat_count: usize,
    cwd_routing_csv_contaminated: bool,
    generated_files: Vec<String>,
}

struct RunContext<'a> {
    spec: &'a ValidationModelSpec,
    prompt_profile: &'a str,
    prompt_text: &'a str,
    ticks: usize,
    heartbeat_enabled: bool,
    repeat_idx: usize,
    repeat_count: usize,
    resolved: &'a ResolvedTelemetry,
    run_id: String,
    output_root: PathBuf,
    model_family_override: Option<corinth_canal::ModelFamily>,
    saaq_rule: SaaqUpdateRule,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let _ = dotenvy::from_filename(".env.local");
    let cfg = RunConfig::from_env();

    // Effective tick count: when TICKS=0 and CSV replay is live, use the full
    // CSV length so SR.jl corpus runs cover exactly one loop with zero
    // wraparound contamination (per plan: wraparound is for smoke only).
    let effective_ticks = match (cfg.ticks, cfg.telemetry.source, cfg.telemetry.row_count()) {
        (0, TelemetrySource::Csv, Some(rows)) if rows > 0 => rows,
        (0, _, _) => 1, // degenerate guard; never emit a zero-tick run
        (other, _, _) => other,
    };

    if let Some(rows) = cfg.telemetry.row_count() {
        if effective_ticks > rows {
            eprintln!(
                "wraparound: ticks={effective_ticks} > rows={rows}; regression corpus may be contaminated by looped endings. Prefer TICKS=0 or TICKS<={rows}.",
            );
        }
    }

    println!(
        "saaq_latent_calibration: telemetry_source={} ticks={} repeat_count={} output_root={}",
        cfg.telemetry.source_label,
        effective_ticks,
        cfg.repeat_count,
        cfg.output_root.display(),
    );

    if cfg.validation_models.is_empty() {
        return Err(Error::other(
            "No GGUF validation models discovered. Set GGUF_CHECKPOINT_PATH or place models under ~/Downloads/SNN_Quantization.",
        )
        .into());
    }

    for spec in &cfg.validation_models {
        for repeat_idx in 0..cfg.repeat_count {
            for &heartbeat_enabled in &cfg.heartbeat_matrix {
                let run_id = build_run_id(&cfg.prompt_profile, repeat_idx);
                let ctx = RunContext {
                    spec,
                    prompt_profile: &cfg.prompt_profile,
                    prompt_text: cfg.prompt_text,
                    ticks: effective_ticks,
                    heartbeat_enabled,
                    repeat_idx,
                    repeat_count: cfg.repeat_count,
                    resolved: &cfg.telemetry,
                    run_id,
                    output_root: cfg.output_root.clone(),
                    model_family_override: cfg.model_family_override,
                    saaq_rule: cfg.saaq_rule,
                };
                run_validation(&ctx)?;
            }
        }
    }

    Ok(())
}

fn run_validation(ctx: &RunContext<'_>) -> Result<(), Box<dyn std::error::Error>> {
    // Pre-create the run directory so we can anchor the GPU routing
    // telemetry CSV inside it via ModelConfig and avoid polluting CWD.
    let run_dir = build_run_dir(ctx)?;
    fs::create_dir_all(&run_dir)?;
    let routing_csv_path = run_dir.join("snn_gpu_routing_telemetry.csv");

    let mut config = default_spiking_model_config(ctx.spec.path.clone(), 1);
    config.model_family = ctx.model_family_override.or(ctx.spec.family);
    config.heartbeat.enabled = ctx.heartbeat_enabled;
    config.gpu_routing_telemetry_path = Some(routing_csv_path.clone());
    let saaq_rule = ctx.saaq_rule;

    let mut accelerator = GpuAccelerator::new();
    let mut model = Model::new(config.clone())?;

    if !model.router_loaded() {
        return Err(Error::other(format!(
            "router did not load for checkpoint '{}'",
            ctx.spec.path
        ))
        .into());
    }

    let tick_path = run_dir.join("tick_telemetry.txt");
    let latent_path = run_dir.join("latent_telemetry.csv");
    let manifest_path = run_dir.join("run_manifest.json");
    let generated_files = vec![
        tick_path.file_name().unwrap().to_string_lossy().into_owned(),
        latent_path.file_name().unwrap().to_string_lossy().into_owned(),
        manifest_path
            .file_name()
            .unwrap()
            .to_string_lossy()
            .into_owned(),
        routing_csv_path
            .file_name()
            .unwrap()
            .to_string_lossy()
            .into_owned(),
    ];
    let target_neurons = model.projector_mut().input_neurons();
    let (prompt_embedding, prompt_embedding_source) =
        match prompt_embedding_for_validation(&ctx.spec.path, ctx.prompt_text, target_neurons) {
            Ok(result) => result,
            Err(error) => {
                write_manifest(
                    &manifest_path,
                    build_manifest(
                        ctx,
                        &config,
                        &model,
                        &run_dir,
                        "unavailable",
                        saaq_rule,
                        "prompt_embedding_failed",
                        Some(error.to_string()),
                        generated_files.clone(),
                    ),
                )?;
                return Err(error);
            }
        };

    write_manifest(
        &manifest_path,
        build_manifest(
            ctx,
            &config,
            &model,
            &run_dir,
            &prompt_embedding_source,
            saaq_rule,
            "preflight",
            None,
            generated_files.clone(),
        ),
    )?;

    if let Err(error) = model.prepare_gpu_temporal(&mut accelerator) {
        write_manifest(
            &manifest_path,
            build_manifest(
                ctx,
                &config,
                &model,
                &run_dir,
                &prompt_embedding_source,
                saaq_rule,
                "gpu_setup_failed",
                Some(error.to_string()),
                generated_files,
            ),
        )?;
        return Err(Box::new(error));
    }

    let mut tick_writer = BufWriter::new(File::create(&tick_path)?);
    let mut latent_exporter = SnnLatentCsvExporter::create(&latent_path)?;
    let mut calibrator = SnnDualLatentCalibrator::new(saaq_rule);
    let heartbeat = HeartbeatInjector::new(config.heartbeat.clone());

    println!(
        "validation_start model_slug={} family={:?} architecture={} heartbeat_enabled={} ticks={} repeat={}/{} routing_tensor={} synapse_source={} telemetry_source={}",
        ctx.spec.slug,
        model.router_family(),
        model.router_architecture(),
        ctx.heartbeat_enabled,
        ctx.ticks,
        ctx.repeat_idx + 1,
        ctx.repeat_count,
        model.routing_tensor_name(),
        model.synapse_source(),
        ctx.resolved.source_label,
    );

    let run_result = (|| -> Result<(), Box<dyn std::error::Error>> {
        for tick in 0..ctx.ticks {
            let snap = telemetry_snapshot_for_tick(tick, ctx.resolved);
            let snap = heartbeat.apply(&snap, tick);
            let gain = heartbeat_gain(snap.heartbeat_signal);
            let input_spikes: Vec<f32> =
                prompt_embedding.iter().map(|value| value * gain).collect();

            let started = Instant::now();
            let best_walker = model.tick_gpu_temporal(&mut accelerator, &input_spikes)?;
            let elapsed_us = started.elapsed().as_micros();

            let spikes = accelerator.temporal_spikes_to_vec(target_neurons)?;
            let active_neurons: Vec<usize> = spikes
                .iter()
                .enumerate()
                .filter(|(_, value)| **value != 0)
                .map(|(idx, _)| idx)
                .collect();
            let potentials = accelerator
                .temporal_membrane_to_vec(target_neurons)?
                .into_iter()
                .map(|value| value.clamp(0.0, 1.0))
                .collect::<Vec<f32>>();
            let activity = FunnelActivity {
                ternary_events: [0; 4],
                input_spike_train: vec![active_neurons.clone()],
                spike_train: vec![active_neurons],
                potentials: potentials.clone(),
                iz_potentials: vec![0.0; 5],
            };
            let output = model.forward_activity(
                &activity.spike_train,
                &activity.potentials,
                &activity.iz_potentials,
            )?;
            let latent = calibrator.observe(&snap, &activity, &output)?;
            latent_exporter.write_row(&latent)?;

            writeln!(
                tick_writer,
                "tick={} best_walker={} elapsed_us={} heartbeat_signal={:.6} gpu_temp_c={:.3} gpu_power_w={:.3} cpu_tctl_c={:.3} cpu_package_power_w={:.3}",
                tick + 1,
                best_walker,
                elapsed_us,
                snap.heartbeat_signal,
                snap.gpu_temp_c,
                snap.gpu_power_w,
                snap.cpu_tctl_c,
                snap.cpu_package_power_w,
            )?;
        }

        latent_exporter.flush()?;
        tick_writer.flush()?;
        Ok(())
    })();

    if let Err(error) = run_result {
        let _ = latent_exporter.flush();
        let _ = tick_writer.flush();
        write_manifest(
            &manifest_path,
            build_manifest(
                ctx,
                &config,
                &model,
                &run_dir,
                &prompt_embedding_source,
                saaq_rule,
                "tick_failed",
                Some(error.to_string()),
                generated_files.clone(),
            ),
        )?;
        return Err(error);
    }

    let manifest = build_manifest(
        ctx,
        &config,
        &model,
        &run_dir,
        &prompt_embedding_source,
        saaq_rule,
        "completed",
        None,
        generated_files,
    );
    write_manifest(&manifest_path, manifest)?;

    println!(
        "validation_complete model_slug={} heartbeat_enabled={} repeat={}/{} run_dir={}",
        ctx.spec.slug,
        ctx.heartbeat_enabled,
        ctx.repeat_idx + 1,
        ctx.repeat_count,
        run_dir.display()
    );

    drop(model);
    drop(accelerator);

    Ok(())
}

fn build_manifest(
    ctx: &RunContext<'_>,
    config: &corinth_canal::model::ModelConfig,
    model: &Model,
    run_dir: &std::path::Path,
    prompt_embedding_source: &str,
    saaq_rule: SaaqUpdateRule,
    validation_status: &'static str,
    error: Option<String>,
    generated_files: Vec<String>,
) -> ValidationManifest {
    let telemetry_row_count = ctx.resolved.row_count();
    let wraparound_enabled = telemetry_row_count
        .map(|rows| rows > 0 && ctx.ticks > rows)
        .unwrap_or(false);
    // Count only *extra* passes beyond the first. `ticks == rows` is exactly
    // one pass with zero wraparound (SR.jl corpus sweet spot); `ticks =
    // 2*rows` is one wraparound; and so on. This makes the field a clean
    // predicate for "did this run teach the regressor on looped data?".
    let wraparound_loops = telemetry_row_count
        .filter(|rows| *rows > 0 && ctx.ticks > *rows)
        .map(|rows| ((ctx.ticks - rows) as u64) / (rows as u64) + 1)
        .unwrap_or(0);

    ValidationManifest {
        model_slug: ctx.spec.slug.clone(),
        model_family: format!("{:?}", model.router_family()),
        architecture: model.router_architecture().to_owned(),
        checkpoint_path: ctx.spec.path.clone(),
        routing_tensor_name: model.routing_tensor_name().to_owned(),
        synapse_source: model.synapse_source().to_owned(),
        checkpoint_format: "gguf",
        prompt_embedding_source: prompt_embedding_source.to_owned(),
        prompt_profile: ctx.prompt_profile.to_owned(),
        prompt_text: ctx.prompt_text.to_owned(),
        ticks: ctx.ticks,
        saaq_rule: saaq_rule_label(saaq_rule),
        saaq_primary_rule: saaq_rule_label(saaq_rule),
        saaq_dual_emit: true,
        validation_status,
        error,
        heartbeat_enabled: ctx.heartbeat_enabled,
        heartbeat_amplitude: config.heartbeat.amplitude,
        heartbeat_period_ticks: config.heartbeat.period_ticks,
        heartbeat_duty_cycle: config.heartbeat.duty_cycle,
        heartbeat_phase_offset_ticks: config.heartbeat.phase_offset_ticks,
        telemetry_source: ctx.resolved.source_label.clone(),
        telemetry_csv_path: ctx
            .resolved
            .csv_path
            .as_ref()
            .map(|path| path.to_string_lossy().into_owned()),
        telemetry_row_count,
        wraparound_enabled,
        wraparound_loops,
        ticks_effective: ctx.ticks,
        run_id: ctx.run_id.clone(),
        run_dir: run_dir.to_string_lossy().into_owned(),
        output_root: ctx.output_root.to_string_lossy().into_owned(),
        repeat_idx: ctx.repeat_idx,
        repeat_count: ctx.repeat_count,
        // True iff the routing CSV would land in CWD. Since Stage E this
        // runner always sets `ModelConfig::gpu_routing_telemetry_path` to
        // an absolute path inside `run_dir`, so CWD contamination is
        // impossible regardless of whether `tick_gpu_temporal` or
        // `forward_gpu_temporal` is used.
        cwd_routing_csv_contaminated: config.gpu_routing_telemetry_path.is_none(),
        generated_files,
    }
}

fn write_manifest(
    manifest_path: &PathBuf,
    manifest: ValidationManifest,
) -> Result<(), Box<dyn std::error::Error>> {
    fs::write(manifest_path, serde_json::to_string_pretty(&manifest)?)?;
    Ok(())
}

fn saaq_rule_label(rule: SaaqUpdateRule) -> &'static str {
    match rule {
        SaaqUpdateRule::LegacyV1_0 => "LegacyV1_0",
        SaaqUpdateRule::SaaqV1_5SqrtRate => "SaaqV1_5SqrtRate",
    }
}

fn build_run_id(prompt_profile: &str, repeat_idx: usize) -> String {
    format!(
        "{}_{}_r{}",
        format_local_timestamp_compact(),
        prompt_profile,
        repeat_idx
    )
}

fn build_run_dir(ctx: &RunContext<'_>) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let heartbeat_slug = if ctx.heartbeat_enabled {
        "heartbeat_on"
    } else {
        "heartbeat_off"
    };
    Ok(ctx
        .output_root
        .join(&ctx.spec.slug)
        .join(&ctx.resolved.source_label)
        .join(heartbeat_slug)
        .join(&ctx.run_id))
}

/// Return a local-time-ish compact timestamp `YYYYMMDDTHHMMSS` suitable for
/// sortable directory naming. We derive it from seconds since UNIX_EPOCH with
/// the local TZ offset pulled from `std::time` via a manual conversion — no
/// external crate needed and no chrono dependency.
fn format_local_timestamp_compact() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};

    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);

    // Pure-Rust gregorian breakdown (UTC). Using UTC keeps timestamps unique
    // and sortable without depending on the machine's TZ configuration; the
    // full ISO-ish form is still human-readable in the path.
    let (year, month, day, hour, minute, second) = civil_from_unix_secs(secs as i64);
    format!(
        "{year:04}{month:02}{day:02}T{hour:02}{minute:02}{second:02}"
    )
}

/// Civil (Y, M, D, h, m, s) in UTC from a unix timestamp.
/// Based on Howard Hinnant's public-domain date algorithm.
fn civil_from_unix_secs(secs: i64) -> (i32, u32, u32, u32, u32, u32) {
    let days = secs.div_euclid(86_400);
    let rem = secs.rem_euclid(86_400) as u32;
    let hour = rem / 3_600;
    let minute = (rem % 3_600) / 60;
    let second = rem % 60;

    // Hinnant's civil_from_days
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097) as u64;
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let y = (yoe as i64) + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = if m <= 2 { y + 1 } else { y };
    (year as i32, m as u32, d as u32, hour, minute, second)
}
