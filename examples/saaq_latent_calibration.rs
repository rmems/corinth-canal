mod support;

use corinth_canal::{
    FunnelActivity, HeartbeatInjector, SaaqUpdateRule, SnnDualLatentCalibrator,
    SnnLatentCsvExporter, gpu::GpuAccelerator, model::Model,
};
use serde::Serialize;
use std::collections::BTreeMap;
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Error, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;
use support::{
    ResolvedTelemetry, RunConfig, TelemetrySource, ValidationModelSpec,
    default_spiking_model_config, heartbeat_gain, observability, prompt_embedding_for_validation,
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
    #[serde(skip_serializing_if = "Option::is_none")]
    run_tag: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    routing_mode: Option<&'static str>,
    generated_files: Vec<String>,
}

/// Compact, stable per-run summary consumed by downstream aggregators. Lives
/// alongside `run_manifest.json` inside every run directory.
#[derive(Debug, Serialize)]
struct RunSummary {
    run_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    run_tag: Option<String>,
    model_slug: String,
    model_family: String,
    telemetry_source: String,
    heartbeat_enabled: bool,
    repeat_idx: usize,
    repeat_count: usize,
    saaq_rule: &'static str,
    validation_status: &'static str,
    run_dir: String,
    manifest_path: String,
    tick_telemetry_path: String,
    latent_telemetry_path: String,
    metrics: RunMetrics,
    /// `None` when strict-repeat is disabled or `repeat_count < 2`; populated
    /// to `"matched"` / `"mismatch"` by the strict-repeat pass at the end of
    /// `main()` only when the check actually ran.
    #[serde(skip_serializing_if = "Option::is_none")]
    repeat_determinism: Option<&'static str>,
}

#[derive(Debug, Default, Serialize)]
struct RunMetrics {
    ticks_completed: usize,
    latent_rows: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    mean_tick_elapsed_us: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    first_timestamp_ms: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    last_timestamp_ms: Option<u64>,
}

/// Row buffered in-memory during a sweep, flushed once at end-of-`main`
/// into `artifacts/index.csv`. Append-only by construction: every row is
/// written exactly once, after the strict-repeat pass has had a chance to
/// stamp `repeat_determinism`.
#[derive(Debug, Clone)]
struct PendingIndexRow {
    run_id: String,
    run_tag: Option<String>,
    model_slug: String,
    model_family: String,
    telemetry_source: String,
    heartbeat_enabled: bool,
    heartbeat_slug: &'static str,
    repeat_idx: usize,
    repeat_count: usize,
    saaq_rule: &'static str,
    validation_status: &'static str,
    run_dir: String,
    ticks_completed: usize,
    latent_rows: usize,
    mean_tick_elapsed_us: Option<f64>,
    /// Absolute path to `latent_telemetry.csv` — consumed only by the
    /// strict-repeat pass; never written out to the CSV index.
    latent_path: PathBuf,
    /// Absolute path to `summary.json` for in-place determinism stamping.
    summary_path: PathBuf,
    repeat_determinism: Option<&'static str>,
}

fn heartbeat_slug_for(enabled: bool) -> &'static str {
    if enabled {
        "heartbeat_on"
    } else {
        "heartbeat_off"
    }
}

fn emit_validation_finish(
    ctx: &RunContext<'_>,
    started: Instant,
    status: &'static str,
    error: Option<&str>,
) {
    tracing::info!(
        event = "validation_finish",
        repo = "corinth-canal",
        command = "saaq_latent_calibration",
        run_id = %ctx.run_id,
        git_sha = %observability::git_sha(),
        model_slug = %ctx.spec.slug,
        heartbeat_enabled = ctx.heartbeat_enabled,
        repeat_idx = ctx.repeat_idx,
        repeat_count = ctx.repeat_count,
        ticks = ctx.ticks,
        latency_ms = started.elapsed().as_millis() as u64,
        success = status == "completed",
        error_category = observability::error_category(Some(status), error),
        "validation_finish"
    );
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
    /// Already-sanitized run tag (or `None`). Sanitization is performed
    /// once in `main()` so manifest/summary/run_id all see the same value.
    run_tag: Option<&'a str>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let _ = dotenvy::from_filename(".env.local");
    observability::init_tracing();
    let command_run_id = observability::run_id();
    let git_sha = observability::git_sha();
    let started = Instant::now();

    tracing::info!(
        event = "command_start",
        repo = "corinth-canal",
        command = "saaq_latent_calibration",
        run_id = %command_run_id,
        git_sha = %git_sha,
        "command_start"
    );

    let result = run_main();
    let error = result.as_ref().err().map(|error| error.to_string());
    tracing::info!(
        event = "command_finish",
        repo = "corinth-canal",
        command = "saaq_latent_calibration",
        run_id = %command_run_id,
        git_sha = %git_sha,
        latency_ms = started.elapsed().as_millis() as u64,
        success = result.is_ok(),
        error_category = observability::error_category(None, error.as_deref()),
        "command_finish"
    );

    result
}

fn run_main() -> Result<(), Box<dyn std::error::Error>> {
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

    // Sanitize the run tag once so run_id, manifest, and summary all agree
    // on the canonical form.
    let sanitized_tag: Option<String> = cfg.run_tag.as_deref().map(sanitize_run_tag);
    let run_tag_ref: Option<&str> = sanitized_tag.as_deref();

    // In-memory index buffer. Every successful write_manifest_and_summary
    // call at a TERMINAL status pushes a row here; the CSV is materialized
    // exactly once at the end of main() after strict-repeat stamping.
    let mut pending: Vec<PendingIndexRow> = Vec::new();

    let sweep_result =
        (|pending: &mut Vec<PendingIndexRow>| -> Result<(), Box<dyn std::error::Error>> {
            for spec in &cfg.validation_models {
                for repeat_idx in 0..cfg.repeat_count {
                    for &heartbeat_enabled in &cfg.heartbeat_matrix {
                        let run_id = build_run_id(&cfg.prompt_profile, repeat_idx, run_tag_ref);
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
                            run_tag: run_tag_ref,
                        };
                        run_validation(&ctx, pending)?;
                    }
                }
            }
            Ok(())
        })(&mut pending);

    // Strict-repeat verdict only runs on a clean sweep; partial sweeps can't
    // give a trustworthy grouping. On opt-in + success, stamp verdicts into
    // `pending` AND rewrite the referenced summary.json files in-place.
    let mut strict_mismatch = false;
    if sweep_result.is_ok() && cfg.strict_repeat_check && cfg.repeat_count >= 2 {
        strict_mismatch = apply_strict_repeat_check(&mut pending)?;
    }

    // Flush the append-only index last, so every row already carries its
    // final `repeat_determinism` verdict.
    flush_index_csv(&cfg.output_root, &pending)?;

    sweep_result?;

    if strict_mismatch {
        return Err(Error::other(
            "strict_repeat_check: latent_telemetry.csv mismatch between repeats; see summary.json files stamped repeat_determinism=\"mismatch\"",
        )
        .into());
    }

    Ok(())
}

fn run_validation(
    ctx: &RunContext<'_>,
    pending: &mut Vec<PendingIndexRow>,
) -> Result<(), Box<dyn std::error::Error>> {
    let validation_started = Instant::now();
    tracing::info!(
        event = "validation_start",
        repo = "corinth-canal",
        command = "saaq_latent_calibration",
        run_id = %ctx.run_id,
        git_sha = %observability::git_sha(),
        model_slug = %ctx.spec.slug,
        heartbeat_enabled = ctx.heartbeat_enabled,
        repeat_idx = ctx.repeat_idx,
        repeat_count = ctx.repeat_count,
        ticks = ctx.ticks,
        "validation_start"
    );

    // Pre-create the run directory so we can anchor the GPU routing
    // telemetry CSV inside it via ModelConfig and avoid polluting CWD.
    let run_dir = build_run_dir(ctx)?;
    fs::create_dir_all(&run_dir)?;
    let routing_csv_path = run_dir.join("snn_gpu_routing_telemetry.csv");

    let mut config = default_spiking_model_config(ctx.spec.path.clone(), 1);
    config.model_family = ctx.model_family_override.or(ctx.spec.family);
    config.heartbeat.enabled = ctx.heartbeat_enabled;
    config.gpu_routing_telemetry_path = Some(routing_csv_path.clone());
    // Per-model routing mode override from lineup config (Stage-campaign).
    if let Some(rm) = ctx.spec.routing_mode {
        config.routing_mode = rm;
    }
    let saaq_rule = ctx.saaq_rule;

    let mut accelerator = GpuAccelerator::new();
    let mut model = match Model::new(config.clone()) {
        Ok(model) => model,
        Err(error) => {
            let error_message = error.to_string();
            emit_validation_finish(
                ctx,
                validation_started,
                "model_setup_failed",
                Some(&error_message),
            );
            return Err(Box::new(error));
        }
    };

    if !model.router_loaded() {
        let error = format!("router did not load for checkpoint '{}'", ctx.spec.path);
        emit_validation_finish(ctx, validation_started, "router_load_failed", Some(&error));
        return Err(Error::other(error).into());
    }

    let tick_path = run_dir.join("tick_telemetry.txt");
    let latent_path = run_dir.join("latent_telemetry.csv");
    let manifest_path = run_dir.join("run_manifest.json");
    let summary_path = run_dir.join("summary.json");

    // Metrics accumulator. Populated during the tick loop; stamped into
    // summary.json at every terminal status (completed / *_failed).
    let mut metrics = RunMetrics::default();

    let target_neurons = model.projector_mut().input_neurons();
    let (prompt_embedding, prompt_embedding_source) =
        match prompt_embedding_for_validation(&ctx.spec.path, ctx.prompt_text, target_neurons) {
            Ok(result) => result,
            Err(error) => {
                let error_message = error.to_string();
                write_manifest_and_summary(
                    ctx,
                    &config,
                    &model,
                    &run_dir,
                    &manifest_path,
                    &summary_path,
                    &tick_path,
                    &latent_path,
                    "unavailable",
                    saaq_rule,
                    "prompt_embedding_failed",
                    Some(error_message.clone()),
                    &metrics,
                    collect_generated_files(
                        &manifest_path,
                        &summary_path,
                        &[&tick_path, &latent_path, &routing_csv_path],
                    ),
                )?;
                pending.push(pending_row_from_state(
                    ctx,
                    &model,
                    &run_dir,
                    &latent_path,
                    &summary_path,
                    &metrics,
                    "prompt_embedding_failed",
                ));
                emit_validation_finish(
                    ctx,
                    validation_started,
                    "prompt_embedding_failed",
                    Some(&error_message),
                );
                return Err(error);
            }
        };

    write_manifest_and_summary(
        ctx,
        &config,
        &model,
        &run_dir,
        &manifest_path,
        &summary_path,
        &tick_path,
        &latent_path,
        &prompt_embedding_source,
        saaq_rule,
        "preflight",
        None,
        &metrics,
        collect_generated_files(
            &manifest_path,
            &summary_path,
            &[&tick_path, &latent_path, &routing_csv_path],
        ),
    )?;

    if let Err(error) = model.prepare_gpu_temporal(&mut accelerator) {
        let error_message = error.to_string();
        write_manifest_and_summary(
            ctx,
            &config,
            &model,
            &run_dir,
            &manifest_path,
            &summary_path,
            &tick_path,
            &latent_path,
            &prompt_embedding_source,
            saaq_rule,
            "gpu_setup_failed",
            Some(error_message.clone()),
            &metrics,
            collect_generated_files(
                &manifest_path,
                &summary_path,
                &[&tick_path, &latent_path, &routing_csv_path],
            ),
        )?;
        pending.push(pending_row_from_state(
            ctx,
            &model,
            &run_dir,
            &latent_path,
            &summary_path,
            &metrics,
            "gpu_setup_failed",
        ));
        emit_validation_finish(
            ctx,
            validation_started,
            "gpu_setup_failed",
            Some(&error_message),
        );
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

    let mut elapsed_sum_us: u128 = 0;
    let mut elapsed_count: usize = 0;
    let run_result = (|| -> Result<(), Box<dyn std::error::Error>> {
        for tick in 0..ctx.ticks {
            let snap = telemetry_snapshot_for_tick(tick, ctx.resolved);
            let snap = heartbeat.apply(&snap, tick);
            let gain = heartbeat_gain(snap.heartbeat_signal);
            let input_spikes: Vec<f32> =
                prompt_embedding.iter().map(|value| value * gain).collect();

            // First/last timestamp bookkeeping. `telemetry_snapshot_for_tick`
            // rewrites timestamps to `tick + 1` for 1-to-1 CSV join but we
            // still want the summary to reflect the exact values emitted.
            if metrics.first_timestamp_ms.is_none() {
                metrics.first_timestamp_ms = Some(snap.timestamp_ms);
            }
            metrics.last_timestamp_ms = Some(snap.timestamp_ms);

            let started = Instant::now();
            let best_walker = model.tick_gpu_temporal(&mut accelerator, &input_spikes)?;
            let elapsed_us = started.elapsed().as_micros();
            elapsed_sum_us = elapsed_sum_us.saturating_add(elapsed_us);
            elapsed_count += 1;
            metrics.ticks_completed = elapsed_count;

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
            metrics.latent_rows += 1;

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

    // Finalize mean-tick metric (f64 microseconds). Safe for any
    // elapsed_count <= usize::MAX; `as f64` lossiness is acceptable here.
    if elapsed_count > 0 {
        metrics.mean_tick_elapsed_us = Some(elapsed_sum_us as f64 / elapsed_count as f64);
    }

    if let Err(error) = run_result {
        let error_message = error.to_string();
        let _ = latent_exporter.flush();
        let _ = tick_writer.flush();
        drop(latent_exporter);
        drop(tick_writer);
        write_manifest_and_summary(
            ctx,
            &config,
            &model,
            &run_dir,
            &manifest_path,
            &summary_path,
            &tick_path,
            &latent_path,
            &prompt_embedding_source,
            saaq_rule,
            "tick_failed",
            Some(error_message.clone()),
            &metrics,
            collect_generated_files(
                &manifest_path,
                &summary_path,
                &[&tick_path, &latent_path, &routing_csv_path],
            ),
        )?;
        pending.push(pending_row_from_state(
            ctx,
            &model,
            &run_dir,
            &latent_path,
            &summary_path,
            &metrics,
            "tick_failed",
        ));
        emit_validation_finish(ctx, validation_started, "tick_failed", Some(&error_message));
        return Err(error);
    }

    write_manifest_and_summary(
        ctx,
        &config,
        &model,
        &run_dir,
        &manifest_path,
        &summary_path,
        &tick_path,
        &latent_path,
        &prompt_embedding_source,
        saaq_rule,
        "completed",
        None,
        &metrics,
        collect_generated_files(
            &manifest_path,
            &summary_path,
            &[&tick_path, &latent_path, &routing_csv_path],
        ),
    )?;
    pending.push(pending_row_from_state(
        ctx,
        &model,
        &run_dir,
        &latent_path,
        &summary_path,
        &metrics,
        "completed",
    ));

    println!(
        "validation_complete model_slug={} heartbeat_enabled={} repeat={}/{} run_dir={}",
        ctx.spec.slug,
        ctx.heartbeat_enabled,
        ctx.repeat_idx + 1,
        ctx.repeat_count,
        run_dir.display()
    );

    emit_validation_finish(ctx, validation_started, "completed", None);

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
        run_tag: ctx.run_tag.map(|s| s.to_owned()),
        routing_mode: Some(routing_mode_label(config.routing_mode)),
        generated_files,
    }
}

fn routing_mode_label(mode: corinth_canal::moe::RoutingMode) -> &'static str {
    use corinth_canal::moe::RoutingMode::*;
    match mode {
        StubUniform => "stub_uniform",
        DenseSim => "dense_sim",
        SpikingSim => "spiking_sim",
    }
}

fn write_manifest(
    manifest_path: &PathBuf,
    manifest: ValidationManifest,
) -> Result<(), Box<dyn std::error::Error>> {
    fs::write(manifest_path, serde_json::to_string_pretty(&manifest)?)?;
    Ok(())
}

fn write_summary(
    summary_path: &std::path::Path,
    summary: &RunSummary,
) -> Result<(), Box<dyn std::error::Error>> {
    fs::write(summary_path, serde_json::to_string_pretty(summary)?)?;
    Ok(())
}

/// Convenience wrapper that builds + writes both `run_manifest.json` and
/// `summary.json`. Lives here (not in config.rs) because it pulls on
/// several run-local paths + live model state.
#[allow(clippy::too_many_arguments)]
fn write_manifest_and_summary(
    ctx: &RunContext<'_>,
    config: &corinth_canal::model::ModelConfig,
    model: &Model,
    run_dir: &std::path::Path,
    manifest_path: &PathBuf,
    summary_path: &std::path::Path,
    tick_path: &std::path::Path,
    latent_path: &std::path::Path,
    prompt_embedding_source: &str,
    saaq_rule: SaaqUpdateRule,
    validation_status: &'static str,
    error: Option<String>,
    metrics: &RunMetrics,
    generated_files: Vec<String>,
) -> Result<(), Box<dyn std::error::Error>> {
    write_manifest(
        manifest_path,
        build_manifest(
            ctx,
            config,
            model,
            run_dir,
            prompt_embedding_source,
            saaq_rule,
            validation_status,
            error,
            generated_files,
        ),
    )?;
    write_summary(
        summary_path,
        &build_summary(
            ctx,
            model.router_family(),
            run_dir,
            manifest_path,
            tick_path,
            latent_path,
            saaq_rule,
            validation_status,
            metrics,
        ),
    )?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn build_summary(
    ctx: &RunContext<'_>,
    model_family: corinth_canal::ModelFamily,
    run_dir: &std::path::Path,
    manifest_path: &std::path::Path,
    tick_path: &std::path::Path,
    latent_path: &std::path::Path,
    saaq_rule: SaaqUpdateRule,
    validation_status: &'static str,
    metrics: &RunMetrics,
) -> RunSummary {
    RunSummary {
        run_id: ctx.run_id.clone(),
        run_tag: ctx.run_tag.map(|s| s.to_owned()),
        model_slug: ctx.spec.slug.clone(),
        model_family: format!("{model_family:?}"),
        telemetry_source: ctx.resolved.source_label.clone(),
        heartbeat_enabled: ctx.heartbeat_enabled,
        repeat_idx: ctx.repeat_idx,
        repeat_count: ctx.repeat_count,
        saaq_rule: saaq_rule_label(saaq_rule),
        validation_status,
        run_dir: run_dir.to_string_lossy().into_owned(),
        manifest_path: manifest_path.to_string_lossy().into_owned(),
        tick_telemetry_path: tick_path.to_string_lossy().into_owned(),
        latent_telemetry_path: latent_path.to_string_lossy().into_owned(),
        metrics: RunMetrics {
            ticks_completed: metrics.ticks_completed,
            latent_rows: metrics.latent_rows,
            mean_tick_elapsed_us: metrics.mean_tick_elapsed_us,
            first_timestamp_ms: metrics.first_timestamp_ms,
            last_timestamp_ms: metrics.last_timestamp_ms,
        },
        repeat_determinism: None,
    }
}

fn saaq_rule_label(rule: SaaqUpdateRule) -> &'static str {
    match rule {
        SaaqUpdateRule::LegacyV1_0 => "LegacyV1_0",
        SaaqUpdateRule::SaaqV1_5SqrtRate => "SaaqV1_5SqrtRate",
    }
}

/// Build a `PendingIndexRow` from the live model + run state. Mirrors the
/// shape of the CSV header and keeps paths that `apply_strict_repeat_check`
/// and `flush_index_csv` both need.
fn pending_row_from_state(
    ctx: &RunContext<'_>,
    model: &Model,
    run_dir: &Path,
    latent_path: &Path,
    summary_path: &Path,
    metrics: &RunMetrics,
    validation_status: &'static str,
) -> PendingIndexRow {
    PendingIndexRow {
        run_id: ctx.run_id.clone(),
        run_tag: ctx.run_tag.map(|s| s.to_owned()),
        model_slug: ctx.spec.slug.clone(),
        model_family: format!("{:?}", model.router_family()),
        telemetry_source: ctx.resolved.source_label.clone(),
        heartbeat_enabled: ctx.heartbeat_enabled,
        heartbeat_slug: heartbeat_slug_for(ctx.heartbeat_enabled),
        repeat_idx: ctx.repeat_idx,
        repeat_count: ctx.repeat_count,
        saaq_rule: saaq_rule_label(ctx.saaq_rule),
        validation_status,
        run_dir: run_dir.to_string_lossy().into_owned(),
        ticks_completed: metrics.ticks_completed,
        latent_rows: metrics.latent_rows,
        mean_tick_elapsed_us: metrics.mean_tick_elapsed_us,
        latent_path: latent_path.to_path_buf(),
        summary_path: summary_path.to_path_buf(),
        repeat_determinism: None,
    }
}

/// Build the `generated_files` list stamped into `run_manifest.json`,
/// including only files that actually exist on disk at the moment of the
/// call. `manifest_path` and `summary_path` are added unconditionally:
/// they are about to be (or have just been) written by the calling
/// `write_manifest_and_summary`. Optional paths are filtered by
/// `Path::exists()` so failure-path manifests no longer claim files like
/// `tick_telemetry.txt` or `latent_telemetry.csv` when those terminal
/// statuses (`prompt_embedding_failed`, `gpu_setup_failed`) bail out
/// before the corresponding writers are opened.
fn collect_generated_files(
    manifest_path: &Path,
    summary_path: &Path,
    optional_paths: &[&Path],
) -> Vec<String> {
    let mut files = Vec::with_capacity(2 + optional_paths.len());
    files.push(
        manifest_path
            .file_name()
            .unwrap()
            .to_string_lossy()
            .into_owned(),
    );
    files.push(
        summary_path
            .file_name()
            .unwrap()
            .to_string_lossy()
            .into_owned(),
    );
    for path in optional_paths {
        if path.exists() {
            files.push(path.file_name().unwrap().to_string_lossy().into_owned());
        }
    }
    files
}

const INDEX_CSV_HEADER: &str = "run_id,run_tag,model_slug,model_family,telemetry_source,heartbeat_enabled,repeat_idx,repeat_count,saaq_rule,validation_status,run_dir,ticks_completed,latent_rows,mean_tick_elapsed_us,repeat_determinism";

/// Append every buffered row to `<output_root>/index.csv`. The file is
/// opened once per `main()` invocation with `append(true)`; if it's empty
/// (new file or zero-byte), the header is emitted first.
///
/// If an existing `index.csv` has a header that does not match the current
/// `INDEX_CSV_HEADER` (e.g. left over from a prior schema), it is rotated
/// out of the way to `index.csv.legacy-<unix_ts>` so new rows never get
/// appended under a stale, mismatched header. The legacy file is preserved
/// untouched for offline migration.
fn flush_index_csv(
    output_root: &Path,
    rows: &[PendingIndexRow],
) -> Result<(), Box<dyn std::error::Error>> {
    if rows.is_empty() {
        return Ok(());
    }
    fs::create_dir_all(output_root)?;
    let path = output_root.join("index.csv");
    let needs_header = ensure_index_header_compatible(&path)?;
    let file = OpenOptions::new().create(true).append(true).open(&path)?;
    let mut writer = BufWriter::new(file);
    if needs_header {
        writeln!(writer, "{INDEX_CSV_HEADER}")?;
    }
    for row in rows {
        writeln!(writer, "{}", format_index_row(row))?;
    }
    writer.flush()?;
    Ok(())
}

/// Inspect `path` to decide whether `flush_index_csv` should write a fresh
/// header. Returns `Ok(true)` for missing / empty files and for files whose
/// existing header matches the current schema slot-for-slot. Returns
/// `Ok(true)` after rotating a schema-mismatched file out of the way to
/// `index.csv.legacy-<unix_ts>`. Returns `Ok(false)` only when the existing
/// file already has the correct header and rows can be appended safely.
fn ensure_index_header_compatible(path: &Path) -> Result<bool, Box<dyn std::error::Error>> {
    let file = match File::open(path) {
        Ok(f) => f,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(true),
        Err(err) => return Err(err.into()),
    };
    let mut first_line = String::new();
    BufReader::new(file).read_line(&mut first_line)?;
    let trimmed = first_line.trim_end_matches(['\r', '\n']);
    if trimmed.is_empty() {
        return Ok(true);
    }
    if trimmed == INDEX_CSV_HEADER {
        return Ok(false);
    }
    let suffix = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let backup = path.with_file_name(format!("index.csv.legacy-{suffix}"));
    fs::rename(path, &backup)?;
    eprintln!(
        "saaq_latent_calibration: rotated incompatible index.csv schema -> {}",
        backup.display()
    );
    Ok(true)
}

fn format_index_row(row: &PendingIndexRow) -> String {
    let mean_us = row
        .mean_tick_elapsed_us
        .map(|v| format!("{v:.3}"))
        .unwrap_or_default();
    let tag = row.run_tag.as_deref().unwrap_or("");
    let determinism = row.repeat_determinism.unwrap_or("");
    let fields = [
        csv_escape(&row.run_id),
        csv_escape(tag),
        csv_escape(&row.model_slug),
        csv_escape(&row.model_family),
        csv_escape(&row.telemetry_source),
        row.heartbeat_enabled.to_string(),
        row.repeat_idx.to_string(),
        row.repeat_count.to_string(),
        csv_escape(row.saaq_rule),
        csv_escape(row.validation_status),
        csv_escape(&row.run_dir),
        row.ticks_completed.to_string(),
        row.latent_rows.to_string(),
        mean_us,
        csv_escape(determinism),
    ];
    fields.join(",")
}

/// Minimal RFC-4180-style escaping: wrap in double quotes iff the field
/// contains a comma, quote, CR, or LF, and double any embedded quotes.
fn csv_escape(s: &str) -> String {
    if s.bytes().any(|b| matches!(b, b',' | b'"' | b'\r' | b'\n')) {
        let mut out = String::with_capacity(s.len() + 2);
        out.push('"');
        for ch in s.chars() {
            if ch == '"' {
                out.push_str("\"\"");
            } else {
                out.push(ch);
            }
        }
        out.push('"');
        out
    } else {
        s.to_owned()
    }
}

/// Strict-repeat verdict pass. Groups rows by
/// `(model_slug, telemetry_source, heartbeat_slug, saaq_rule)` and compares
/// each repeat `k >= 1`'s `latent_telemetry.csv` to repeat `0`'s byte-wise.
///
/// Only rows with `validation_status == "completed"` participate — partial
/// / failed runs can't meaningfully claim determinism. Groups with fewer
/// than two completed repeats are skipped silently (they can't mismatch).
///
/// Side effects:
///   - Stamps `repeat_determinism` on every participating row in-place.
///   - Rewrites each participating row's `summary.json` with the verdict.
///
/// Returns `true` if any mismatch was recorded.
fn apply_strict_repeat_check(
    rows: &mut [PendingIndexRow],
) -> Result<bool, Box<dyn std::error::Error>> {
    type GroupKey = (String, String, &'static str, &'static str);
    let mut groups: BTreeMap<GroupKey, Vec<usize>> = BTreeMap::new();
    for (idx, row) in rows.iter().enumerate() {
        if row.validation_status != "completed" {
            continue;
        }
        let key = (
            row.model_slug.clone(),
            row.telemetry_source.clone(),
            row.heartbeat_slug,
            row.saaq_rule,
        );
        groups.entry(key).or_default().push(idx);
    }

    let mut any_mismatch = false;
    for (_, indices) in groups {
        if indices.len() < 2 {
            continue;
        }
        // Order repeats by repeat_idx so we always compare against the
        // canonical repeat 0.
        let mut sorted = indices.clone();
        sorted.sort_by_key(|i| rows[*i].repeat_idx);
        let baseline_idx = sorted[0];
        let baseline_bytes = fs::read(&rows[baseline_idx].latent_path)?;

        let mut group_mismatch = false;
        for &i in sorted.iter().skip(1) {
            let candidate_bytes = fs::read(&rows[i].latent_path)?;
            if candidate_bytes != baseline_bytes {
                rows[i].repeat_determinism = Some("mismatch");
                rewrite_summary_determinism(&rows[i].summary_path, "mismatch")?;
                group_mismatch = true;
                any_mismatch = true;
            }
        }
        if group_mismatch {
            // Mark baseline as mismatch too so the group has a single verdict.
            rows[baseline_idx].repeat_determinism = Some("mismatch");
            rewrite_summary_determinism(&rows[baseline_idx].summary_path, "mismatch")?;
        } else {
            for &i in &sorted {
                rows[i].repeat_determinism = Some("matched");
                rewrite_summary_determinism(&rows[i].summary_path, "matched")?;
            }
        }
    }
    Ok(any_mismatch)
}

/// Rewrite `summary.json` in place to stamp `repeat_determinism`. We load
/// the JSON, mutate one field, and write it back pretty-printed. This is
/// the only rewrite path in the whole runner and only fires when
/// strict-repeat check is active and produced a verdict.
fn rewrite_summary_determinism(
    summary_path: &Path,
    verdict: &'static str,
) -> Result<(), Box<dyn std::error::Error>> {
    let raw = fs::read_to_string(summary_path)?;
    let mut value: serde_json::Value = serde_json::from_str(&raw)?;
    if let Some(obj) = value.as_object_mut() {
        obj.insert(
            "repeat_determinism".to_owned(),
            serde_json::Value::String(verdict.to_owned()),
        );
    }
    fs::write(summary_path, serde_json::to_string_pretty(&value)?)?;
    Ok(())
}

fn build_run_id(prompt_profile: &str, repeat_idx: usize, run_tag: Option<&str>) -> String {
    let timestamp = format_local_timestamp_compact();
    match run_tag {
        Some(tag) if !tag.is_empty() => {
            format!("{timestamp}_{prompt_profile}_r{repeat_idx}_{tag}")
        }
        _ => format!("{timestamp}_{prompt_profile}_r{repeat_idx}"),
    }
}

/// Keep only `[A-Za-z0-9._-]` from the raw tag; any other character becomes
/// `_`. Runs of `_` are collapsed so the result never produces the visual
/// `__` join with the preceding `r<idx>_`. Empty / whitespace-only -> `""`.
fn sanitize_run_tag(raw: &str) -> String {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return String::new();
    }
    let mut out = String::with_capacity(trimmed.len());
    let mut last_was_underscore = false;
    for ch in trimmed.chars() {
        let ok = ch.is_ascii_alphanumeric() || matches!(ch, '.' | '_' | '-');
        let c = if ok { ch } else { '_' };
        if c == '_' {
            if !last_was_underscore {
                out.push('_');
            }
            last_was_underscore = true;
        } else {
            out.push(c);
            last_was_underscore = false;
        }
    }
    // Trim leading/trailing underscores introduced by sanitization so we
    // never append `_r0__tag` or `_r0_tag_`.
    out.trim_matches('_').to_owned()
}

fn build_run_dir(ctx: &RunContext<'_>) -> Result<PathBuf, Box<dyn std::error::Error>> {
    Ok(ctx
        .output_root
        .join(&ctx.spec.slug)
        .join(&ctx.resolved.source_label)
        .join(heartbeat_slug_for(ctx.heartbeat_enabled))
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
    format!("{year:04}{month:02}{day:02}T{hour:02}{minute:02}{second:02}")
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
