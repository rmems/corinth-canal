// SPDX-License-Identifier: Apache-2.0 OR MIT
//! CPU dual-SAAQ (1.0 + 1.5) smoke over already-mapped snapshots.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use corinth_canal::moe::{Router, RoutingMode};
use corinth_canal::projector::{ProjectionMode, Projector};
use corinth_canal::types::ModelOutput;
use corinth_canal::{
    ExperimentManifest, ExperimentMetrics, FUNNEL_HIDDEN_NEURONS, SaaqUpdateRule,
    SnnDualLatentCalibrator, SnnLatentCsvExporter, TelemetryFunnel,
};

use super::domain::SpikenautDomain;
use super::smoke_artifacts::{SmokePaths, smoke_manifest, write_smoke_summary};

/// CPU dual-SAAQ (1.0 + 1.5) smoke over already-mapped snapshots.
///
/// Uses the funnel + projector + stub router — no CUDA, no checkpoint.
/// Writes `run_manifest.json`, `summary.json`, `latent_telemetry.csv`, and
/// `tick_telemetry.txt` under `run_dir`.
pub fn run_dual_saaq_cpu_smoke(
    rows: &[corinth_canal::TelemetrySnapshot],
    run_dir: &Path,
    domain: SpikenautDomain,
    csv_path: Option<&Path>,
    output_root: &Path,
) -> Result<ExperimentManifest, Box<dyn std::error::Error>> {
    if rows.is_empty() {
        return Err(std::io::Error::other("dual-SAAQ smoke needs at least one mapped row").into());
    }
    std::fs::create_dir_all(run_dir)?;
    let paths = SmokePaths::new(run_dir);
    clear_previous_completion(&paths)?;
    prepare_data_artifacts(&paths)?;
    let metrics = run_smoke_ticks(rows, &paths)?;
    let manifest = smoke_manifest(rows, run_dir, domain, csv_path, output_root);
    publish_smoke_completion(&paths, &manifest, metrics)?;
    Ok(manifest)
}

fn prepare_data_artifacts(paths: &SmokePaths) -> std::io::Result<()> {
    for path in [&paths.latent, &paths.tick] {
        match std::fs::symlink_metadata(path) {
            Ok(metadata) if metadata.file_type().is_symlink() => {
                return Err(std::io::Error::other(format!(
                    "smoke artifact '{}' must not be a symlink",
                    path.display()
                )));
            }
            Ok(metadata) if metadata.is_file() => std::fs::remove_file(path)?,
            Ok(_) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => return Err(error),
        }
    }
    Ok(())
}

fn publish_smoke_completion(
    paths: &SmokePaths,
    manifest: &ExperimentManifest,
    metrics: ExperimentMetrics,
) -> Result<(), Box<dyn std::error::Error>> {
    let manifest_json = serde_json::to_string_pretty(manifest)?;
    if let Err(error) = write_smoke_summary(paths, manifest, metrics) {
        let _ = std::fs::remove_file(&paths.summary);
        return Err(error);
    }
    if let Err(error) = std::fs::write(&paths.manifest, manifest_json) {
        let _ = std::fs::remove_file(&paths.summary);
        let _ = std::fs::remove_file(&paths.manifest);
        return Err(error.into());
    }
    Ok(())
}

fn clear_previous_completion(paths: &SmokePaths) -> std::io::Result<()> {
    for path in [&paths.manifest, &paths.summary] {
        match std::fs::remove_file(path) {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => return Err(error),
        }
    }
    Ok(())
}

struct SmokeEngine {
    funnel: TelemetryFunnel,
    projector: Projector,
    router: Router,
    calibrator: SnnDualLatentCalibrator,
}

impl SmokeEngine {
    fn new() -> Result<Self, Box<dyn std::error::Error>> {
        const THRESHOLDS: [f32; 4] = [1.0, 5.0, 1.0, 5.0];
        Ok(Self {
            funnel: TelemetryFunnel::new(THRESHOLDS, 8),
            projector: Projector::new(ProjectionMode::SpikingTernary),
            router: Router::load_with_mode("", 8, 2, RoutingMode::StubUniform)?,
            calibrator: SnnDualLatentCalibrator::new(SaaqUpdateRule::SaaqV1_5SqrtRate),
        })
    }
}

fn run_smoke_ticks(
    rows: &[corinth_canal::TelemetrySnapshot],
    paths: &SmokePaths,
) -> Result<ExperimentMetrics, Box<dyn std::error::Error>> {
    let mut engine = SmokeEngine::new()?;
    let mut exporter = SnnLatentCsvExporter::create(&paths.latent)?;
    let mut tick_writer = open_tick_writer(&paths.tick)?;
    let mut metrics = empty_smoke_metrics();
    write_smoke_tick_rows(
        rows,
        &mut engine,
        &mut exporter,
        &mut tick_writer,
        &mut metrics,
    )?;
    finish_smoke_writers(&mut exporter, &mut tick_writer)?;
    Ok(metrics)
}

fn open_tick_writer(path: &Path) -> Result<BufWriter<File>, Box<dyn std::error::Error>> {
    let mut tick_writer = BufWriter::new(File::create(path)?);
    writeln!(
        tick_writer,
        "tick,timestamp_ms,gpu_temp_c,gpu_power_w,cpu_tctl_c,cpu_package_power_w,ternary"
    )?;
    Ok(tick_writer)
}

fn empty_smoke_metrics() -> ExperimentMetrics {
    ExperimentMetrics {
        ticks_completed: 0,
        latent_rows: 0,
        ..ExperimentMetrics::default()
    }
}

fn write_smoke_tick_rows<W: Write>(
    rows: &[corinth_canal::TelemetrySnapshot],
    engine: &mut SmokeEngine,
    exporter: &mut SnnLatentCsvExporter,
    tick_writer: &mut W,
    metrics: &mut ExperimentMetrics,
) -> Result<(), Box<dyn std::error::Error>> {
    for (tick, snap) in rows.iter().enumerate() {
        let mut replay_snap = snap.clone();
        // Match telemetry_snapshot_for_tick so smoke and CSV campaigns use the
        // same elapsed-time basis while the source CSV keeps its timestamps.
        replay_snap.timestamp_ms = tick as u64 + 1;
        process_smoke_tick(engine, exporter, tick_writer, tick, &replay_snap, metrics)?;
    }
    Ok(())
}

fn finish_smoke_writers<W: Write>(
    exporter: &mut SnnLatentCsvExporter,
    tick_writer: &mut W,
) -> Result<(), Box<dyn std::error::Error>> {
    exporter.flush()?;
    tick_writer.flush()?;
    Ok(())
}

fn process_smoke_tick<W: Write>(
    engine: &mut SmokeEngine,
    exporter: &mut SnnLatentCsvExporter,
    tick_writer: &mut W,
    tick: usize,
    snap: &corinth_canal::TelemetrySnapshot,
    metrics: &mut ExperimentMetrics,
) -> Result<(), Box<dyn std::error::Error>> {
    let activity = engine.funnel.encode_snapshot(snap);
    let embedding = engine.projector.project(
        &activity.spike_train,
        &activity.potentials,
        &activity.iz_potentials,
    )?;
    let routed = engine.router.forward(&embedding)?;
    let output = stub_model_output(&activity, embedding, routed);
    let latent = engine.calibrator.observe(snap, &activity, &output)?;
    exporter.write_row(&latent)?;
    write_tick_line(tick_writer, tick, snap, &activity.ternary_events)?;
    record_smoke_metrics(metrics, snap);
    Ok(())
}

fn stub_model_output(
    activity: &corinth_canal::FunnelActivity,
    embedding: Vec<f32>,
    routed: corinth_canal::moe::RouterOutput,
) -> ModelOutput {
    ModelOutput {
        spike_train: activity.spike_train.clone(),
        firing_rates: vec![0.0; FUNNEL_HIDDEN_NEURONS],
        membrane_potentials: activity.potentials.clone(),
        embedding,
        expert_weights: Some(routed.expert_weights),
        selected_experts: Some(routed.selected_experts),
        reasoning: None,
    }
}

fn write_tick_line<W: Write>(
    tick_writer: &mut W,
    tick: usize,
    snap: &corinth_canal::TelemetrySnapshot,
    ternary_events: &impl std::fmt::Debug,
) -> std::io::Result<()> {
    writeln!(
        tick_writer,
        "{},{},{:.6},{:.6},{:.6},{:.6},{:?}",
        tick + 1,
        snap.timestamp_ms,
        snap.gpu_temp_c,
        snap.gpu_power_w,
        snap.cpu_tctl_c,
        snap.cpu_package_power_w,
        ternary_events
    )
}

fn record_smoke_metrics(metrics: &mut ExperimentMetrics, snap: &corinth_canal::TelemetrySnapshot) {
    if metrics.first_timestamp_ms.is_none() {
        metrics.first_timestamp_ms = Some(snap.timestamp_ms);
    }
    metrics.last_timestamp_ms = Some(snap.timestamp_ms);
    metrics.ticks_completed += 1;
    metrics.latent_rows += 1;
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch_run_dir(label: &str) -> std::path::PathBuf {
        let scratch = std::env::var_os("CARGO_TARGET_TMPDIR")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| std::path::PathBuf::from("target").join("tmp-tests"));
        scratch.join(format!(
            "spikenaut_{label}_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ))
    }

    #[test]
    fn summary_write_failure_does_not_publish_completed_manifest() {
        let run_dir = scratch_run_dir("summary_failure");
        std::fs::create_dir_all(&run_dir).unwrap();
        let paths = SmokePaths::new(&run_dir);
        std::fs::create_dir(&paths.summary).unwrap();
        let manifest = smoke_manifest(&[], &run_dir, SpikenautDomain::Gpu, None, &run_dir);

        assert!(publish_smoke_completion(&paths, &manifest, ExperimentMetrics::default()).is_err());
        assert!(!paths.manifest.exists());
        std::fs::remove_dir_all(run_dir).unwrap();
    }

    #[test]
    fn manifest_write_failure_removes_completed_summary() {
        let run_dir = scratch_run_dir("manifest_failure");
        std::fs::create_dir_all(&run_dir).unwrap();
        let paths = SmokePaths::new(&run_dir);
        std::fs::create_dir(&paths.manifest).unwrap();
        let manifest = smoke_manifest(&[], &run_dir, SpikenautDomain::Gpu, None, &run_dir);

        assert!(publish_smoke_completion(&paths, &manifest, ExperimentMetrics::default()).is_err());
        assert!(!paths.summary.exists());
        std::fs::remove_dir_all(run_dir).unwrap();
    }
}
