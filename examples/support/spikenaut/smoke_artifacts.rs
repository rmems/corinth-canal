// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Smoke-run manifest and summary artifacts.

use std::path::{Path, PathBuf};

use corinth_canal::{ExperimentManifest, ExperimentMetrics, ExperimentSummary};

use super::domain::SpikenautDomain;
use super::timestamp::format_system_time_rfc3339_utc;

pub(super) struct SmokePaths {
    pub latent: PathBuf,
    pub tick: PathBuf,
    pub manifest: PathBuf,
    pub summary: PathBuf,
}

impl SmokePaths {
    pub(super) fn new(run_dir: &Path) -> Self {
        Self {
            latent: run_dir.join("latent_telemetry.csv"),
            tick: run_dir.join("tick_telemetry.txt"),
            manifest: run_dir.join("run_manifest.json"),
            summary: run_dir.join("summary.json"),
        }
    }
}

pub(super) fn smoke_manifest(
    rows: &[corinth_canal::TelemetrySnapshot],
    run_dir: &Path,
    domain: SpikenautDomain,
    csv_path: Option<&Path>,
    output_root: &Path,
) -> ExperimentManifest {
    let mut manifest = smoke_manifest_identity(run_dir, domain, rows.len());
    apply_smoke_paths(&mut manifest, run_dir, csv_path, output_root);
    manifest
}

fn smoke_run_id(run_dir: &Path) -> String {
    run_dir
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("spikenaut_dual_saaq_smoke")
        .to_owned()
}

fn smoke_manifest_identity(
    run_dir: &Path,
    domain: SpikenautDomain,
    ticks: usize,
) -> ExperimentManifest {
    ExperimentManifest {
        run_id: smoke_run_id(run_dir),
        run_tag: Some(format!("spikenaut_{}", domain.as_str())),
        created_at: format_system_time_rfc3339_utc(std::time::SystemTime::now()),
        repo: "corinth-canal".to_owned(),
        commit_sha: None,
        model_slug: "stub_olmoe".to_owned(),
        model_family: "Olmoe".to_owned(),
        architecture: "stub".to_owned(),
        checkpoint_path: String::new(),
        checkpoint_format: "gguf".to_owned(),
        routing_tensor_name: "synthetic".to_owned(),
        synapse_source: "synthetic-fallback".to_owned(),
        prompt_embedding_source: "none".to_owned(),
        prompt_profile: "spikenaut_ingest".to_owned(),
        prompt_text: None,
        ticks,
        saaq_rule: "SaaqV1_5SqrtRate".to_owned(),
        saaq_primary_rule: "SaaqV1_5SqrtRate".to_owned(),
        saaq_dual_emit: true,
        telemetry_source: format!("csv_{}", domain.source_slug()),
        telemetry_csv_path: None,
        telemetry_row_count: Some(ticks),
        wraparound_enabled: false,
        wraparound_loops: 0,
        ticks_effective: ticks,
        run_dir: String::new(),
        output_root: String::new(),
        repeat_idx: 0,
        repeat_count: 1,
        validation_status: "completed".to_owned(),
        error: None,
        routing_mode: Some("stub_uniform".to_owned()),
        projection_mode: Some("spiking_ternary".to_owned()),
        lineup_declared_count: None,
        lineup_resolved_count: None,
        generated_files: smoke_generated_files(),
    }
}

fn apply_smoke_paths(
    manifest: &mut ExperimentManifest,
    run_dir: &Path,
    csv_path: Option<&Path>,
    output_root: &Path,
) {
    manifest.telemetry_csv_path = csv_path.map(|p| p.to_string_lossy().into_owned());
    manifest.run_dir = run_dir.to_string_lossy().into_owned();
    manifest.output_root = output_root.to_string_lossy().into_owned();
}

fn smoke_generated_files() -> Vec<String> {
    vec![
        "run_manifest.json".to_owned(),
        "summary.json".to_owned(),
        "tick_telemetry.txt".to_owned(),
        "latent_telemetry.csv".to_owned(),
    ]
}

pub(super) fn write_smoke_summary(
    paths: &SmokePaths,
    manifest: &ExperimentManifest,
    metrics: ExperimentMetrics,
) -> Result<(), Box<dyn std::error::Error>> {
    let summary = ExperimentSummary {
        run_id: manifest.run_id.clone(),
        run_tag: manifest.run_tag.clone(),
        model_slug: manifest.model_slug.clone(),
        model_family: manifest.model_family.clone(),
        telemetry_source: manifest.telemetry_source.clone(),
        repeat_idx: 0,
        repeat_count: 1,
        saaq_rule: manifest.saaq_rule.clone(),
        projection_mode: manifest.projection_mode.clone(),
        validation_status: "completed".to_owned(),
        run_dir: manifest.run_dir.clone(),
        manifest_path: paths.manifest.to_string_lossy().into_owned(),
        tick_telemetry_path: paths.tick.to_string_lossy().into_owned(),
        latent_telemetry_path: paths.latent.to_string_lossy().into_owned(),
        metrics,
        repeat_determinism: None,
    };
    std::fs::write(&paths.summary, serde_json::to_string_pretty(&summary)?)?;
    Ok(())
}
