// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Summarize SAAQ run artifacts for a sprint-level report.
//!
//! Usage:
//!   cargo run --example summarize_local_saaq -- <runs-dir>
//!
//! Recursively reads all `run_manifest.json` / `summary.json` files under
//! `<runs-dir>` and emits a markdown summary table to stdout.

use corinth_canal::{ExperimentManifest, experiment::artifact_io};
use std::env;
use std::fs;
use std::path::Path;

fn classify_status(validation: &str, has_manifest: bool) -> String {
    if validation == "completed" || validation == "ok" {
        "ok".into()
    } else if validation == "error"
        || validation == "manifest_unreadable"
        || validation.ends_with("_failed")
        || validation == "tick_failed"
    {
        "error".into()
    } else if has_manifest {
        validation.into()
    } else {
        "missing".into()
    }
}

fn collect_run_summaries(runs_dir: &Path, entries: &mut Vec<RunSummary>) {
    if let Ok(subdirs) = fs::read_dir(runs_dir) {
        for entry in subdirs.flatten() {
            let path = entry.path();
            if !path.is_dir() {
                continue;
            }

            let manifest_path = path.join("run_manifest.json");

            if manifest_path.exists() {
                let summary_path = path.join("summary.json");
                // An unreadable or truncated manifest is an error row, not a
                // silent skip: the runner now writes atomically (RM-1447), so
                // a bad file means something outside the runner touched it.
                let (manifest, read_error) =
                    match artifact_io::read_json::<ExperimentManifest>(&manifest_path) {
                        Ok(manifest) => (Some(manifest), None),
                        Err(error) => (None, Some(error.to_string())),
                    };
                if let Some(error) = &read_error {
                    eprintln!("error: unreadable run_manifest.json: {error}");
                }

                let ticks = manifest.as_ref().map(|m| m.ticks).unwrap_or(0);
                let model_slug = manifest
                    .as_ref()
                    .map(|m| m.model_slug.clone())
                    .unwrap_or_else(|| "?".into());
                let model_family = manifest
                    .as_ref()
                    .map(|m| m.model_family.clone())
                    .unwrap_or_else(|| "?".into());
                let validation = manifest
                    .as_ref()
                    .map(|m| m.validation_status.clone())
                    .unwrap_or_else(|| "manifest_unreadable".into());

                let run_id = path
                    .file_name()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .into_owned();

                let has_manifest = true;
                let has_summary = summary_path.exists();
                let has_telemetry = path.join("tick_telemetry.txt").exists()
                    || path.join("latent_telemetry.csv").exists();

                let status = classify_status(&validation, has_manifest);

                entries.push(RunSummary {
                    run_id,
                    manifest_unreadable: read_error.is_some(),
                    model_slug,
                    model_family,
                    ticks,
                    status,
                    has_manifest,
                    has_summary,
                    has_telemetry,
                });
            } else {
                collect_run_summaries(&path, entries);
            }
        }
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() != 2 {
        eprintln!("Usage: summarize_local_saaq <runs-dir>");
        std::process::exit(1);
    }

    let runs_dir = Path::new(&args[1]);
    if !runs_dir.is_dir() {
        eprintln!("Error: '{}' is not a directory", runs_dir.display());
        std::process::exit(1);
    }

    let mut entries: Vec<RunSummary> = Vec::new();
    collect_run_summaries(runs_dir, &mut entries);

    if entries.is_empty() {
        println!("No run directories found under `{}`.", runs_dir.display());
        return;
    }

    println!("# SAAQ Sprint Summary");
    println!();
    println!("**Runs directory:** `{}`", runs_dir.display());
    println!("**Total runs:** {}", entries.len());
    println!();

    let ok_count = entries.iter().filter(|e| e.status == "ok").count();
    let err_count = entries.iter().filter(|e| e.status == "error").count();
    let missing_count = entries.iter().filter(|e| e.status == "missing").count();
    let other_count = entries.len() - ok_count - err_count - missing_count;

    println!("| Status | Count |");
    println!("|--------|-------|");
    println!("| ok | {} |", ok_count);
    println!("| error | {} |", err_count);
    println!("| missing | {} |", missing_count);
    if other_count > 0 {
        println!("| other | {} |", other_count);
    }
    println!();

    println!("| Run ID | Model | Family | Ticks | Status | Manifest | Summary | Telemetry |");
    println!("|--------|-------|--------|-------|--------|----------|---------|-----------|");
    for e in &entries {
        println!(
            "| {} | {} | {} | {} | {} | {} | {} | {} |",
            e.run_id,
            e.model_slug,
            e.model_family,
            e.ticks,
            e.status,
            if e.has_manifest { "yes" } else { "no" },
            if e.has_summary { "yes" } else { "no" },
            if e.has_telemetry { "yes" } else { "no" },
        );
    }

    let unreadable = entries.iter().filter(|e| e.manifest_unreadable).count();
    if unreadable > 0 {
        eprintln!(
            "error: {unreadable} run_manifest.json file(s) could not be read; see messages above"
        );
        std::process::exit(2);
    }
}

struct RunSummary {
    run_id: String,
    manifest_unreadable: bool,
    model_slug: String,
    model_family: String,
    ticks: usize,
    status: String,
    has_manifest: bool,
    has_summary: bool,
    has_telemetry: bool,
}
