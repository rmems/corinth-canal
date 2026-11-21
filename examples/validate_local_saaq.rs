// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Dry-run validation for the local SAAQ experiment matrix.
//!
//! Usage:
//!   cargo run --example validate_local_saaq --no-default-features -- <matrix.toml> [--check-paths]
//!   cargo run --example validate_local_saaq --no-default-features -- --count-usable-rows <telemetry.csv>
//!
//! `--count-usable-rows` prints the number of data rows the SAAQ runner will
//! actually replay from a canonical telemetry CSV, using the exact loader
//! (`support/telemetry_csv.rs::load_csv_telemetry_rows`) the runner uses, and exits 0. Rows
//! that look numeric but the loader rejects (e.g. `1e39` overflows `f32`)
//! are not counted. `just saaq-campaign` uses this for its TICKS preflight
//! (RM-1449). Exits 1 if the file is missing or the header is wrong.
//!
//! Validates matrix entries for:
//! - Structural correctness (RunMatrix::validate)
//! - Policy and source format value checks (strict, not just warnings)
//! - Path existence (when --check-paths is given, with env var resolution)
//!
//! Exits 0 on success, 1 on validation failure.

// Path-include only the CSV loader so this example stays in the CPU
// `--no-default-features` set (the full `mod support` tree is cuda-gated).
#[allow(dead_code)]
#[path = "support/telemetry_csv.rs"]
mod telemetry_csv;

use corinth_canal::RunMatrix;
use std::env;
use std::fs;
use std::path::Path;
use std::process;

fn resolve_env_path(raw: &str) -> (String, bool) {
    if let Some(env_val) = raw.strip_prefix("$")
        && let Some(var_name) = env_val.split('/').next()
    {
        if let Ok(val) = std::env::var(var_name) {
            return (raw.replacen(&format!("${}", var_name), &val, 1), true);
        }
        return (raw.to_string(), false);
    }
    (raw.to_string(), true)
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.get(1).map(String::as_str) == Some("--count-usable-rows") {
        let Some(csv_path) = args.get(2) else {
            eprintln!("Usage: validate_local_saaq --count-usable-rows <telemetry.csv>");
            process::exit(1);
        };
        match telemetry_csv::load_csv_telemetry_rows(Path::new(csv_path)) {
            Ok(rows) => {
                println!("{}", rows.len());
                return;
            }
            Err(error) => {
                eprintln!("Error: {error}");
                process::exit(1);
            }
        }
    }
    if args.len() < 2 {
        eprintln!("Usage: validate_local_saaq <matrix.toml> [--check-paths]");
        process::exit(1);
    }

    let matrix_path = &args[1];
    let check_paths = args.iter().any(|a| a == "--check-paths");

    let contents = match fs::read_to_string(matrix_path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Error reading '{}': {}", matrix_path, e);
            process::exit(1);
        }
    };

    let matrix: RunMatrix = match toml::from_str(&contents) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("Error parsing '{}': {}", matrix_path, e);
            process::exit(1);
        }
    };

    println!("Loaded {} run(s) from '{}'", matrix.runs.len(), matrix_path);

    if let Err(e) = matrix.validate() {
        eprintln!("Validation failed: {}", e);
        process::exit(1);
    }

    let mut errors = Vec::new();
    let mut enabled = 0;
    let mut skipped = 0;

    for run in &matrix.runs {
        if run.skip_reason.is_some() {
            skipped += 1;
            continue;
        }
        enabled += 1;

        if !["gguf", "safetensors", "custom_artifact"].contains(&run.source_format.as_str()) {
            errors.push(format!(
                "run '{}': invalid source_format '{}' (expected gguf|safetensors|custom_artifact)",
                run.run_id, run.source_format
            ));
        }

        if !["protect", "quantize", "drop"].contains(&run.router_policy.as_str()) {
            errors.push(format!(
                "run '{}': invalid router_policy '{}' (expected protect|quantize|drop)",
                run.run_id, run.router_policy
            ));
        }

        if !["protect", "quantize", "drop"].contains(&run.norm_policy.as_str()) {
            errors.push(format!(
                "run '{}': invalid norm_policy '{}' (expected protect|quantize|drop)",
                run.run_id, run.norm_policy
            ));
        }

        if check_paths {
            let (resolved, env_ok) = resolve_env_path(&run.model_id_or_path);
            if !env_ok {
                if let Some(env_val) = run.model_id_or_path.strip_prefix("$") {
                    let var_name = env_val.split('/').next().unwrap_or("?");
                    errors.push(format!(
                        "run '{}': env var '{}' not set (from model_id_or_path '{}')",
                        run.run_id, var_name, run.model_id_or_path
                    ));
                }
            } else if run.source_format != "custom_artifact" {
                let model_path = Path::new(&resolved);
                if !model_path.exists() {
                    errors.push(format!(
                        "run '{}': model path not found: {} (resolved from '{}')",
                        run.run_id, resolved, run.model_id_or_path
                    ));
                }
            }

            let (resolved_output, output_env_ok) = resolve_env_path(&run.output_root);
            if output_env_ok {
                let output_root = Path::new(&resolved_output);
                if output_root.exists() {
                    let run_dir = output_root.join(&run.run_id);
                    if run_dir.exists() {
                        errors.push(format!(
                            "run '{}': output directory already exists: {}",
                            run.run_id,
                            run_dir.display()
                        ));
                    }
                }
            }
        }
    }

    if errors.is_empty() {
        println!("Validation passed.");
    } else {
        eprintln!("Validation failed with {} error(s):", errors.len());
        for e in &errors {
            eprintln!("  - {}", e);
        }
        process::exit(1);
    }

    println!("  Enabled runs: {}", enabled);
    println!("  Skipped runs: {}", skipped);
}
