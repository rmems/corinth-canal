// SPDX-License-Identifier: Apache-2.0 OR MIT
//! CSV replay example: ingest canonical telemetry CSV into Model.
//!
//! Canonical CSV format: timestamp_ms,gpu_temp_c,gpu_power_w,cpu_tctl_c,cpu_package_power_w

mod support;

use support::RunConfig;
use support::default_spiking_model_config;
use support::observability::{self, CommandObserver, SafeDiagnosticData};

use corinth_canal::{
    EMBEDDING_DIM, FUNNEL_HIDDEN_NEURONS, HybridError, TelemetryFunnel, model::Model,
    telemetry::TelemetrySnapshot,
};

const EXPECTED_HEADER: &str = "timestamp_ms,gpu_temp_c,gpu_power_w,cpu_tctl_c,cpu_package_power_w";
const TELEMETRY_THRESHOLDS: [f32; 4] = [1.0, 5.0, 1.0, 5.0];

fn parse_u64(v: &str) -> Option<u64> {
    v.parse::<u64>().ok()
}

fn parse_f32(v: &str) -> Option<f32> {
    let n = v.parse::<f32>().ok()?;
    if n.is_finite() { Some(n) } else { None }
}

fn mean_squared_error(output: &[f32], target: &[f32]) -> f32 {
    let len = output.len().min(target.len());
    if len == 0 {
        return 0.0;
    }
    let sum = output
        .iter()
        .zip(target.iter())
        .take(len)
        .map(|(o, t)| {
            let d = o - t;
            d * d
        })
        .sum::<f32>();
    sum / len as f32
}

fn main() -> corinth_canal::Result<()> {
    let _ = dotenvy::from_filename(".env.local");

    // Validate args before initializing Sentry/observer so usage errors exit cleanly.
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: cargo run --example csv_replay <telemetry.csv>");
        eprintln!("CSV format: {EXPECTED_HEADER}");
        std::process::exit(1);
    }

    let _sentry_guard = observability::init_sentry("csv_replay");
    let observer = observability::start_command("csv_replay");
    let result = run(&observer, &args);
    observer.finish(&result, SafeDiagnosticData::default());
    result
}

fn run(observer: &CommandObserver, args: &[String]) -> corinth_canal::Result<()> {
    let csv_path = &args[1];
    let run_cfg = RunConfig::from_env();
    let mut safe = SafeDiagnosticData::default().with_telemetry_source("csv");
    if let Some(model_slug) = observability::checkpoint_slug(&run_cfg.checkpoint_path) {
        safe = safe.with_model_slug(&model_slug);
        observer.annotate(safe);
    } else {
        observer.annotate(safe);
    }

    let cfg = default_spiking_model_config(run_cfg.checkpoint_path.clone(), 20);
    let mut model = Model::new_with_projector_neurons(cfg.clone(), FUNNEL_HIDDEN_NEURONS)?;
    let mut funnel = TelemetryFunnel::new(TELEMETRY_THRESHOLDS, cfg.snn_steps);
    let csv_content = std::fs::read_to_string(csv_path)?;
    let mut lines = csv_content.lines();
    let header = lines
        .next()
        .ok_or_else(|| HybridError::InvalidConfig("empty CSV file".to_owned()))?
        .trim();
    validate_header(header)?;
    process_rows(&mut model, &mut funnel, lines.map(String::from))?;
    Ok(())
}

fn validate_header(header: &str) -> corinth_canal::Result<()> {
    if header != EXPECTED_HEADER {
        return Err(HybridError::InvalidConfig(format!(
            "invalid CSV header: expected '{EXPECTED_HEADER}', got '{header}'"
        )));
    }
    Ok(())
}

fn dummy_snap() -> TelemetrySnapshot {
    TelemetrySnapshot {
        timestamp_ms: 0,
        gpu_temp_c: 0.0,
        gpu_power_w: 0.0,
        cpu_tctl_c: 0.0,
        cpu_package_power_w: 0.0,
    }
}

fn process_one_line(
    model: &mut Model,
    funnel: &mut TelemetryFunnel,
    target: &[f32],
    line: &str,
    line_number: usize,
) -> corinth_canal::Result<(bool, usize, usize, f32, TelemetrySnapshot, [i8; 4])> {
    if line.is_empty() {
        return Ok((true, 0, 0, 0.0, dummy_snap(), [0; 4]));
    }
    let fields: Vec<&str> = line.split(',').collect();
    if fields.len() != 5 {
        eprintln!(
            "Skipping malformed row {}: expected 5 columns, got {}",
            line_number,
            fields.len(),
        );
        return Ok((true, 0, 0, 0.0, dummy_snap(), [0; 4]));
    }
    let snap = match parse_csv_row(fields, line_number) {
        Some(s) => s,
        None => {
            eprintln!(
                "Skipping malformed row {}: parse/finite check failed",
                line_number
            );
            return Ok((true, 0, 0, 0.0, dummy_snap(), [0; 4]));
        }
    };
    let activity = funnel.encode_snapshot(&snap);
    let ternary = activity.ternary_events;
    let output = model.forward_activity(
        &activity.spike_train,
        &activity.potentials,
        &activity.iz_potentials,
    )?;
    let input_spikes = activity
        .input_spike_train
        .iter()
        .map(Vec::len)
        .sum::<usize>();
    let hidden_spikes = activity.spike_train.iter().map(Vec::len).sum::<usize>();
    let loss = mean_squared_error(output.embedding.as_slice(), target);
    Ok((false, input_spikes, hidden_spikes, loss, snap, ternary))
}

fn process_rows(
    model: &mut Model,
    funnel: &mut TelemetryFunnel,
    lines: impl Iterator<Item = String>,
) -> corinth_canal::Result<()> {
    let mut total_loss = 0.0_f32;
    let mut rows_processed = 0_usize;
    let mut rows_skipped = 0_usize;
    let mut total_input_spikes = 0_usize;
    let mut total_hidden_spikes = 0_usize;
    let target = vec![0.0_f32; EMBEDDING_DIM];

    for (idx, raw_line) in lines.enumerate() {
        let line_number = idx + 2;
        let (skip, input_spikes, hidden_spikes, loss, snap, ternary) =
            process_one_line(model, funnel, &target, raw_line.trim(), line_number)?;
        if skip {
            rows_skipped += 1;
            continue;
        }
        total_loss += loss;
        rows_processed += 1;
        total_input_spikes += input_spikes;
        total_hidden_spikes += hidden_spikes;
        if rows_processed.is_multiple_of(100) || rows_processed <= 5 {
            println!(
                "step={:>4} gpu_temp={:5.1}C gpu_power={:6.1}W cpu_temp={:5.1}C ternary={:?} input_spikes={:>3} hidden_spikes={:>4} loss={:.6}",
                rows_processed,
                snap.gpu_temp_c,
                snap.gpu_power_w,
                snap.cpu_tctl_c,
                ternary,
                input_spikes,
                hidden_spikes,
                loss
            );
        }
    }
    print_summary(
        rows_processed,
        rows_skipped,
        total_loss,
        total_input_spikes,
        total_hidden_spikes,
        model,
    );
    Ok(())
}

fn parse_csv_row(fields: Vec<&str>, _line_number: usize) -> Option<TelemetrySnapshot> {
    let timestamp_ms = parse_u64(fields[0])?;
    let gpu_temp_c = parse_f32(fields[1])?;
    let gpu_power_w = parse_f32(fields[2])?;
    let cpu_tctl_c = parse_f32(fields[3])?;
    let cpu_package_power_w = parse_f32(fields[4])?;
    Some(TelemetrySnapshot {
        timestamp_ms,
        gpu_temp_c,
        gpu_power_w,
        cpu_tctl_c,
        cpu_package_power_w,
    })
}

fn print_summary(
    rows_processed: usize,
    rows_skipped: usize,
    total_loss: f32,
    total_input_spikes: usize,
    total_hidden_spikes: usize,
    model: &Model,
) {
    let avg_loss = if rows_processed > 0 {
        total_loss / rows_processed as f32
    } else {
        0.0
    };
    println!("\n=== Replay Summary ===");
    println!("rows_processed={}", rows_processed);
    println!("rows_skipped={}", rows_skipped);
    println!("avg_loss={:.6}", avg_loss);
    println!(
        "avg_input_spikes_per_row={:.3}",
        if rows_processed > 0 {
            total_input_spikes as f32 / rows_processed as f32
        } else {
            0.0
        }
    );
    println!(
        "avg_hidden_spikes_per_row={:.3}",
        if rows_processed > 0 {
            total_hidden_spikes as f32 / rows_processed as f32
        } else {
            0.0
        }
    );
    println!("global_step={}", model.global_step());
    println!("router_loaded={}", model.router_loaded());
}
