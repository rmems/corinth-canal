// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Convert Spikenaut-SNN-Telemetry JSONL into canonical corinth replay CSV.
//!
//! Usage:
//!   cargo run --example spikenaut_ingest --no-default-features -- \
//!     <input.jsonl> [output.csv] [--domain auto|gpu|mining|hft|qubic|state] \
//!     [--limit N] [--smoke] [--output-root DIR]
//!
//! `--smoke` runs a CPU dual-SAAQ (1.0 + 1.5) loop over the mapped rows and
//! writes `run_manifest.json` under `--output-root` (default `artifacts`).
//!
//! Path-includes only the CSV/ingest helpers so this example stays in the
//! CPU `--no-default-features` set without compiling the Sentry/OpenSSL
//! observability stack (the other `mod support` examples are cuda-gated).

#[allow(dead_code)]
#[path = "support/spikenaut/mod.rs"]
mod spikenaut;
#[allow(dead_code)]
#[path = "support/telemetry_csv.rs"]
mod telemetry_csv;

use std::process;

use spikenaut::cli::Cli;
use spikenaut::{ingest_jsonl, run_dual_saaq_cpu_smoke, write_canonical_csv};
use telemetry_csv::{TELEMETRY_CSV_HEADER, load_csv_telemetry_rows};

fn main() {
    match load_cli() {
        Ok(cli) => {
            if let Err(error) = run(&cli) {
                eprintln!("spikenaut_ingest failed: {error}");
                process::exit(1);
            }
        }
        Err(message) => {
            eprintln!("{message}");
            eprintln!(
                "Usage: spikenaut_ingest <input.jsonl> [output.csv] [--domain auto|gpu|mining|hft|qubic|state] [--limit N] [--smoke] [--output-root DIR]"
            );
            process::exit(1);
        }
    }
}

fn load_cli() -> Result<Cli, String> {
    let bytes = std::fs::read("/proc/self/cmdline").map_err(|error| error.to_string())?;
    spikenaut::cli::parse_argv(spikenaut::cli::tokens_from_cmdline(&bytes)?)
}

fn run(cli: &Cli) -> Result<(), Box<dyn std::error::Error>> {
    let ingested = ingest_mapped_rows(cli)?;
    write_and_reload_csv(&cli.output, &ingested.rows)?;
    print_ingest_report(&ingested, &cli.output);
    maybe_run_smoke(cli, &ingested)
}

fn ingest_mapped_rows(cli: &Cli) -> Result<spikenaut::IngestResult, Box<dyn std::error::Error>> {
    let ingested = ingest_jsonl(&cli.input, cli.domain, cli.limit)?;
    if ingested.rows.is_empty() {
        return Err(std::io::Error::other(format!(
            "no mappable rows in '{}' (malformed={}, unmapped={})",
            cli.input.display(),
            ingested.skipped_malformed,
            ingested.skipped_unmapped
        ))
        .into());
    }
    Ok(ingested)
}

fn write_and_reload_csv(
    output: &std::path::Path,
    rows: &[corinth_canal::TelemetrySnapshot],
) -> Result<(), Box<dyn std::error::Error>> {
    write_canonical_csv(output, rows)?;
    let reloaded = load_csv_telemetry_rows(output)?;
    if reloaded.len() != rows.len() {
        return Err(std::io::Error::other(format!(
            "round-trip mismatch: wrote {} rows, canonical loader accepted {}",
            rows.len(),
            reloaded.len()
        ))
        .into());
    }
    Ok(())
}

fn print_ingest_report(ingested: &spikenaut::IngestResult, output: &std::path::Path) {
    println!(
        "domain={} rows={} skipped_malformed={} skipped_unmapped={} csv={} header={TELEMETRY_CSV_HEADER}",
        ingested.domain.as_str(),
        ingested.rows.len(),
        ingested.skipped_malformed,
        ingested.skipped_unmapped,
        output.display()
    );
}

fn maybe_run_smoke(
    cli: &Cli,
    ingested: &spikenaut::IngestResult,
) -> Result<(), Box<dyn std::error::Error>> {
    if !cli.smoke {
        return Ok(());
    }
    let run_dir = cli
        .output_root
        .join(ingested.domain.source_slug())
        .join("dual_saaq_smoke");
    let manifest = run_dual_saaq_cpu_smoke(
        &ingested.rows,
        &run_dir,
        ingested.domain,
        Some(&cli.output),
        &cli.output_root,
    )?;
    println!(
        "smoke run_id={} telemetry={} saaq_dual_emit={} latent_rows={} run_dir={}",
        manifest.run_id,
        manifest.telemetry_source,
        manifest.saaq_dual_emit,
        ingested.rows.len(),
        run_dir.display()
    );
    Ok(())
}
