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
#[path = "support/spikenaut.rs"]
mod spikenaut;
#[allow(dead_code)]
#[path = "support/telemetry_csv.rs"]
mod telemetry_csv;

use std::env;
use std::path::{Path, PathBuf};
use std::process;

use spikenaut::{SpikenautDomain, ingest_jsonl, run_dual_saaq_cpu_smoke, write_canonical_csv};
use telemetry_csv::{TELEMETRY_CSV_HEADER, load_csv_telemetry_rows};

struct Args {
    input: PathBuf,
    output: PathBuf,
    domain: Option<SpikenautDomain>,
    limit: Option<usize>,
    smoke: bool,
    output_root: PathBuf,
}

fn main() {
    let args = match parse_args(env::args().skip(1)) {
        Ok(args) => args,
        Err(message) => {
            eprintln!("{message}");
            eprintln!(
                "Usage: spikenaut_ingest <input.jsonl> [output.csv] [--domain auto|gpu|mining|hft|qubic|state] [--limit N] [--smoke] [--output-root DIR]"
            );
            process::exit(1);
        }
    };

    if let Err(error) = run(&args) {
        eprintln!("spikenaut_ingest failed: {error}");
        process::exit(1);
    }
}

fn run(args: &Args) -> Result<(), Box<dyn std::error::Error>> {
    let ingested = ingest_jsonl(&args.input, args.domain, args.limit)?;
    if ingested.rows.is_empty() {
        return Err(std::io::Error::other(format!(
            "no mappable rows in '{}' (malformed={}, unmapped={})",
            args.input.display(),
            ingested.skipped_malformed,
            ingested.skipped_unmapped
        ))
        .into());
    }

    write_canonical_csv(&args.output, &ingested.rows)?;
    let reloaded = load_csv_telemetry_rows(&args.output)?;
    if reloaded.len() != ingested.rows.len() {
        return Err(std::io::Error::other(format!(
            "round-trip mismatch: wrote {} rows, canonical loader accepted {}",
            ingested.rows.len(),
            reloaded.len()
        ))
        .into());
    }

    println!(
        "domain={} rows={} skipped_malformed={} skipped_unmapped={} csv={} header={TELEMETRY_CSV_HEADER}",
        ingested.domain.as_str(),
        ingested.rows.len(),
        ingested.skipped_malformed,
        ingested.skipped_unmapped,
        args.output.display()
    );

    if args.smoke {
        let run_dir = args
            .output_root
            .join(ingested.domain.source_slug())
            .join("dual_saaq_smoke");
        let manifest = run_dual_saaq_cpu_smoke(
            &ingested.rows,
            &run_dir,
            ingested.domain,
            Some(&args.output),
        )?;
        println!(
            "smoke run_id={} telemetry={} saaq_dual_emit={} latent_rows={} run_dir={}",
            manifest.run_id,
            manifest.telemetry_source,
            manifest.saaq_dual_emit,
            ingested.rows.len(),
            run_dir.display()
        );
    }
    Ok(())
}

fn parse_args<I, S>(args: I) -> Result<Args, String>
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    let mut positional = Vec::new();
    let mut domain = None;
    let mut limit = None;
    let mut smoke = false;
    let mut output_root = PathBuf::from("artifacts");
    let mut items = args.into_iter().peekable();
    while let Some(raw) = items.next() {
        let arg = raw.as_ref();
        match arg {
            "--help" | "-h" => {
                return Err("convert Spikenaut JSONL to canonical replay CSV".to_owned());
            }
            "--smoke" => smoke = true,
            "--domain" => {
                let value = items
                    .next()
                    .ok_or_else(|| "missing value for --domain".to_owned())?;
                let value = value.as_ref();
                if value.eq_ignore_ascii_case("auto") {
                    domain = None;
                } else {
                    domain = Some(
                        SpikenautDomain::from_alias(value)
                            .ok_or_else(|| format!("unknown --domain '{value}'"))?,
                    );
                }
            }
            "--limit" => {
                let value = items
                    .next()
                    .ok_or_else(|| "missing value for --limit".to_owned())?;
                limit = Some(
                    value
                        .as_ref()
                        .parse()
                        .map_err(|_| format!("invalid --limit '{}'", value.as_ref()))?,
                );
            }
            "--output-root" => {
                let value = items
                    .next()
                    .ok_or_else(|| "missing value for --output-root".to_owned())?;
                output_root = PathBuf::from(value.as_ref());
            }
            flag if flag.starts_with('-') => {
                return Err(format!("unknown flag '{flag}'"));
            }
            other => positional.push(PathBuf::from(other)),
        }
    }

    let input = positional
        .first()
        .cloned()
        .ok_or_else(|| "missing <input.jsonl>".to_owned())?;
    let output = if let Some(path) = positional.get(1) {
        path.clone()
    } else {
        default_output_csv(&input)
    };
    Ok(Args {
        input,
        output,
        domain,
        limit,
        smoke,
        output_root,
    })
}

fn default_output_csv(input: &Path) -> PathBuf {
    input.with_extension("csv")
}
