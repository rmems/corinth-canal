// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Integration harness for `examples/support/spikenaut.rs`.
//!
//! Example targets do not run unit-test harnesses, so this `#[path]` include
//! is what makes the ingest/mapping tests execute under
//! `cargo test --no-default-features`.

#[path = "../examples/support/spikenaut.rs"]
mod spikenaut;
#[path = "../examples/support/telemetry_csv.rs"]
mod telemetry_csv;

use std::path::PathBuf;

use spikenaut::{
    SPIKENAUT_CSV_HEADER, SpikenautDomain, ingest_jsonl, run_dual_saaq_cpu_smoke,
    write_canonical_csv,
};
use telemetry_csv::{TELEMETRY_CSV_HEADER, load_csv_telemetry_rows};

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/spikenaut")
        .join(name)
}

fn scratch_dir() -> PathBuf {
    if let Some(dir) = std::env::var_os("CARGO_TARGET_TMPDIR") {
        return PathBuf::from(dir);
    }
    let dir = PathBuf::from("target").join("tmp-tests");
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

#[test]
fn ingest_header_matches_canonical_loader() {
    assert_eq!(SPIKENAUT_CSV_HEADER, TELEMETRY_CSV_HEADER);
}

#[test]
fn ingest_gpu_fixture_drops_null_power_and_stamps_row_index() {
    let ingested = ingest_jsonl(&fixture("gpu_sample.jsonl"), None, None).unwrap();
    assert_eq!(ingested.domain, SpikenautDomain::Gpu);
    assert_eq!(ingested.rows.len(), 4, "null power_w row must be dropped");
    assert_eq!(ingested.skipped_unmapped, 1);
    assert_eq!(ingested.skipped_malformed, 0);
    assert_eq!(ingested.rows[0].timestamp_ms, 12159);
    assert_eq!(ingested.rows[3].timestamp_ms, 58270);
    assert!((ingested.rows[2].gpu_temp_c - 44.0).abs() < 1e-4);
}

#[test]
fn ingest_gpu_csv_round_trips_through_canonical_loader() {
    let ingested = ingest_jsonl(&fixture("gpu_sample.jsonl"), None, None).unwrap();
    let csv_path = scratch_dir().join(format!(
        "spikenaut_gpu_{}.csv",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    write_canonical_csv(&csv_path, &ingested.rows).unwrap();
    let header = std::fs::read_to_string(&csv_path)
        .unwrap()
        .lines()
        .next()
        .unwrap()
        .to_owned();
    assert_eq!(header, TELEMETRY_CSV_HEADER);
    let reloaded = load_csv_telemetry_rows(&csv_path).unwrap();
    assert_eq!(reloaded.len(), ingested.rows.len());
    assert_eq!(reloaded[0].timestamp_ms, ingested.rows[0].timestamp_ms);
    assert!((reloaded[2].gpu_power_w - ingested.rows[2].gpu_power_w).abs() < 1e-4);
    let _ = std::fs::remove_file(csv_path);
}

#[test]
fn ingest_mining_hft_qubic_domains() {
    let mining = ingest_jsonl(&fixture("mining_sample.jsonl"), None, None).unwrap();
    assert_eq!(mining.domain, SpikenautDomain::Mining);
    assert_eq!(mining.rows.len(), 4);
    assert_eq!(mining.rows[2].timestamp_ms, 1_773_921_536_380);

    let hft = ingest_jsonl(&fixture("hft_sample.jsonl"), None, None).unwrap();
    assert_eq!(hft.domain, SpikenautDomain::Hft);
    assert_eq!(hft.rows.len(), 4);
    assert!(hft.rows[0].gpu_temp_c > 400.0);

    let qubic = ingest_jsonl(&fixture("qubic_sample.jsonl"), None, None).unwrap();
    assert_eq!(qubic.domain, SpikenautDomain::Qubic);
    assert_eq!(qubic.rows.len(), 4);
    // Independent signals only: first row tick_trace≈0.0076, tick_rate=0.5333.
    // `gpu_temp_c_derived=75` / `power_w_derived=400` must not be copied through.
    assert!((qubic.rows[0].gpu_temp_c - 0.762634).abs() < 1e-3);
    assert!((qubic.rows[0].gpu_power_w - 213.32).abs() < 1e-2);
    assert!((qubic.rows[0].gpu_temp_c - 75.0).abs() > 1.0);
}

#[test]
fn dual_saaq_cpu_smoke_writes_manifest_and_both_rule_columns() {
    let ingested = ingest_jsonl(
        &fixture("gpu_sample.jsonl"),
        Some(SpikenautDomain::Gpu),
        None,
    )
    .unwrap();
    let run_dir = scratch_dir().join(format!(
        "spikenaut_smoke_{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let csv_path = run_dir.join("spikenaut_gpu.csv");
    std::fs::create_dir_all(&run_dir).unwrap();
    write_canonical_csv(&csv_path, &ingested.rows).unwrap();

    let manifest =
        run_dual_saaq_cpu_smoke(&ingested.rows, &run_dir, ingested.domain, Some(&csv_path))
            .unwrap();
    assert!(manifest.saaq_dual_emit);
    assert_eq!(manifest.telemetry_source, "csv_spikenaut_gpu");
    assert_eq!(manifest.run_tag.as_deref(), Some("spikenaut_gpu"));
    assert_eq!(manifest.saaq_rule, "SaaqV1_5SqrtRate");
    assert_eq!(manifest.validation_status, "completed");
    assert_eq!(manifest.ticks, 4);

    let latent = std::fs::read_to_string(run_dir.join("latent_telemetry.csv")).unwrap();
    let mut lines = latent.lines();
    let header = lines.next().unwrap();
    assert!(header.contains("saaq_delta_q_legacy_target"));
    assert!(header.contains("saaq_delta_q_v15_target"));
    let data_rows: Vec<&str> = lines.filter(|line| !line.is_empty()).collect();
    assert_eq!(data_rows.len(), 4);
    for row in data_rows {
        let cols: Vec<&str> = row.split(',').collect();
        assert_eq!(cols.len(), 14);
        assert!(cols[10].parse::<f32>().is_ok());
        assert!(cols[11].parse::<f32>().is_ok());
        assert!(cols[12].parse::<f32>().is_ok());
        assert!(cols[13].parse::<f32>().is_ok());
    }

    let parsed: corinth_canal::ExperimentManifest =
        serde_json::from_str(&std::fs::read_to_string(run_dir.join("run_manifest.json")).unwrap())
            .unwrap();
    assert!(parsed.saaq_dual_emit);
    let _ = std::fs::remove_dir_all(run_dir);
}
