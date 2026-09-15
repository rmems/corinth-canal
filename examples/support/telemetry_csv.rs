// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Telemetry CSV load/replay helpers for example binaries.
//!
//! Self-contained (public `corinth_canal` types + `std` only) so integration
//! tests can `#[path]`-include this file under `--no-default-features`.

use std::io::Error;
use std::path::{Path, PathBuf};

/// Default is [`TelemetrySource::Synthetic`] so a fresh clone never silently
/// depends on a machine-specific CSV path. CSV replay is opt-in via
/// `TELEMETRY_SOURCE=csv` for RE4/Cyberpunk corpus generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TelemetrySource {
    Synthetic,
    Csv,
}

/// Telemetry state resolved once per process and reused by every tick.
///
/// `source_label` is what lands in the directory path and manifest: one of
/// `synthetic`, `synthetic_fallback`, or `csv_<stem>` (e.g. `csv_re4` for
/// `telemetry.csv`). `rows` is only populated on a successful CSV load.
#[derive(Debug, Clone)]
pub struct ResolvedTelemetry {
    pub source: TelemetrySource,
    pub source_label: String,
    pub csv_path: Option<PathBuf>,
    pub rows: Option<Vec<corinth_canal::TelemetrySnapshot>>,
}

impl ResolvedTelemetry {
    #[cfg(feature = "cuda")]
    pub fn row_count(&self) -> Option<usize> {
        self.rows.as_ref().map(|rows| rows.len())
    }
}

/// Canonical replay header consumed by `load_csv_telemetry_rows` and produced
/// by `gaming-telemetry` `export_csv` and the Spikenaut JSONL adapter.
pub const TELEMETRY_CSV_HEADER: &str =
    "timestamp_ms,gpu_temp_c,gpu_power_w,cpu_tctl_c,cpu_package_power_w";

/// Parse a canonical telemetry CSV exported by `gaming-telemetry` into a
/// vector of [`corinth_canal::TelemetrySnapshot`] ready for replay.
///
/// Fails fast on header mismatch; silently skips malformed data rows (counted
/// in the returned log line via stderr) so a few dropped samples don't abort
/// a 2000-row sweep.
pub fn load_csv_telemetry_rows(
    path: &Path,
) -> Result<Vec<corinth_canal::TelemetrySnapshot>, Box<dyn std::error::Error>> {
    let contents = std::fs::read_to_string(path).map_err(|error| {
        Error::other(format!(
            "telemetry CSV '{}' could not be read: {error}",
            path.display()
        ))
    })?;
    let mut lines = contents.lines();
    validate_telemetry_csv_header(path, lines.next())?;
    let (rows, skipped) = collect_telemetry_csv_rows(lines);
    if skipped > 0 {
        eprintln!(
            "load_csv_telemetry_rows: skipped {skipped} malformed row(s) in '{}'",
            path.display()
        );
    }
    Ok(rows)
}

fn collect_telemetry_csv_rows<'a>(
    lines: impl Iterator<Item = &'a str>,
) -> (Vec<corinth_canal::TelemetrySnapshot>, usize) {
    let mut rows = Vec::new();
    let mut skipped = 0usize;
    for raw_line in lines {
        match parse_telemetry_csv_data_line(raw_line) {
            ParseDataLine::Empty => {}
            ParseDataLine::Ok(snap) => rows.push(snap),
            ParseDataLine::Malformed => skipped += 1,
        }
    }
    (rows, skipped)
}

fn validate_telemetry_csv_header(
    path: &Path,
    header_line: Option<&str>,
) -> Result<(), Box<dyn std::error::Error>> {
    let header = header_line
        .ok_or_else(|| Error::other(format!("telemetry CSV '{}' is empty", path.display())))?
        .trim();
    if header != TELEMETRY_CSV_HEADER {
        return Err(Error::other(format!(
            "telemetry CSV '{}' header mismatch: expected '{TELEMETRY_CSV_HEADER}', got '{header}'",
            path.display()
        ))
        .into());
    }
    Ok(())
}

enum ParseDataLine {
    Empty,
    Ok(corinth_canal::TelemetrySnapshot),
    Malformed,
}

fn parse_telemetry_csv_data_line(raw_line: &str) -> ParseDataLine {
    let line = raw_line.trim();
    if line.is_empty() {
        return ParseDataLine::Empty;
    }
    let fields: Vec<&str> = line.split(',').collect();
    if fields.len() != 5 {
        return ParseDataLine::Malformed;
    }
    let Some(timestamp_ms) = fields[0].parse::<u64>().ok() else {
        return ParseDataLine::Malformed;
    };
    let Some(gpu_temp_c) = parse_finite_f32(fields[1]) else {
        return ParseDataLine::Malformed;
    };
    let Some(gpu_power_w) = parse_finite_f32(fields[2]) else {
        return ParseDataLine::Malformed;
    };
    let Some(cpu_tctl_c) = parse_finite_f32(fields[3]) else {
        return ParseDataLine::Malformed;
    };
    let Some(cpu_package_power_w) = parse_finite_f32(fields[4]) else {
        return ParseDataLine::Malformed;
    };
    ParseDataLine::Ok(corinth_canal::TelemetrySnapshot {
        timestamp_ms,
        gpu_temp_c,
        gpu_power_w,
        cpu_tctl_c,
        cpu_package_power_w,
    })
}

/// Resolve telemetry from an already-selected source and CSV path.
///
/// Env/path resolution stays in `examples/support/mod.rs` (machine-local
/// config boundary). For `Csv`, loads and validates the CSV up front; if the
/// file is missing, malformed, or empty, emits a single warning to stderr and
/// degrades to `Synthetic`, stamping `source_label = "synthetic_fallback"` so
/// the manifest faithfully records what actually happened.
pub fn resolve_telemetry_from(source: TelemetrySource, csv_path: PathBuf) -> ResolvedTelemetry {
    match source {
        TelemetrySource::Synthetic => ResolvedTelemetry {
            source: TelemetrySource::Synthetic,
            source_label: "synthetic".to_string(),
            csv_path: None,
            rows: None,
        },
        TelemetrySource::Csv => match load_csv_telemetry_rows(&csv_path) {
            Ok(rows) if !rows.is_empty() => {
                let label = csv_source_label(&csv_path);
                ResolvedTelemetry {
                    source: TelemetrySource::Csv,
                    source_label: label,
                    csv_path: Some(csv_path),
                    rows: Some(rows),
                }
            }
            Ok(_) => {
                eprintln!(
                    "resolve_telemetry_from: CSV '{}' is empty; falling back to synthetic",
                    csv_path.display()
                );
                ResolvedTelemetry {
                    source: TelemetrySource::Synthetic,
                    source_label: "synthetic_fallback".to_string(),
                    csv_path: Some(csv_path),
                    rows: None,
                }
            }
            Err(error) => {
                eprintln!(
                    "resolve_telemetry_from: CSV '{}' failed to load: {error}; falling back to synthetic",
                    csv_path.display()
                );
                ResolvedTelemetry {
                    source: TelemetrySource::Synthetic,
                    source_label: "synthetic_fallback".to_string(),
                    csv_path: Some(csv_path),
                    rows: None,
                }
            }
        },
    }
}

/// Convert a CSV path into a directory-safe source slug. The stem
/// `telemetry` is treated as the canonical RE4 corpus and renders as
/// `csv_re4`; any other stem becomes `csv_<stem>`.
fn csv_source_label(path: &Path) -> String {
    let stem = path
        .file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or("unknown")
        .to_ascii_lowercase();
    if stem == "telemetry" {
        "csv_re4".to_string()
    } else {
        let sanitized = stem.replace([' ', '.'], "_");
        format!("csv_{sanitized}")
    }
}

/// Produce the telemetry snapshot for a given tick. For CSV replay this
/// wraps around when `tick >= rows.len()`; the caller is responsible for
/// warning when `TICKS > row_count` (see `saaq_latent_calibration`).
///
/// `timestamp_ms` is always rewritten to `tick + 1` so the resulting latent
/// CSV joins 1-to-1 against `tick_telemetry.txt` on tick index regardless
/// of the underlying CSV's absolute timestamps.
pub fn telemetry_snapshot_for_tick(
    tick: usize,
    resolved: &ResolvedTelemetry,
) -> corinth_canal::TelemetrySnapshot {
    let mut snap = match (resolved.source, resolved.rows.as_ref()) {
        (TelemetrySource::Csv, Some(rows)) if !rows.is_empty() => {
            let idx = tick % rows.len();
            rows[idx].clone()
        }
        _ => synthetic_base_snapshot(tick),
    };
    snap.timestamp_ms = (tick as u64) + 1;
    snap
}

fn parse_finite_f32(value: &str) -> Option<f32> {
    let parsed = value.parse::<f32>().ok()?;
    if parsed.is_finite() {
        Some(parsed)
    } else {
        None
    }
}

pub fn synthetic_base_snapshot(tick: usize) -> corinth_canal::TelemetrySnapshot {
    let phase = tick as f32 * 0.041;
    corinth_canal::TelemetrySnapshot {
        gpu_temp_c: 68.0 + phase.sin() * 2.8,
        gpu_power_w: 232.0 + phase.cos() * 11.5,
        cpu_tctl_c: 73.0 + (phase * 0.9).sin() * 2.2,
        cpu_package_power_w: 116.0 + (phase * 1.1).cos() * 7.4,
        timestamp_ms: tick as u64,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Cargo sets `CARGO_TARGET_TMPDIR` for test sandboxes; avoid shared
    /// `std::env::temp_dir()` (Semgrep: multi-user temp dir).
    fn test_scratch_dir() -> PathBuf {
        if let Some(dir) = std::env::var_os("CARGO_TARGET_TMPDIR") {
            return PathBuf::from(dir);
        }
        let dir = PathBuf::from("target").join("tmp-tests");
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn write_temp_csv(name: &str, contents: &str) -> PathBuf {
        let path = test_scratch_dir().join(format!(
            "corinth_canal_support_{}_{}.csv",
            name,
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::write(&path, contents).unwrap();
        path
    }

    #[test]
    fn load_csv_accepts_canonical_header_and_parses_rows() {
        let csv = "timestamp_ms,gpu_temp_c,gpu_power_w,cpu_tctl_c,cpu_package_power_w\n\
                   1000,60.5,250.0,70.0,120.0\n\
                   2000,61.0,252.5,70.5,121.5\n";
        let path = write_temp_csv("canonical", csv);
        let rows = load_csv_telemetry_rows(&path).unwrap();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].timestamp_ms, 1000);
        assert!((rows[0].gpu_temp_c - 60.5).abs() < 1e-6);
        assert_eq!(rows[1].timestamp_ms, 2000);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn load_csv_rejects_bad_header() {
        let csv = "t,gpu,gpuw,cpu,cpuw\n1000,60,250,70,120\n";
        let path = write_temp_csv("bad_header", csv);
        let err = load_csv_telemetry_rows(&path).unwrap_err();
        assert!(err.to_string().contains("header mismatch"));
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn load_csv_skips_malformed_rows() {
        let csv = "timestamp_ms,gpu_temp_c,gpu_power_w,cpu_tctl_c,cpu_package_power_w\n\
                   1000,60.5,250.0,70.0,120.0\n\
                   malformed,row\n\
                   2000,NaN,250.0,70.0,120.0\n\
                   3000,61.0,252.5,70.5,121.5\n";
        let path = write_temp_csv("skip_bad", csv);
        let rows = load_csv_telemetry_rows(&path).unwrap();
        assert_eq!(rows.len(), 2, "expected only the two fully-valid rows");
        assert_eq!(rows[0].timestamp_ms, 1000);
        assert_eq!(rows[1].timestamp_ms, 3000);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn telemetry_snapshot_for_tick_wraps_around_csv_rows() {
        let rows = vec![
            corinth_canal::TelemetrySnapshot {
                timestamp_ms: 111,
                gpu_temp_c: 10.0,
                gpu_power_w: 100.0,
                cpu_tctl_c: 20.0,
                cpu_package_power_w: 200.0,
            },
            corinth_canal::TelemetrySnapshot {
                timestamp_ms: 222,
                gpu_temp_c: 30.0,
                gpu_power_w: 300.0,
                cpu_tctl_c: 40.0,
                cpu_package_power_w: 400.0,
            },
        ];
        let resolved = ResolvedTelemetry {
            source: TelemetrySource::Csv,
            source_label: "csv_re4".to_string(),
            csv_path: Some(PathBuf::from("/tmp/telemetry.csv")),
            rows: Some(rows),
        };

        // tick=0 uses row[0], tick=3 wraps back to row[1] (3 % 2 == 1).
        let snap0 = telemetry_snapshot_for_tick(0, &resolved);
        let snap3 = telemetry_snapshot_for_tick(3, &resolved);

        assert!((snap0.gpu_temp_c - 10.0).abs() < 1e-6);
        assert!((snap3.gpu_temp_c - 30.0).abs() < 1e-6);
        // timestamps are rewritten to (tick + 1) for 1-to-1 join with tick txt.
        assert_eq!(snap0.timestamp_ms, 1);
        assert_eq!(snap3.timestamp_ms, 4);
    }

    #[test]
    fn telemetry_snapshot_for_tick_uses_synthetic_on_fallback() {
        let resolved = ResolvedTelemetry {
            source: TelemetrySource::Synthetic,
            source_label: "synthetic_fallback".to_string(),
            csv_path: Some(PathBuf::from("/nonexistent/telemetry.csv")),
            rows: None,
        };
        let snap = telemetry_snapshot_for_tick(5, &resolved);
        // Synthetic path writes its own timestamp, then we overwrite to tick+1.
        assert_eq!(snap.timestamp_ms, 6);
        // And the synthetic sinusoid produces non-zero, finite values.
        assert!(snap.gpu_temp_c.is_finite() && snap.gpu_temp_c > 0.0);
    }

    #[test]
    fn resolve_telemetry_from_synthetic_ignores_path() {
        let resolved =
            resolve_telemetry_from(TelemetrySource::Synthetic, PathBuf::from("telemetry.csv"));
        assert_eq!(resolved.source, TelemetrySource::Synthetic);
        assert_eq!(resolved.source_label, "synthetic");
        assert!(resolved.csv_path.is_none());
        assert!(resolved.rows.is_none());
    }

    #[test]
    fn resolve_telemetry_from_csv_loads_valid_file() {
        let csv = "timestamp_ms,gpu_temp_c,gpu_power_w,cpu_tctl_c,cpu_package_power_w\n\
                   1000,60.5,250.0,70.0,120.0\n";
        let path = write_temp_csv("resolve_ok", csv);
        let resolved = resolve_telemetry_from(TelemetrySource::Csv, path.clone());
        assert_eq!(resolved.source, TelemetrySource::Csv);
        assert!(resolved.rows.as_ref().is_some_and(|r| r.len() == 1));
        assert_eq!(resolved.csv_path.as_ref(), Some(&path));
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn resolve_telemetry_from_csv_empty_falls_back() {
        let csv = "timestamp_ms,gpu_temp_c,gpu_power_w,cpu_tctl_c,cpu_package_power_w\n";
        let path = write_temp_csv("resolve_empty", csv);
        let resolved = resolve_telemetry_from(TelemetrySource::Csv, path.clone());
        assert_eq!(resolved.source, TelemetrySource::Synthetic);
        assert_eq!(resolved.source_label, "synthetic_fallback");
        assert_eq!(resolved.csv_path.as_ref(), Some(&path));
        assert!(resolved.rows.is_none());
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn resolve_telemetry_from_csv_missing_falls_back() {
        let path = test_scratch_dir().join(format!(
            "corinth_canal_missing_{}.csv",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let resolved = resolve_telemetry_from(TelemetrySource::Csv, path.clone());
        assert_eq!(resolved.source, TelemetrySource::Synthetic);
        assert_eq!(resolved.source_label, "synthetic_fallback");
        assert_eq!(resolved.csv_path.as_ref(), Some(&path));
        assert!(resolved.rows.is_none());
    }
}
