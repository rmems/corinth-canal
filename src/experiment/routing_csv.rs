// SPDX-License-Identifier: Apache-2.0 OR MIT
//! CPU-testable writer for the GPU routing telemetry CSV
//! (`snn_gpu_routing_telemetry.csv`).
//!
//! The GPU side (`src/model/telemetry_io.rs`, `cuda`-gated) only formats a
//! row and calls [`append_row_with_header`]; the header decision and the
//! file I/O live here so hosted CPU CI covers them (RM-1450).
//!
//! Header atomicity: the old writer decided "needs header" from
//! `!path.exists() || len == 0` and *then* opened in append mode, so two
//! writers racing on a fresh file both wrote the header. This writer first
//! tries an exclusive `create_new` open; the winner writes header and row in
//! a single `write_all`, every loser sees `AlreadyExists` and appends only
//! the row. An existing zero-length file (left by a crash between create
//! and the first write) is handled by re-checking the length on the already
//! open handle and writing the header through that same handle, so no second
//! `exists()`/`open()` window remains. In-process concurrency is additionally
//! serialised through a per-path lock so two threads of one process never
//! both see the empty-file case. Cross-process exclusion on that
//! empty-file recovery path is out of scope: one runner process owns a
//! run directory's CSV, and anything stronger would need `flock`.
//!
//! This module reads no environment variables and resolves no paths.

use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};

/// Column header of the routing telemetry CSV. Do not change (schema rule).
pub const GPU_ROUTING_TELEMETRY_HEADER: &str =
    "token_idx,best_score,best_walker,spike_count,mean_adaptation,active_fraction";

/// Per-process, per-path serialisation of appends so two threads writing the
/// same CSV cannot both observe the zero-length state. The map holds one
/// `Arc<Mutex<()>>` per distinct path for the process lifetime (no leak:
/// the `Arc` is cloned out under the map lock).
fn path_lock(path: &Path) -> Arc<Mutex<()>> {
    static LOCKS: OnceLock<Mutex<HashMap<PathBuf, Arc<Mutex<()>>>>> = OnceLock::new();
    let locks = LOCKS.get_or_init(|| Mutex::new(HashMap::new()));
    let mut guard = locks
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    Arc::clone(guard.entry(path.to_path_buf()).or_default())
}

/// Append `row` (without trailing newline) to the CSV at `path`, writing
/// `header` first exactly once per file.
///
/// Errors are returned, never swallowed; the caller decides whether a
/// telemetry write failure is fatal for its run.
pub fn append_row_with_header(path: &Path, header: &str, row: &str) -> io::Result<()> {
    let lock = path_lock(path);
    let _serialised = lock.lock().unwrap_or_else(|poisoned| poisoned.into_inner());

    match OpenOptions::new().write(true).create_new(true).open(path) {
        Ok(mut file) => {
            // We created the file: header and first row go out together.
            file.write_all(format!("{header}\n{row}\n").as_bytes())
        }
        Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {
            let mut file = OpenOptions::new().append(true).open(path)?;
            append_to_existing(&mut file, header, row)
        }
        Err(error) => Err(error),
    }
}

/// Append to an already-open handle, re-checking emptiness on *that* handle
/// (no path re-stat) so a crash-stranded empty file still gets its header.
fn append_to_existing(file: &mut File, header: &str, row: &str) -> io::Result<()> {
    if file.metadata()?.len() == 0 {
        file.write_all(format!("{header}\n{row}\n").as_bytes())
    } else {
        file.write_all(format!("{row}\n").as_bytes())
    }
}

/// Format one routing telemetry row. Kept separate from the I/O so the GPU
/// side can stay a one-liner and the format is unit-tested on the CPU.
pub fn format_row(
    token_idx: usize,
    best_score: i32,
    best_walker: i32,
    spike_count: usize,
    mean_adaptation: f32,
    active_fraction: f32,
) -> String {
    format!(
        "{token_idx},{best_score},{best_walker},{spike_count},{mean_adaptation:.4},{active_fraction:.4}"
    )
}

/// Arithmetic mean of a per-neuron adaptation readback; `0.0` for an empty
/// slice. Replaces the hardcoded `0.25` the GPU path used to stamp.
pub fn mean_adaptation(adaptation: &[f32]) -> f32 {
    if adaptation.is_empty() {
        return 0.0;
    }
    let sum: f64 = adaptation.iter().map(|&v| v as f64).sum();
    (sum / adaptation.len() as f64) as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    static NONCE: AtomicU64 = AtomicU64::new(0);

    fn scratch(tag: &str) -> PathBuf {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("target")
            .join("routing-csv-tests");
        std::fs::create_dir_all(&dir).unwrap();
        dir.join(format!(
            "{tag}-{}-{}.csv",
            std::process::id(),
            NONCE.fetch_add(1, Ordering::Relaxed)
        ))
    }

    fn lines(path: &Path) -> Vec<String> {
        std::fs::read_to_string(path)
            .unwrap()
            .lines()
            .map(str::to_owned)
            .collect()
    }

    #[test]
    fn header_is_written_once_and_rows_match_legacy_format() {
        let path = scratch("once");
        append_row_with_header(
            &path,
            GPU_ROUTING_TELEMETRY_HEADER,
            &format_row(0, 11, 3, 42, 0.12, 0.021),
        )
        .unwrap();
        append_row_with_header(
            &path,
            GPU_ROUTING_TELEMETRY_HEADER,
            &format_row(1, 7, 2, 18, 0.45, 0.009),
        )
        .unwrap();
        assert_eq!(
            lines(&path),
            vec![
                GPU_ROUTING_TELEMETRY_HEADER.to_owned(),
                "0,11,3,42,0.1200,0.0210".to_owned(),
                "1,7,2,18,0.4500,0.0090".to_owned(),
            ]
        );
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn crash_stranded_empty_file_still_gets_exactly_one_header() {
        let path = scratch("empty");
        std::fs::write(&path, b"").unwrap();
        append_row_with_header(&path, "h", "r0").unwrap();
        append_row_with_header(&path, "h", "r1").unwrap();
        assert_eq!(lines(&path), vec!["h", "r0", "r1"]);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn concurrent_appends_produce_exactly_one_header() {
        let path = scratch("threads");
        let writers = 8;
        let rows_per_writer = 25;
        std::thread::scope(|scope| {
            for w in 0..writers {
                let path = &path;
                scope.spawn(move || {
                    for i in 0..rows_per_writer {
                        append_row_with_header(path, "h", &format!("{w},{i}")).unwrap();
                    }
                });
            }
        });
        let all = lines(&path);
        assert_eq!(all.len(), 1 + writers * rows_per_writer);
        assert_eq!(all[0], "h");
        assert_eq!(all.iter().filter(|l| *l == "h").count(), 1);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn write_errors_propagate() {
        let dir = scratch("dir");
        std::fs::create_dir_all(&dir).unwrap();
        let error = append_row_with_header(&dir, "h", "r").unwrap_err();
        assert_ne!(error.kind(), io::ErrorKind::NotFound);
        let _ = std::fs::remove_dir_all(dir);

        let missing_parent = scratch("x").join("no-such-dir").join("t.csv");
        assert!(append_row_with_header(&missing_parent, "h", "r").is_err());
    }

    #[test]
    fn mean_adaptation_is_the_arithmetic_mean() {
        assert_eq!(mean_adaptation(&[]), 0.0);
        assert!((mean_adaptation(&[0.25; 32]) - 0.25).abs() < 1e-6);
        assert!((mean_adaptation(&[0.0, 1.0]) - 0.5).abs() < 1e-6);
    }
}
