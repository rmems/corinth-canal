// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Shared CSV telemetry helpers for GPU routing outputs.

use super::core::GPU_ROUTING_TELEMETRY_HEADER;
use crate::gpu::{GpuError, GpuResult};
use std::fs::OpenOptions;
use std::io::{ErrorKind, Write};
use std::path::Path;

/// `mean_adaptation` is reserved in the routing CSV schema but not populated yet;
/// writers use `0.0` until GPU adaptation state is exported into this column.
pub(super) const GPU_ROUTING_TELEMETRY_MEAN_ADAPTATION_UNUSED: f32 = 0.0;

fn format_gpu_routing_row(
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

fn append_row_only(file: &mut std::fs::File, row: &str) -> GpuResult<()> {
    writeln!(file, "{row}").map_err(|e| GpuError::MemoryError(format!("CSV Write Failed: {e}")))
}

pub(super) fn append_gpu_routing_telemetry_row(
    path: &Path,
    token_idx: usize,
    best_score: i32,
    best_walker: i32,
    spike_count: usize,
    mean_adaptation: f32,
    active_fraction: f32,
) -> GpuResult<()> {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent).map_err(|e| {
            GpuError::MemoryError(format!("CSV Write Failed: could not create parent: {e}"))
        })?;
    }

    let row = format_gpu_routing_row(
        token_idx,
        best_score,
        best_walker,
        spike_count,
        mean_adaptation,
        active_fraction,
    );

    match OpenOptions::new().write(true).create_new(true).open(path) {
        Ok(mut file) => {
            writeln!(file, "{GPU_ROUTING_TELEMETRY_HEADER}")
                .map_err(|e| GpuError::MemoryError(format!("CSV Write Failed: {e}")))?;
            append_row_only(&mut file, &row)?;
            return Ok(());
        }
        Err(error) if error.kind() == ErrorKind::AlreadyExists => {}
        Err(error) => {
            return Err(GpuError::MemoryError(format!("CSV Write Failed: {error}")));
        }
    }

    let needs_header = std::fs::metadata(path)
        .map(|metadata| metadata.len() == 0)
        .unwrap_or(false);

    let mut file = OpenOptions::new()
        .append(true)
        .open(path)
        .map_err(|e| GpuError::MemoryError(format!("CSV Write Failed: {e}")))?;

    if needs_header {
        writeln!(file, "{GPU_ROUTING_TELEMETRY_HEADER}")
            .map_err(|e| GpuError::MemoryError(format!("CSV Write Failed: {e}")))?;
    }

    append_row_only(&mut file, &row)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Read;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Barrier};
    use std::thread;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_csv_path(label: &str) -> std::path::PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("corinth_canal_gpu_routing_{label}_{nanos}.csv"))
    }

    #[test]
    fn append_adds_header_once_sequential() {
        let path = temp_csv_path("sequential");
        append_gpu_routing_telemetry_row(&path, 0, 11, 3, 42, 0.12, 0.021).unwrap();
        append_gpu_routing_telemetry_row(&path, 1, 7, 2, 18, 0.45, 0.009).unwrap();

        let contents = std::fs::read_to_string(&path).unwrap();
        let mut lines = contents.lines();
        assert_eq!(lines.next().unwrap(), GPU_ROUTING_TELEMETRY_HEADER);
        assert_eq!(lines.next().unwrap(), "0,11,3,42,0.1200,0.0210");
        assert_eq!(lines.next().unwrap(), "1,7,2,18,0.4500,0.0090");
        assert!(lines.next().is_none());

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn append_propagates_write_failure() {
        let path = temp_csv_path("bad");
        std::fs::create_dir_all(&path).unwrap();
        let err = append_gpu_routing_telemetry_row(&path, 0, 0, 0, 0, 0.0, 0.0).unwrap_err();
        assert!(
            err.to_string().contains("CSV Write Failed"),
            "unexpected error: {err}"
        );
        let _ = std::fs::remove_dir_all(&path);
    }

    #[test]
    fn concurrent_appends_emit_single_header() {
        let path = Arc::new(temp_csv_path("concurrent"));
        let barrier = Arc::new(Barrier::new(9));
        let errors = Arc::new(AtomicUsize::new(0));
        let mut handles = Vec::with_capacity(8);

        for thread_idx in 0..8 {
            let path = Arc::clone(&path);
            let barrier = Arc::clone(&barrier);
            let errors = Arc::clone(&errors);
            handles.push(thread::spawn(move || {
                barrier.wait();
                for row in 0..25 {
                    if append_gpu_routing_telemetry_row(
                        path.as_ref(),
                        thread_idx * 100 + row,
                        row as i32,
                        thread_idx as i32,
                        row,
                        0.0,
                        0.1,
                    )
                    .is_err()
                    {
                        errors.fetch_add(1, Ordering::Relaxed);
                    }
                }
            }));
        }

        barrier.wait();
        for handle in handles {
            handle.join().expect("writer thread panicked");
        }

        assert_eq!(
            errors.load(Ordering::Relaxed),
            0,
            "expected all concurrent CSV appends to succeed"
        );

        let mut file = std::fs::File::open(path.as_ref()).unwrap();
        let mut contents = String::new();
        file.read_to_string(&mut contents).unwrap();
        let line_count = contents.lines().count();
        assert_eq!(
            line_count,
            1 + 8 * 25,
            "expected one header and 200 data rows, got {line_count} lines"
        );
        let header_count = contents
            .lines()
            .filter(|line| *line == GPU_ROUTING_TELEMETRY_HEADER)
            .count();
        assert_eq!(header_count, 1, "expected exactly one CSV header");

        let _ = std::fs::remove_file(path.as_ref());
    }
}
