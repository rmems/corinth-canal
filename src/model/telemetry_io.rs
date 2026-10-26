// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Shared CSV telemetry helpers for GPU routing outputs.

use super::core::GPU_ROUTING_TELEMETRY_HEADER;
use crate::gpu::{GpuError, GpuResult};
use std::fs::OpenOptions;
use std::io::{ErrorKind, Write};
use std::path::Path;

/// Append one routing row, writing the header exactly once.
///
/// The header is decided atomically: the first writer wins `create_new`
/// (exclusive create) and writes header + row in a single `write_all`;
/// every other writer sees `AlreadyExists` and appends only the row. A
/// pre-existing zero-length file (e.g. externally touched) gets header + row
/// in one write as well.
pub(super) fn append_gpu_routing_telemetry_row(
    path: &Path,
    token_idx: usize,
    best_score: i32,
    best_walker: i32,
    spike_count: usize,
    mean_adaptation: f32,
    active_fraction: f32,
) -> GpuResult<()> {
    let werr = |e: std::io::Error| GpuError::MemoryError(format!("CSV Write Failed: {e}"));
    let row = format!(
        "{token_idx},{best_score},{best_walker},{spike_count},{mean_adaptation:.4},{active_fraction:.4}
"
    );
    let with_header = format!(
        "{GPU_ROUTING_TELEMETRY_HEADER}
{row}"
    );

    match OpenOptions::new().write(true).create_new(true).open(path) {
        Ok(mut file) => return file.write_all(with_header.as_bytes()).map_err(werr),
        Err(e) if e.kind() == ErrorKind::AlreadyExists => {}
        Err(e) => return Err(werr(e)),
    }

    let mut file = OpenOptions::new().append(true).open(path).map_err(werr)?;
    let empty = file.metadata().map(|m| m.len() == 0).map_err(werr)?;
    let payload = if empty { &with_header } else { &row };
    file.write_all(payload.as_bytes()).map_err(werr)
}
