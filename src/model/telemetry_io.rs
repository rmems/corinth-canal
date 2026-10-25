// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Shared CSV telemetry helpers for GPU routing outputs.
//!
//! The header decision, file I/O and row format live in the CPU-reachable
//! [`crate::experiment::routing_csv`] module so hosted CI covers them; this
//! file only adapts errors to [`GpuError`] (RM-1450).

use crate::experiment::routing_csv::{
    GPU_ROUTING_TELEMETRY_HEADER, append_row_with_header, format_row,
};
use crate::gpu::{GpuError, GpuResult};
use std::path::Path;

/// Append one routing telemetry row to `path`, writing the header exactly
/// once per file (exclusive create, no exists-then-open window). Errors are
/// returned, never swallowed.
pub(super) fn append_gpu_routing_telemetry_row(
    path: &Path,
    token_idx: usize,
    best_score: i32,
    best_walker: i32,
    spike_count: usize,
    mean_adaptation: f32,
    active_fraction: f32,
) -> GpuResult<()> {
    let row = format_row(
        token_idx,
        best_score,
        best_walker,
        spike_count,
        mean_adaptation,
        active_fraction,
    );
    append_row_with_header(path, GPU_ROUTING_TELEMETRY_HEADER, &row).map_err(|e| {
        GpuError::MemoryError(format!(
            "routing telemetry CSV write failed at '{}': {e}",
            path.display()
        ))
    })
}
