// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Canonical five-column replay CSV writer.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

/// Must match [`super::super::telemetry_csv::TELEMETRY_CSV_HEADER`] when both
/// modules are loaded from `examples/support/mod.rs`. Duplicated so this
/// tree stays `#[path]`-includable as a standalone test module.
pub const SPIKENAUT_CSV_HEADER: &str =
    "timestamp_ms,gpu_temp_c,gpu_power_w,cpu_tctl_c,cpu_package_power_w";

/// Write snapshots using the canonical five-column replay header.
pub fn write_canonical_csv(
    path: &Path,
    rows: &[corinth_canal::TelemetrySnapshot],
) -> Result<(), Box<dyn std::error::Error>> {
    ensure_parent_dir(path)?;
    let mut writer = BufWriter::new(File::create(path)?);
    writeln!(writer, "{SPIKENAUT_CSV_HEADER}")?;
    write_csv_rows(&mut writer, rows)?;
    writer.flush()?;
    Ok(())
}

fn ensure_parent_dir(path: &Path) -> std::io::Result<()> {
    match path.parent() {
        Some(parent) if !parent.as_os_str().is_empty() => std::fs::create_dir_all(parent),
        _ => Ok(()),
    }
}

fn write_csv_rows<W: Write>(
    writer: &mut W,
    rows: &[corinth_canal::TelemetrySnapshot],
) -> std::io::Result<()> {
    for row in rows {
        write_csv_row(writer, row)?;
    }
    Ok(())
}

fn write_csv_row<W: Write>(
    writer: &mut W,
    row: &corinth_canal::TelemetrySnapshot,
) -> std::io::Result<()> {
    writeln!(
        writer,
        "{},{:.6},{:.6},{:.6},{:.6}",
        row.timestamp_ms, row.gpu_temp_c, row.gpu_power_w, row.cpu_tctl_c, row.cpu_package_power_w
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn csv_header_matches_canonical_replay_contract() {
        assert_eq!(
            SPIKENAUT_CSV_HEADER,
            "timestamp_ms,gpu_temp_c,gpu_power_w,cpu_tctl_c,cpu_package_power_w"
        );
    }
}
