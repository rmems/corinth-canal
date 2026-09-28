// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Canonical five-column replay CSV writer.

use std::ffi::OsString;
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

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
    write_csv_atomically(path, |writer| {
        writeln!(writer, "{SPIKENAUT_CSV_HEADER}")?;
        write_csv_rows(writer, rows)
    })?;
    Ok(())
}

fn write_csv_atomically(
    path: &Path,
    write: impl FnOnce(&mut BufWriter<File>) -> std::io::Result<()>,
) -> std::io::Result<()> {
    let (temp_path, file) = create_temp_csv(path)?;
    let result = (|| {
        let mut writer = BufWriter::new(file);
        write(&mut writer)?;
        writer.flush()?;
        writer.get_ref().sync_all()?;
        drop(writer);
        std::fs::rename(&temp_path, path)
    })();
    if result.is_err() {
        let _ = std::fs::remove_file(&temp_path);
    }
    result
}

fn create_temp_csv(path: &Path) -> std::io::Result<(PathBuf, File)> {
    static NEXT_TEMP: AtomicU64 = AtomicU64::new(0);
    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let file_name = path
        .file_name()
        .ok_or_else(|| std::io::Error::other("CSV output has no filename"))?;
    for _ in 0..10 {
        let mut name = OsString::from(".");
        name.push(file_name);
        name.push(format!(
            ".{}.{}.tmp",
            std::process::id(),
            NEXT_TEMP.fetch_add(1, Ordering::Relaxed)
        ));
        let temp_path = parent.join(name);
        match OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temp_path)
        {
            Ok(file) => return Ok((temp_path, file)),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(error),
        }
    }
    Err(std::io::Error::other(
        "could not create a unique temporary CSV",
    ))
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
    fn failed_replacement_preserves_existing_csv() {
        let root = Path::new("target/tmp-tests").join(format!(
            "spikenaut_atomic_csv_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&root).unwrap();
        let path = root.join("replay.csv");
        std::fs::write(&path, b"last valid replay").unwrap();
        let result = write_csv_atomically(&path, |writer| {
            writer.write_all(b"partial")?;
            Err(std::io::Error::other("injected write failure"))
        });
        assert!(result.is_err());
        assert_eq!(std::fs::read(&path).unwrap(), b"last valid replay");
        assert_eq!(std::fs::read_dir(&root).unwrap().count(), 1);
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn csv_header_matches_canonical_replay_contract() {
        assert_eq!(
            SPIKENAUT_CSV_HEADER,
            "timestamp_ms,gpu_temp_c,gpu_power_w,cpu_tctl_c,cpu_package_power_w"
        );
    }
}
