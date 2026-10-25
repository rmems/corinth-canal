// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Durable JSON artifact I/O shared by the SAAQ runner and its readers.
//!
//! `run_manifest.json` and `summary.json` are rewritten several times per run
//! (preflight, terminal status, strict-repeat stamping). A plain
//! `fs::write` truncates in place, so a crash between the truncate and the
//! final flush leaves a zero-length or half-written file that downstream
//! readers then skip or misreport. [`write_json_atomic`] replaces that with
//! temp-file + fsync + rename (+ directory fsync on Unix), so a reader only
//! ever observes the previous complete file or the new complete file.
//!
//! This module reads no environment variables and resolves no paths; callers
//! pass the destination explicitly. (RM-1447)

use serde::Serialize;
use serde::de::DeserializeOwned;
use std::fs::{self, File};
use std::io::{self, Write};
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Outcome of a successful atomic write.
///
/// Once `rename` has succeeded the artifact is published and readable; a
/// failure to fsync the parent directory afterwards only weakens durability
/// across a power loss, so it is reported here rather than as an `Err` that
/// would wrongly tell the caller the write did not happen.
#[derive(Debug, Default, PartialEq, Eq)]
#[must_use]
pub struct AtomicWriteOutcome {
    /// `Some(message)` when the parent-directory fsync failed after the
    /// rename succeeded. Callers should log it as a warning.
    pub dir_sync_warning: Option<String>,
}

/// Serialize `value` as pretty JSON and durably replace `path` with it.
///
/// Serialization happens before any file is touched, so a serialization
/// error leaves an existing artifact intact. The payload goes to a uniquely
/// named sibling temp file in the same directory (so `rename` never crosses
/// a filesystem), created with `create_new` so it can never clobber, is
/// `sync_all`ed, then renamed over `path`; on Unix the parent directory is
/// fsynced as well so the rename itself is durable. If any step before the
/// rename fails the temp file is removed and the previous `path` contents
/// (if any) are untouched. A directory-fsync failure after the rename is
/// returned in [`AtomicWriteOutcome::dir_sync_warning`], not as `Err`.
pub fn write_json_atomic<T: Serialize>(path: &Path, value: &T) -> io::Result<AtomicWriteOutcome> {
    let payload = serde_json::to_string_pretty(value).map_err(io::Error::other)?;
    write_bytes_atomic(path, payload.as_bytes())
}

/// Durably replace `path` with `bytes` (see [`write_json_atomic`]).
pub fn write_bytes_atomic(path: &Path, bytes: &[u8]) -> io::Result<AtomicWriteOutcome> {
    write_bytes_atomic_with(path, bytes, sync_dir)
}

/// [`write_bytes_atomic`] with the directory-fsync step injectable, so the
/// "rename succeeded, directory fsync failed" branch is testable without
/// faulting the filesystem.
fn write_bytes_atomic_with(
    path: &Path,
    bytes: &[u8],
    sync_parent: fn(&Path) -> io::Result<()>,
) -> io::Result<AtomicWriteOutcome> {
    let parent = parent_dir(path);
    let temp_path = sibling_temp_path(path)?;

    // Stage 1: everything before the rename. Failure here means nothing was
    // published, so the stranded temp file is removed.
    if let Err(error) = stage_temp_file(&temp_path, bytes) {
        let _ = fs::remove_file(&temp_path);
        return Err(error);
    }
    if let Err(error) = fs::rename(&temp_path, path) {
        let _ = fs::remove_file(&temp_path);
        return Err(error);
    }

    // Stage 2: the artifact is published. Do not touch `temp_path` again
    // (it no longer exists) and do not report the write as failed.
    Ok(AtomicWriteOutcome {
        dir_sync_warning: sync_parent(parent).err().map(|error| {
            format!(
                "{}: parent directory fsync failed after rename; artifact is written but its directory entry may not survive a power loss: {error}",
                path.display()
            )
        }),
    })
}

fn parent_dir(path: &Path) -> &Path {
    match path.parent() {
        Some(parent) if !parent.as_os_str().is_empty() => parent,
        _ => Path::new("."),
    }
}

/// `<dir>/.<name>.tmp-<pid>-<counter>`: same directory as the destination so
/// the final `rename` is atomic and never fails with `EXDEV`.
fn sibling_temp_path(path: &Path) -> io::Result<std::path::PathBuf> {
    let file_name = path
        .file_name()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "path has no file name"))?
        .to_string_lossy()
        .into_owned();
    let nonce = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
    Ok(parent_dir(path).join(format!(".{file_name}.tmp-{}-{nonce}", std::process::id())))
}

/// Create (exclusively), fill and fsync the staging file.
fn stage_temp_file(temp_path: &Path, bytes: &[u8]) -> io::Result<()> {
    let mut file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(temp_path)?;
    file.write_all(bytes)?;
    file.sync_all()
}

#[cfg(unix)]
fn sync_dir(dir: &Path) -> io::Result<()> {
    File::open(dir)?.sync_all()
}

#[cfg(not(unix))]
fn sync_dir(_dir: &Path) -> io::Result<()> {
    Ok(())
}

/// Read and deserialize a JSON artifact.
///
/// A missing file surfaces as `ErrorKind::NotFound`; an unreadable or
/// unparsable file (including a truncated one) as `ErrorKind::InvalidData`
/// with the parse error attached, so callers can distinguish "never written"
/// from "written badly" instead of collapsing both into a silent skip.
pub fn read_json<T: DeserializeOwned>(path: &Path) -> io::Result<T> {
    let raw = fs::read_to_string(path)?;
    serde_json::from_str(&raw).map_err(|error| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("{}: {error}", path.display()),
        )
    })
}

/// Civil (Y, M, D, h, m, s) in UTC from a Unix timestamp in seconds.
/// Based on Howard Hinnant's public-domain `civil_from_days`.
pub fn civil_from_unix_secs(secs: i64) -> (i32, u32, u32, u32, u32, u32) {
    let days = secs.div_euclid(86_400);
    let rem = secs.rem_euclid(86_400) as u32;
    let hour = rem / 3_600;
    let minute = (rem % 3_600) / 60;
    let second = rem % 60;

    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097) as u64;
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let y = (yoe as i64) + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = if m <= 2 { y + 1 } else { y };
    (year as i32, m as u32, d as u32, hour, minute, second)
}

/// Format a Unix timestamp (seconds) as RFC 3339 UTC, e.g.
/// `2026-10-25T14:03:09Z`.
pub fn format_rfc3339_utc(secs: i64) -> String {
    let (year, month, day, hour, minute, second) = civil_from_unix_secs(secs);
    format!("{year:04}-{month:02}-{day:02}T{hour:02}:{minute:02}:{second:02}Z")
}

/// Current wall-clock time as RFC 3339 UTC (second precision).
pub fn now_rfc3339_utc() -> String {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    format_rfc3339_utc(secs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::{Deserialize, Serialize};
    use std::collections::BTreeMap;

    #[derive(Debug, PartialEq, Serialize, Deserialize)]
    struct Doc {
        run_id: String,
        status: String,
    }

    /// Per-test scratch directory under the build's target dir (never the
    /// shared system temp dir).
    fn temp_dir(tag: &str) -> std::path::PathBuf {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("target")
            .join("artifact-io-tests")
            .join(format!(
                "{tag}-{}-{}",
                std::process::id(),
                TEMP_COUNTER.fetch_add(1, Ordering::Relaxed)
            ));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn write_then_read_round_trips_and_leaves_no_temp_file() {
        let dir = temp_dir("roundtrip");
        let path = dir.join("run_manifest.json");
        let doc = Doc {
            run_id: "r0".into(),
            status: "completed".into(),
        };
        let outcome = write_json_atomic(&path, &doc).unwrap();
        assert_eq!(outcome, AtomicWriteOutcome::default());
        let back: Doc = read_json(&path).unwrap();
        assert_eq!(back, doc);
        let leftovers: Vec<_> = fs::read_dir(&dir)
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        assert_eq!(leftovers, vec!["run_manifest.json".to_owned()]);
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn crash_mid_write_never_exposes_truncated_json_to_readers() {
        // A crash between temp-file creation and rename leaves only the
        // sibling temp file behind; the published path still holds the
        // previous complete document.
        let dir = temp_dir("crash");
        let path = dir.join("summary.json");
        let first = Doc {
            run_id: "r0".into(),
            status: "preflight".into(),
        };
        let _ = write_json_atomic(&path, &first).unwrap();

        // Simulate the half-written temp file a crash would strand.
        let stranded = dir.join(".summary.json.tmp-crash");
        fs::write(&stranded, b"{\"run_id\": \"r0\", \"stat").unwrap();

        let back: Doc = read_json(&path).unwrap();
        assert_eq!(back, first, "reader must see the previous complete file");

        // The next successful write replaces it wholesale.
        let second = Doc {
            run_id: "r0".into(),
            status: "completed".into(),
        };
        let _ = write_json_atomic(&path, &second).unwrap();
        let back: Doc = read_json(&path).unwrap();
        assert_eq!(back, second);
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn read_json_distinguishes_missing_from_truncated() {
        let dir = temp_dir("kinds");
        let missing = read_json::<Doc>(&dir.join("absent.json")).unwrap_err();
        assert_eq!(missing.kind(), io::ErrorKind::NotFound);

        let truncated = dir.join("truncated.json");
        fs::write(&truncated, b"{\"run_id\": \"r0\"").unwrap();
        let error = read_json::<Doc>(&truncated).unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::InvalidData);
        assert!(error.to_string().contains("truncated.json"));

        let empty = dir.join("empty.json");
        fs::write(&empty, b"").unwrap();
        let error = read_json::<Doc>(&empty).unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::InvalidData);
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn serialization_failure_leaves_existing_artifact_untouched() {
        // serde_json refuses non-string map keys; the existing file must
        // survive because serialization runs before any file is opened.
        let dir = temp_dir("serfail");
        let path = dir.join("summary.json");
        let good = Doc {
            run_id: "r0".into(),
            status: "completed".into(),
        };
        let _ = write_json_atomic(&path, &good).unwrap();
        let mut bad: BTreeMap<(u8, u8), u8> = BTreeMap::new();
        bad.insert((1, 2), 3);
        assert!(write_json_atomic(&path, &bad).is_err());
        let back: Doc = read_json(&path).unwrap();
        assert_eq!(back, good);
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn dir_sync_failure_after_rename_is_a_warning_not_an_error() {
        fn failing_sync(_dir: &Path) -> io::Result<()> {
            Err(io::Error::other("injected fsync failure"))
        }
        let dir = temp_dir("dirsync");
        let path = dir.join("summary.json");
        let outcome = write_bytes_atomic_with(&path, b"{\"ok\":true}", failing_sync).unwrap();
        let warning = outcome.dir_sync_warning.expect("warning surfaced");
        assert!(warning.contains("injected fsync failure"));
        assert!(warning.contains("summary.json"));
        // The artifact is published and complete...
        assert_eq!(fs::read(&path).unwrap(), b"{\"ok\":true}");
        // ...and no temp file was left behind or wrongly "cleaned up".
        let names: Vec<_> = fs::read_dir(&dir)
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        assert_eq!(names, vec!["summary.json".to_owned()]);
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn temp_file_is_a_sibling_of_the_destination() {
        let path = Path::new("artifacts/run/run_manifest.json");
        let temp = sibling_temp_path(path).unwrap();
        assert_eq!(temp.parent(), path.parent());
        let name = temp.file_name().unwrap().to_string_lossy().into_owned();
        assert!(name.starts_with(".run_manifest.json.tmp-"));
        assert!(sibling_temp_path(Path::new("/")).is_err());
    }

    #[test]
    fn pre_rename_failure_cleans_up_and_keeps_previous_artifact() {
        let dir = temp_dir("prerename");
        let path = dir.join("summary.json");
        let _ = write_bytes_atomic(&path, b"old").unwrap();
        // Make the destination a directory so `rename` fails after the temp
        // file was staged; the stranded temp file must be removed.
        let blocked = dir.join("blocked");
        fs::create_dir_all(blocked.join("child")).unwrap();
        assert!(write_bytes_atomic(&blocked, b"new").is_err());
        let names: Vec<_> = fs::read_dir(&dir)
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .filter(|n| n.starts_with('.'))
            .collect();
        assert!(names.is_empty(), "stranded temp files: {names:?}");
        assert_eq!(fs::read(&path).unwrap(), b"old");
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn rfc3339_utc_formats_known_instants() {
        assert_eq!(format_rfc3339_utc(0), "1970-01-01T00:00:00Z");
        assert_eq!(format_rfc3339_utc(951_782_400), "2000-02-29T00:00:00Z");
        assert_eq!(format_rfc3339_utc(1_761_400_989), "2025-10-25T14:03:09Z");
        let now = now_rfc3339_utc();
        assert_eq!(now.len(), 20);
        assert!(now.ends_with('Z'));
        assert_eq!(&now[4..5], "-");
        assert_eq!(&now[10..11], "T");
    }
}
