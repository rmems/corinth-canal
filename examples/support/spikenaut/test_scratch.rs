// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Unique crate-local scratch paths for the Spikenaut unit tests.
//!
//! Crate-local rather than `std::env::temp_dir()` (Semgrep: multi-user temp
//! dir). `CARGO_TARGET_TMPDIR` is compile-time only, so it is not readable at
//! runtime.

use std::path::PathBuf;

pub(super) fn scratch_path(label: &str) -> PathBuf {
    let base = PathBuf::from("target").join("tmp-tests");
    std::fs::create_dir_all(&base).unwrap();
    base.join(format!(
        "{label}_{}_{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ))
}
