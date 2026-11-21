// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Integration harness for `examples/summarize_local_saaq.rs`.
//!
//! Example targets do not run unit-test harnesses, so this `#[path]` include
//! is what makes the CLI exit-status contract tests execute under
//! `cargo test --no-default-features` (self-hosted / local all-targets;
//! hosted PR CI is `--lib` only and does not run this file).

#[allow(dead_code)] // `fn main` and report-only helpers ride along with the include.
#[path = "../examples/summarize_local_saaq.rs"]
mod summarize_local_saaq;

use std::fs;
use std::path::{Path, PathBuf};

fn temp_runs_dir(tag: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("target")
        .join("summarize-local-saaq-tests")
        .join(format!("{tag}-{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

#[test]
fn unreadable_manifest_exits_2() {
    let dir = temp_runs_dir("bad-manifest");
    let run_dir = dir.join("run-1");
    fs::create_dir_all(&run_dir).unwrap();
    fs::write(run_dir.join("run_manifest.json"), b"{not json").unwrap();
    assert_eq!(summarize_local_saaq::run(&dir), 2);
    fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn clean_runs_dir_exits_0() {
    let dir = temp_runs_dir("clean");
    fs::create_dir_all(dir.join("run-1")).unwrap();
    // No manifests anywhere: an empty report, not an error.
    assert_eq!(summarize_local_saaq::run(&dir), 0);
    fs::remove_dir_all(&dir).unwrap();
}
