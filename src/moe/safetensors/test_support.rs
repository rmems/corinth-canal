// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Drop-cleaned temp directories and file writers for Safetensors test fixtures.

use std::fs::{self, File};
use std::io::Write;
use std::ops::Deref;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

pub(in crate::moe) struct TestDir(PathBuf);

impl Deref for TestDir {
    type Target = Path;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl AsRef<Path> for TestDir {
    fn as_ref(&self) -> &Path {
        &self.0
    }
}

impl Drop for TestDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

pub(in crate::moe) fn temp_dir(name: &str) -> TestDir {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let path = std::env::temp_dir().join(format!(
        "corinth-safetensors-{name}-{}-{nanos}",
        std::process::id()
    ));
    fs::create_dir_all(&path).unwrap();
    TestDir(path)
}

fn write_raw(path: &Path, header: &[u8], payload: &[u8]) {
    let mut file = File::create(path).unwrap();
    file.write_all(&(header.len() as u64).to_le_bytes())
        .unwrap();
    file.write_all(header).unwrap();
    file.write_all(payload).unwrap();
}

/// Write a `.safetensors` file with a raw JSON header and an all-zero payload.
pub(in crate::moe) fn write_safetensors(path: &Path, header: &str, data_len: usize) {
    write_raw(path, header.as_bytes(), &vec![0u8; data_len]);
}

/// Write a `.safetensors` file with the given JSON header and payload bytes.
pub(in crate::moe) fn write_safetensors_bytes(
    path: &Path,
    header: &serde_json::Value,
    payload: &[u8],
) {
    write_raw(path, &serde_json::to_vec(header).unwrap(), payload);
}

/// Write `config.json` into `dir`.
pub(in crate::moe) fn write_hf_config(dir: &Path, config: &serde_json::Value) {
    fs::write(dir.join("config.json"), serde_json::to_vec(config).unwrap()).unwrap();
}
