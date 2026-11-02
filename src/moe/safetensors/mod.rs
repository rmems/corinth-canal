// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Safetensors checkpoint inspection, deterministic manifest generation, and
//! tensor loading for the router bridge.
//!
//! Submodules:
//! - extractable (#116 → `rmems/engram-parser`, feature `safetensors`;
//!   one-way copy, no dependency in either direction): `discovery`, `json`,
//!   `paths`, `manifest`, `validate`
//! - corinth-specific, never extracted: `config` (HF config), `map`
//!   (mmap load / token extract)
//!
//! The `inspect_*` functions read only Safetensors headers and optional
//! Hugging Face shard index metadata.  The `MappedSafetensorsCheckpoint`
//! type memory-maps shards and extracts tensor payload bytes on demand.

use crate::error::HybridError;
use std::path::Path;

mod config;
mod discovery;
mod json;
mod manifest;
mod map;
mod paths;
mod validate;

pub use discovery::{
    SafetensorsCandidateSummary, SafetensorsExpertGroup, SafetensorsRouterCandidate,
};
pub use manifest::{
    SafetensorsCheckpointSource, SafetensorsManifest, SafetensorsTensorRecord,
    inspect_safetensors_checkpoint, write_safetensors_manifest,
};
pub use map::{MappedSafetensorsCheckpoint, SafetensorsMetadata, bf16_to_f32};
// Only `moe/tests.rs` needs this outside `validate`.
#[cfg(test)]
pub(crate) use validate::dtype_size_bytes;
pub(crate) use validate::is_packed_int4;

pub(super) fn relative_path(path: &Path, root: &Path) -> String {
    path.strip_prefix(root)
        .unwrap_or(path)
        .to_string_lossy()
        .replace('\\', "/")
}

pub(super) fn model_load(path: &Path, reason: String) -> HybridError {
    HybridError::ModelLoad {
        path: path.display().to_string(),
        reason,
    }
}

#[cfg(test)]
pub(in crate::moe) mod test_support;
#[cfg(test)]
mod tests;
