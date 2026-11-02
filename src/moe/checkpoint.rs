// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Compatibility re-export; GGUF implementation lives in [`super::gguf`].
//!
//! Callers under `moe` continue to import via `super::checkpoint::{...}`.
//! Implementation is split across private modules:
//! - `gguf/metadata.rs` — header/metadata parse, cursor
//! - `gguf/map.rs` — mmap checkpoint, tensor access, embeddings
//! - `gguf/dequant.rs` — row dequantization and f16 helpers
//! - `gguf/cuda_register.rs` — CUDA host registration (feature-gated)

pub(super) use super::gguf::{
    GgufMetadata, GgufTensorInfo, MappedGgufCheckpoint, f16_to_f32, probe_and_map_checkpoint,
};
