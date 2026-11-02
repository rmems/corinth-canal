// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Error types for `corinth-canal`.
//!
//! Most fallible library APIs return [`HybridError`] wrapped in a [`Result`].
//! Two exceptions remain: the `ProjectionMode` env parsers in `types.rs` return
//! `Result<_, String>`, and the cuda-gated `gpu` and `model` temporal APIs return
//! `GpuResult`, which has no `From` bridge into [`HybridError`] yet. Downstream
//! callers can match on variants to distinguish configuration mistakes from I/O
//! failures or runtime model errors.

use thiserror::Error;

/// Unified error type for the hybrid framework.
#[derive(Debug, Error)]
pub enum HybridError {
    // ── Configuration errors ──────────────────────────────────────────────
    /// Configuration or schema validation failed
    /// ([`ModelConfig`](crate::types::ModelConfig), `RunMatrix`, adapter or router
    /// settings), or a config-like artifact could not be serialized.
    #[error("invalid configuration: {0}")]
    InvalidConfig(String),

    // ── Model loading errors ──────────────────────────────────────────────
    /// A checkpoint (GGUF or Safetensors) could not be opened, parsed or described.
    #[error("model load failed for '{path}': {reason}")]
    ModelLoad { path: String, reason: String },

    /// The model file format is not supported (e.g. wrong GGUF magic).
    #[error("unsupported model format: {0}")]
    UnsupportedFormat(String),

    /// A required tensor or layer was missing from the checkpoint.
    #[error("missing tensor '{name}' in model '{path}'")]
    MissingTensor { name: String, path: String },

    // ── Forward-pass errors ───────────────────────────────────────────────
    /// Input slice had the wrong length.
    #[error("input length mismatch: expected {expected}, got {got}")]
    InputLengthMismatch { expected: usize, got: usize },

    /// Routing output lacked data the caller required (for example
    /// `expert_weights` missing from `ModelOutput`).
    #[error("Router forward pass failed: {0}")]
    RouterForward(String),

    // ── I/O errors ────────────────────────────────────────────────────────
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}

/// Convenience alias used throughout the crate.
pub type Result<T> = std::result::Result<T, HybridError>;
