//! Error types for `corinth-canal`.
//!
//! All public functions in this crate return [`HybridError`] wrapped in a
//! [`Result`].  Downstream callers can match on variants to distinguish
//! configuration mistakes from I/O failures or runtime model errors.

use thiserror::Error;

/// Unified error type for the hybrid framework.
#[derive(Debug, Error)]
pub enum HybridError {
    // ── Configuration errors ──────────────────────────────────────────────
    /// A required field in [`HybridConfig`](crate::types::HybridConfig) was
    /// empty or out of range.
    #[error("invalid configuration: {0}")]
    InvalidConfig(String),

    // ── Model loading errors ──────────────────────────────────────────────
    /// The GGUF file could not be opened / parsed.
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

    /// The synthetic spike front-end produced no spikes.
    #[error("SNN produced no spikes after {steps} steps — network may be silent")]
    SilentNetwork { steps: usize },

    /// OLMoE forward pass returned an error.
    #[error("OLMoE forward pass failed: {0}")]
    OlmoeForward(String),

    // ── I/O errors ────────────────────────────────────────────────────────
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}

/// Convenience alias used throughout the crate.
pub type Result<T> = std::result::Result<T, HybridError>;
