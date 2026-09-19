// SPDX-License-Identifier: Apache-2.0 OR MIT
//! # corinth-canal
//!
//! `corinth-canal` is the single-crate reference implementation for the `rmems`
//! SNN-logic quantization line.
//!
//! The crate keeps the full telemetry -> spiking -> projector -> router path in
//! one repository so the research loop can be exercised end to end before proven
//! components are promoted into separate `rmems-*` crates.
//!
//! ## Runtime pipeline
//!
//! ```text
//! TelemetrySnapshot
//!        │
//!        ▼  TelemetryEncoder (CPU) / project_snapshot_current (GPU)
//! ternary telemetry events (+1 / 0 / -1 per channel)
//!        │
//!        ▼  SignedSplitBankBridge (CPU) / GPU input_spikes buffer
//! input spike train
//!        │
//!        ▼  SparseGifHiddenLayer (CPU) / gif_step_weighted_tick (GPU)
//! hidden spike train + membrane potentials
//!        │
//!        ▼  Projector
//! dense embedding [EMBEDDING_DIM = 2048]
//!        │
//!        ▼  Router
//! expert_weights + selected_experts + routed hidden state
//!        │
//!        ▼  SAAQ latent calibration / telemetry export
//! ```
//!
//! ## Model loading
//!
//! The routing bridge is GGUF-backed and custom to this repository.
//! `Router` parses GGUF checkpoints in-repo, resolves a supported model
//! family, extracts routing tensors and token embeddings, and selects a GPU
//! synapse source.
//!
//! Selection is priority-ordered over the *preferred* tensor's actual
//! `ggml_type` (never the filename); the first match wins:
//!
//! 1. real `F16` — requires a square `[hidden_size, hidden_size]` tensor
//! 2. dequantized `Q8_0` — row width a multiple of 32
//! 3. dequantized `Q5_K` — row width a multiple of 256
//! 4. dequantized `Q6_K` — row width a multiple of 256
//! 5. dequantized `IQ3_M` — internal block type only, not GGUF wire type 31
//! 6. `routing-f32` — fallback when no preferred tensor matched
//!
//! `SynapseSource::SyntheticFallback` is therefore unreachable on the GGUF
//! path: step 6 also covers the "no preferred tensor at all" case. It remains
//! reachable for the Safetensors backend, which additionally has
//! `dequantized-int4`.
//!
//! Model families are enumerated by the `ModelFamily` enum in `types.rs`
//! (21 variants as of this writing) — consult the enum rather than any prose
//! list.
//!
//! ## Quick start
//!
//! The `model` module is available when the `cuda` feature is enabled (the
//! default feature set).
//!
//! ```no_run
//! # #[cfg(feature = "cuda")] {
//! use corinth_canal::model::{Model, ModelConfig};
//! use corinth_canal::TelemetrySnapshot;
//!
//! let mut model = Model::new(ModelConfig::default()).unwrap();
//! let output = model.forward(&TelemetrySnapshot::default()).unwrap();
//!
//! println!("Selected experts: {:?}", output.selected_experts);
//! # }
//! ```

// rustc 1.98 promoted `chunks_exact_to_as_chunks` to warn, and CI runs
// `-D warnings`, so the GGML / Safetensors block parsers that use
// `chunks_exact` on fixed wire sizes fail the lint gate. Converting them is a
// separate refactor. `unknown_lints` keeps this usable on pre-1.98 toolchains,
// where the lint name does not exist yet and would itself be an error.
#![allow(unknown_lints)]
#![allow(clippy::chunks_exact_to_as_chunks)]

pub mod error;
pub mod experiment;
pub mod funnel;
#[cfg(feature = "cuda")]
pub mod gpu;
pub mod latent;
#[cfg(feature = "cuda")]
pub mod model;
pub mod moe;
pub mod projector;
pub mod telemetry;
pub mod types;

pub use error::{HybridError, Result};
pub use funnel::{
    FUNNEL_HIDDEN_NEURONS, FUNNEL_INPUT_NEURONS, FunnelActivity, SignedSplitBankBridge,
    SparseGifHiddenLayer, TelemetryFunnel,
};
pub use latent::{
    SaaqUpdateRule, SnnDualLatentCalibrator, SnnLatentCalibrator, SnnLatentCsvExporter,
    SnnLatentSnapshot,
};
pub use telemetry::TelemetryEncoder;
pub use types::{
    CloudModelSpec, EMBEDDING_DIM, ModelArchitectureClass, ModelFamily, ModelTarget,
    TelemetrySnapshot,
};

pub use experiment::schema::{
    ExperimentBundle, ExperimentManifest, ExperimentMetrics, ExperimentSummary, ExperimentWarning,
    ModelAdapterConfig, RunEntry, RunMatrix,
};

// New folder name metric, came out of triple code duplication
pub(crate) mod metric;
