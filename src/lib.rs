//! # corinth-canal
//!
//! Standalone SNN-logic quantization crate focused on the projector and OLMoE
//! routing path. The telemetry/SNN front-end is a deterministic in-repo stub
//! that generates repeatable spikes without requiring external services.
//!
//! ## Architecture
//!
//! ```text
//! TelemetrySnapshot
//!        │
//!        ▼  deterministic dummy spike generator
//! spike_train + membrane_potentials
//!        │
//!        ▼  Projector
//! dense embedding [EMBEDDING_DIM = 2048]
//!        │
//!        ▼  OLMoE
//! expert_weights + selected_experts + hidden
//! ```
//!
//! ## Quick start
//!
//! ```no_run
//! use corinth_canal::{HybridConfig, HybridModel, TelemetrySnapshot};
//!
//! let mut model = HybridModel::new(HybridConfig::default()).unwrap();
//! let output = model.forward(&TelemetrySnapshot::default()).unwrap();
//!
//! println!("Selected experts: {:?}", output.selected_experts);
//! ```

pub mod error;
pub mod funnel;
pub mod gpu;
pub mod hybrid;
pub mod latent;
pub mod prompt;
pub mod telemetry;
pub mod tensor;
pub mod transformer;
pub mod types;

pub use error::{HybridError, Result};
pub use funnel::{
    FUNNEL_HIDDEN_NEURONS, FUNNEL_INPUT_NEURONS, FunnelActivity, SignedSplitBankBridge,
    SparseGifHiddenLayer, TelemetryFunnel,
};
pub use hybrid::{HybridModel, OLMoE, Projector};
pub use latent::{SnnLatentCalibrator, SnnLatentCsvExporter, SnnLatentSnapshot};
pub use prompt::encode_prompt_with_tokenizer_json;
pub use telemetry::TelemetryEncoder;
pub use types::{
    EMBEDDING_DIM, HybridConfig, HybridOutput, OlmoeExecutionMode, ProjectionMode,
    TelemetrySnapshot,
};
