//! # corinth-canal
//!
//! Standalone SNN-logic quantization crate focused on the projector and OlmoeRouter
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
//!        ▼  OlmoeRouter
//! expert_weights + selected_experts + hidden
//! ```
//!
//! ## Quick start
//!
//! ```no_run
//! use corinth_canal::model::{Model, ModelConfig};
//! use corinth_canal::telemetry::TelemetrySnapshot;
//!
//! let mut model = Model::new(ModelConfig::default()).unwrap();
//! let output = model.forward(&TelemetrySnapshot::default()).unwrap();
//!
//! println!("Selected experts: {:?}", output.selected_experts);
//! ```

pub mod error;
pub mod funnel;
#[cfg(feature = "cuda")]
pub mod gpu;
pub mod heartbeat;
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
pub use heartbeat::HeartbeatInjector;
pub use latent::{
    SaaqUpdateRule, SnnDualLatentCalibrator, SnnLatentCalibrator, SnnLatentCsvExporter,
    SnnLatentSnapshot,
};
pub use telemetry::TelemetryEncoder;
pub use types::{EMBEDDING_DIM, HeartbeatConfig, ModelFamily, TelemetrySnapshot};

pub mod tensor;

// New folder name metric, came out of triple code duplication
pub(crate) mod metric;
