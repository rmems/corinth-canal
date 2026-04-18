//! Public data types for `corinth-canal`.

use serde::{Deserialize, Serialize};

/// Dimensionality of the dense embedding the projector hands to OLMoE.
pub const EMBEDDING_DIM: usize = 2048;

/// Minimal local telemetry payload used to seed deterministic spike patterns.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TelemetrySnapshot {
    pub gpu_temp_c: f32,
    pub gpu_power_w: f32,
    pub cpu_tctl_c: f32,
    pub cpu_package_power_w: f32,
    pub timestamp_ms: u64,
}

impl TelemetrySnapshot {
    pub fn thermal_stress(&self) -> f32 {
        ((self.gpu_temp_c - 60.0) / 30.0).clamp(0.0, 1.0)
    }
}

/// Top-level configuration for the hybrid quantization pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HybridConfig {
    pub olmoe_model_path: String,
    pub local_checkpoint_dir: String,
    pub gpu_synapse_tensor_name: String,
    pub num_experts: usize,
    pub top_k_experts: usize,
    pub olmoe_execution_mode: OlmoeExecutionMode,
    pub snn_steps: usize,
    pub projection_mode: ProjectionMode,
}

impl Default for HybridConfig {
    fn default() -> Self {
        Self {
            olmoe_model_path: String::new(),
            local_checkpoint_dir: String::new(),
            gpu_synapse_tensor_name: "blk.0.attn_q.weight".into(),
            num_experts: 8,
            top_k_experts: 1,
            olmoe_execution_mode: OlmoeExecutionMode::SpikingSim,
            snn_steps: 20,
            projection_mode: ProjectionMode::SpikingTernary,
        }
    }
}

/// Strategy used to convert spike activity into an OLMoE embedding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum ProjectionMode {
    RateSum,
    TemporalHistogram,
    MembraneSnapshot,
    #[default]
    SpikingTernary,
}

/// Execution mode used by the OLMoE router.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum OlmoeExecutionMode {
    StubUniform,
    DenseSim,
    #[default]
    SpikingSim,
}

/// Output of one `HybridModel::forward` pass.
#[derive(Debug, Clone)]
pub struct HybridOutput {
    pub spike_train: Vec<Vec<usize>>,
    pub firing_rates: Vec<f32>,
    pub membrane_potentials: Vec<f32>,
    pub embedding: Vec<f32>,
    pub expert_weights: Option<Vec<f32>>,
    pub selected_experts: Option<Vec<usize>>,
    pub reasoning: Option<String>,
}
