//! Public data types for `corinth-canal`.

use serde::{Deserialize, Serialize};

/// Dimensionality of the dense embedding the projector hands to OLMoE.
pub const EMBEDDING_DIM: usize = 2048;

/// Minimal local telemetry payload used to seed deterministic spike patterns.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TelemetrySnapshot {
    pub timestamp_ms: u64,
    pub gpu_temp_c: f32,
    pub gpu_power_w: f32,
    pub gpu_clock_mhz: f32,
    pub mem_util_pct: f32,
    pub cpu_tctl_c: f32,
    pub cpu_package_power_w: f32,
    pub workload_throughput: f64,
    pub workload_efficiency: f64,
    pub auxiliary_signal: f64,
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
            num_experts: 8,
            top_k_experts: 1,
            olmoe_execution_mode: OlmoeExecutionMode::SpikingSim,
            snn_steps: 20,
            projection_mode: ProjectionMode::SpikingTernary,
        }
    }
}

/// Strategy used to convert spike activity into an OLMoE embedding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProjectionMode {
    RateSum,
    TemporalHistogram,
    MembraneSnapshot,
    SpikingTernary,
}

impl Default for ProjectionMode {
    fn default() -> Self {
        Self::SpikingTernary
    }
}

/// Execution mode used by the OLMoE router.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OlmoeExecutionMode {
    StubUniform,
    DenseSim,
    SpikingSim,
}

impl Default for OlmoeExecutionMode {
    fn default() -> Self {
        Self::SpikingSim
    }
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
