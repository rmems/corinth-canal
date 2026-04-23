//! Public data types for `corinth-canal`.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

/// Dimensionality of the dense embedding the projector hands to OlmoeRouter.
pub const EMBEDDING_DIM: usize = 2048;

/// Supported GGUF model families for the router bridge.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum ModelFamily {
    #[default]
    Olmoe,
    Qwen3Moe,
    Gemma4,
    DeepSeek2,
    LlamaMoe,
}

impl ModelFamily {
    pub fn slug(self) -> &'static str {
        match self {
            Self::Olmoe => "olmoe",
            Self::Qwen3Moe => "qwen3_moe",
            Self::Gemma4 => "gemma4",
            Self::DeepSeek2 => "deepseek2",
            Self::LlamaMoe => "llama_moe",
        }
    }
}

/// Deterministic pulse configuration used to perturb telemetry during validation.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct HeartbeatConfig {
    pub enabled: bool,
    pub amplitude: f32,
    pub period_ticks: usize,
    pub duty_cycle: f32,
    pub phase_offset_ticks: usize,
}

impl Default for HeartbeatConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            amplitude: 0.0,
            period_ticks: 64,
            duty_cycle: 0.25,
            phase_offset_ticks: 0,
        }
    }
}

/// Minimal local telemetry payload used to seed deterministic spike patterns.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TelemetrySnapshot {
    pub gpu_temp_c: f32,
    pub gpu_power_w: f32,
    pub cpu_tctl_c: f32,
    pub cpu_package_power_w: f32,
    #[serde(default)]
    pub heartbeat_signal: f32,
    #[serde(default)]
    pub heartbeat_enabled: bool,
    pub timestamp_ms: u64,
}

impl TelemetrySnapshot {
    pub fn thermal_stress(&self) -> f32 {
        ((self.gpu_temp_c - 60.0) / 30.0).clamp(0.0, 1.0)
    }
}

/// Top-level configuration for the hybrid quantization pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub gguf_checkpoint_path: String,
    pub model_family: Option<ModelFamily>,
    pub gpu_synapse_tensor_name: String,
    pub num_experts: usize,
    pub top_k_experts: usize,
    pub routing_mode: RoutingMode,
    pub snn_steps: usize,
    pub projection_mode: ProjectionMode,
    pub heartbeat: HeartbeatConfig,
    /// Destination path for the GPU routing telemetry CSV written by
    /// `Model::forward_gpu_temporal` (and `Model::forward` on the GPU path).
    /// When `None`, the runtime falls back to the legacy CWD-relative
    /// filename `snn_gpu_routing_telemetry.csv`. Prefer an absolute path
    /// anchored in the caller's per-run artifact directory.
    #[serde(default)]
    pub gpu_routing_telemetry_path: Option<PathBuf>,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            gguf_checkpoint_path: String::new(),
            model_family: None,
            gpu_synapse_tensor_name: String::new(),
            num_experts: 8,
            top_k_experts: 1,
            routing_mode: RoutingMode::SpikingSim,
            snn_steps: 20,
            projection_mode: ProjectionMode::SpikingTernary,
            heartbeat: HeartbeatConfig::default(),
            gpu_routing_telemetry_path: None,
        }
    }
}

/// Strategy used to convert spike activity into an OlmoeRouter embedding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum ProjectionMode {
    RateSum,
    TemporalHistogram,
    MembraneSnapshot,
    #[default]
    SpikingTernary,
}

/// Execution mode used by the OlmoeRouter router.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum RoutingMode {
    StubUniform,
    DenseSim,
    #[default]
    SpikingSim,
}

/// Output of one `Model::forward` pass.
#[derive(Debug, Clone)]
pub struct ModelOutput {
    pub spike_train: Vec<Vec<usize>>,
    pub firing_rates: Vec<f32>,
    pub membrane_potentials: Vec<f32>,
    pub embedding: Vec<f32>,
    pub expert_weights: Option<Vec<f32>>,
    pub selected_experts: Option<Vec<usize>>,
    pub reasoning: Option<String>,
}
