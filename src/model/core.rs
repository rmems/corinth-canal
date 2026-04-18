//! Core model construction, validation, and non-GPU orchestration.

use super::telemetry_io::append_gpu_routing_telemetry_row;
use crate::error::{HybridError, Result};
use crate::gpu::{GpuAccelerator, GpuBuffer, GpuError, GpuResult};
use crate::moe::OlmoeRouter;
use crate::projector::Projector;
use crate::types::{
    EMBEDDING_DIM, ModelConfig, ModelOutput, RoutingMode, TelemetrySnapshot,
};
use std::path::Path;

pub(super) const N_NEURONS: usize = 2048;
pub(super) const IZ_NEURONS: usize = 5;
pub(super) const GPU_ROUTING_TELEMETRY_HEADER: &str =
    "token_idx,best_score,best_walker,spike_count,mean_adaptation,active_fraction";
pub(super) const GPU_ROUTING_TELEMETRY_PATH: &str = "snn_gpu_routing_telemetry.csv";

/// Standalone runtime that keeps the projector and router logic real while
/// replacing the original front-end with deterministic synthetic spikes.
pub struct Model {
    pub(super) config: ModelConfig,
    pub(super) projector: Projector,
    pub(super) router: OlmoeRouter,
    pub(super) global_step: i64,
}

impl Model {
    pub fn new(config: ModelConfig) -> Result<Self> {
        Self::new_with_projector_neurons(config, N_NEURONS)
    }

    pub fn new_with_projector_neurons(
        config: ModelConfig,
        projector_neurons: usize,
    ) -> Result<Self> {
        if config.snn_steps == 0 {
            return Err(HybridError::InvalidConfig("snn_steps must be ≥ 1".into()));
        }
        if config.num_experts == 0 {
            return Err(HybridError::InvalidConfig("num_experts must be ≥ 1".into()));
        }
        if config.top_k_experts == 0 {
            return Err(HybridError::InvalidConfig(
                "top_k_experts must be ≥ 1".into(),
            ));
        }
        if config.top_k_experts > config.num_experts {
            return Err(HybridError::InvalidConfig(format!(
                "top_k_experts ({}) > num_experts ({})",
                config.top_k_experts, config.num_experts
            )));
        }

        let projector = Projector::with_input_neurons(config.projection_mode, projector_neurons);
        let router = OlmoeRouter::load_with_mode(
            &config.gguf_checkpoint_path,
            config.num_experts,
            config.top_k_experts,
            config.routing_mode,
        )?;

        Ok(Self {
            config,
            projector,
            router,
            global_step: 0,
        })
    }

    pub fn forward(&mut self, snap: &TelemetrySnapshot) -> Result<ModelOutput> {
        let (spike_train, potentials, iz_potentials) = self.synthetic_activity(snap);
        self.forward_activity(
            spike_train.as_slice(),
            potentials.as_slice(),
            iz_potentials.as_slice(),
        )
    }

    pub fn forward_activity(
        &mut self,
        spike_train: &[Vec<usize>],
        potentials: &[f32],
        iz_potentials: &[f32],
    ) -> Result<ModelOutput> {
        self.global_step += 1;

        let embedding = self
            .projector
            .project(spike_train, potentials, iz_potentials)?;
        let router_out = self.router.forward(&embedding)?;

        let steps_f = spike_train.len().max(1) as f32;
        let neuron_count = self.projector.input_neurons();
        let mut counts = vec![0usize; neuron_count];
        for step in spike_train {
            for &idx in step {
                if idx < neuron_count {
                    counts[idx] += 1;
                }
            }
        }
        let firing_rates: Vec<f32> = counts.iter().map(|&c| c as f32 / steps_f).collect();

        Ok(ModelOutput {
            spike_train: spike_train.to_vec(),
            firing_rates,
            membrane_potentials: potentials.to_vec(),
            embedding,
            expert_weights: Some(router_out.expert_weights),
            selected_experts: Some(router_out.selected_experts),
            reasoning: None,
        })
    }

    pub fn train_step(&mut self, snap: &TelemetrySnapshot, target: &[f32]) -> Result<f32> {
        if target.len() != EMBEDDING_DIM {
            return Err(HybridError::InputLengthMismatch {
                expected: EMBEDDING_DIM,
                got: target.len(),
            });
        }

        let output = self.forward(snap)?;
        let loss = output
            .embedding
            .iter()
            .zip(target.iter())
            .map(|(hidden, expected)| (hidden - expected).powi(2))
            .sum::<f32>()
            / EMBEDDING_DIM as f32;

        Ok(loss)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn compute_routing_telemetry(
        &self,
        accelerator: &GpuAccelerator,
        assignment: &GpuBuffer<u8>,
        sat_flags: &mut GpuBuffer<u8>,
        scores: &GpuBuffer<i32>,
        clauses: &GpuBuffer<i32>,
        n_walkers: i32,
        n_vars: i32,
        n_clauses: i32,
        clause_len: i32,
        token_idx: usize,
    ) -> GpuResult<()> {
        let mut best_score = GpuBuffer::<i32>::alloc(1)?;
        let mut best_walker = GpuBuffer::<i32>::alloc(1)?;

        accelerator.satsolver_aux_reduce_best(
            assignment,
            sat_flags,
            scores,
            &mut best_score,
            &mut best_walker,
            clauses,
            n_walkers,
            n_vars,
            n_clauses,
            clause_len,
        )?;

        let final_score =
            best_score.to_vec()?.first().copied().ok_or_else(|| {
                GpuError::MemoryError("best_score buffer returned no values".into())
            })?;
        let final_walker =
            best_walker.to_vec()?.first().copied().ok_or_else(|| {
                GpuError::MemoryError("best_walker buffer returned no values".into())
            })?;

        append_gpu_routing_telemetry_row(
            Path::new(GPU_ROUTING_TELEMETRY_PATH),
            token_idx,
            final_score,
            final_walker,
            0,
            0.0,
            0.0,
        )
    }

    pub fn reset(&mut self) {
        self.projector.reset_membrane();
        self.router.reset_state();
        self.global_step = 0;
    }

    pub fn global_step(&self) -> i64 {
        self.global_step
    }

    pub fn routing_mode(&self) -> RoutingMode {
        self.config.routing_mode
    }

    pub fn router_loaded(&self) -> bool {
        self.router.is_loaded()
    }

    /// Extract the embedding vector for a single token ID from the GGUF `token_embd.weight` tensor.
    pub fn extract_token_embedding(&mut self, token_id: usize) -> Result<Vec<f32>> {
        self.router.extract_token_embedding(token_id)
    }

    pub fn config(&self) -> &ModelConfig {
        &self.config
    }

    pub fn projector_mut(&mut self) -> &mut Projector {
        &mut self.projector
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::funnel::FUNNEL_HIDDEN_NEURONS;
    use crate::types::{RoutingMode, ProjectionMode};

    fn default_model() -> Model {
        Model::new(ModelConfig::default()).expect("default model init failed")
    }

    #[test]
    fn test_model_creation() {
        let model = default_model();
        assert_eq!(model.global_step(), 0);
        assert!(!model.router_loaded());
    }

    #[test]
    fn test_forward_smoke() {
        let mut model = default_model();
        let out = model.forward(&TelemetrySnapshot::default()).unwrap();
        assert_eq!(out.firing_rates.len(), N_NEURONS);
        assert_eq!(out.embedding.len(), EMBEDDING_DIM);
        assert!(out.expert_weights.is_some());
        assert!(out.selected_experts.is_some());
    }

    #[test]
    fn test_step_counter_increments() {
        let mut model = default_model();
        let snap = TelemetrySnapshot::default();
        model.forward(&snap).unwrap();
        model.forward(&snap).unwrap();
        assert_eq!(model.global_step(), 2);
    }

    #[test]
    fn test_reset_clears_step_counter() {
        let mut model = default_model();
        model.forward(&TelemetrySnapshot::default()).unwrap();
        model.reset();
        assert_eq!(model.global_step(), 0);
    }

    #[test]
    fn test_reset_clears_router_spiking_state() {
        let cfg = ModelConfig {
            routing_mode: RoutingMode::SpikingSim,
            ..Default::default()
        };
        let mut model = Model::new(cfg).unwrap();
        let strong_embedding = vec![1.0_f32; EMBEDDING_DIM];

        for _ in 0..8 {
            model.router.forward(&strong_embedding).unwrap();
        }

        assert!(model.router.has_state_activity());
        model.reset();
        assert!(!model.router.has_state_activity());
    }

    #[test]
    fn test_invalid_snn_steps_zero() {
        let cfg = ModelConfig {
            snn_steps: 0,
            ..Default::default()
        };
        assert!(Model::new(cfg).is_err());
    }

    #[test]
    fn test_invalid_top_k_exceeds_experts() {
        let cfg = ModelConfig {
            num_experts: 4,
            top_k_experts: 8,
            ..Default::default()
        };
        assert!(Model::new(cfg).is_err());
    }

    #[test]
    fn test_train_step_non_negative_loss() {
        let mut model = default_model();
        let snap = TelemetrySnapshot {
            gpu_temp_c: 72.0,
            gpu_power_w: 280.0,
            cpu_tctl_c: 65.0,
            cpu_package_power_w: 120.0,
            timestamp_ms: 1_000,
        };
        let loss = model
            .train_step(&snap, &vec![0.1_f32; EMBEDDING_DIM])
            .unwrap();
        assert!(loss >= 0.0, "loss must be non-negative, got {loss}");
    }

    #[test]
    fn test_all_projection_modes() {
        for mode in [
            ProjectionMode::RateSum,
            ProjectionMode::TemporalHistogram,
            ProjectionMode::MembraneSnapshot,
            ProjectionMode::SpikingTernary,
        ] {
            let cfg = ModelConfig {
                projection_mode: mode,
                ..Default::default()
            };
            let mut model = Model::new(cfg).unwrap();
            let out = model.forward(&TelemetrySnapshot::default()).unwrap();
            assert_eq!(out.embedding.len(), EMBEDDING_DIM, "mode: {mode:?}");
        }
    }

    #[test]
    fn test_forward_activity_supports_custom_projector_width() {
        let mut model =
            Model::new_with_projector_neurons(ModelConfig::default(), FUNNEL_HIDDEN_NEURONS)
                .unwrap();
        let spike_train = vec![vec![0, 1, 2, 3]; 20];
        let potentials = vec![0.4; FUNNEL_HIDDEN_NEURONS];
        let iz_potentials = vec![0.0; IZ_NEURONS];

        let out = model
            .forward_activity(&spike_train, &potentials, &iz_potentials)
            .unwrap();

        assert_eq!(out.firing_rates.len(), FUNNEL_HIDDEN_NEURONS);
        assert_eq!(out.membrane_potentials.len(), FUNNEL_HIDDEN_NEURONS);
        assert_eq!(out.embedding.len(), EMBEDDING_DIM);
    }

    #[test]
    fn test_forward_gpu_temporal_requires_gpu() {
        let mut accelerator = GpuAccelerator::new_stub_for_tests();
        let mut model = Model::new(ModelConfig::default()).unwrap();
        let snap = TelemetrySnapshot::default();

        let err = model
            .forward_gpu_temporal(&mut accelerator, &snap)
            .unwrap_err();
        assert!(matches!(err, GpuError::NoGpu));
    }

    #[test]
    fn test_tick_gpu_temporal_requires_gpu() {
        let mut accelerator = GpuAccelerator::new_stub_for_tests();
        let mut model = Model::new(ModelConfig::default()).unwrap();
        let input = vec![0.1_f32; N_NEURONS];

        let err = model
            .tick_gpu_temporal(&mut accelerator, &input)
            .unwrap_err();
        assert!(matches!(err, GpuError::NoGpu));
    }

    #[test]
    fn test_tick_gpu_temporal_rejects_wrong_input_length() {
        let mut accelerator = GpuAccelerator::new_stub_for_tests();
        let mut model = Model::new(ModelConfig::default()).unwrap();
        let input = vec![0.1_f32; 64];

        let err = model
            .tick_gpu_temporal(&mut accelerator, &input)
            .unwrap_err();
        assert!(matches!(err, GpuError::MemoryError(_)));
        assert!(
            err.to_string()
                .contains("gpu temporal input length mismatch"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn test_gpu_routing_telemetry_writer_adds_header_once() {
        let path = std::env::temp_dir().join(format!(
            "corinth_canal_gpu_routing_{}.csv",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));

        append_gpu_routing_telemetry_row(&path, 0, 11, 3, 42, 0.12, 0.021).unwrap();
        append_gpu_routing_telemetry_row(&path, 1, 7, 2, 18, 0.45, 0.009).unwrap();

        let contents = std::fs::read_to_string(&path).unwrap();
        let mut lines = contents.lines();
        assert_eq!(lines.next().unwrap(), GPU_ROUTING_TELEMETRY_HEADER);
        assert_eq!(lines.next().unwrap(), "0,11,3,42,0.1200,0.0210");
        assert_eq!(lines.next().unwrap(), "1,7,2,18,0.4500,0.0090");
        assert!(lines.next().is_none());

        let _ = std::fs::remove_file(path);
    }
}
