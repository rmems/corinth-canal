//! `HybridModel` orchestrates the standalone dummy-spike -> projector -> OLMoE path.

use super::olmoe::OLMoE;
use super::projector::Projector;
use crate::error::{HybridError, Result};
use crate::types::{
    EMBEDDING_DIM, HybridConfig, HybridOutput, TelemetrySnapshot,
};

const N_NEURONS: usize = 16;
const IZ_NEURONS: usize = 5;

/// Standalone hybrid model that keeps the projector and OLMoE logic real while
/// replacing the original front-end with deterministic synthetic spikes.
pub struct HybridModel {
    config: HybridConfig,
    projector: Projector,
    pub(crate) olmoe: OLMoE,
    global_step: i64,
}

impl HybridModel {
    pub fn new(config: HybridConfig) -> Result<Self> {
        if config.snn_steps == 0 {
            return Err(HybridError::InvalidConfig("snn_steps must be ≥ 1".into()));
        }
        if config.context_length == 0 {
            return Err(HybridError::InvalidConfig("context_length must be ≥ 1".into()));
        }
        if config.top_k_experts == 0 {
            return Err(HybridError::InvalidConfig("top_k_experts must be ≥ 1".into()));
        }
        if config.top_k_experts > config.num_experts {
            return Err(HybridError::InvalidConfig(format!(
                "top_k_experts ({}) > num_experts ({})",
                config.top_k_experts, config.num_experts
            )));
        }

        let projector = Projector::new(config.projection_mode);
        let olmoe = OLMoE::load_with_mode(
            &config.olmoe_model_path,
            config.num_experts,
            config.top_k_experts,
            config.olmoe_execution_mode,
        )?;

        Ok(Self {
            config,
            projector,
            olmoe,
            global_step: 0,
        })
    }

    pub fn forward(&mut self, snap: &TelemetrySnapshot) -> Result<HybridOutput> {
        self.global_step += 1;

        let mut spike_train = Vec::with_capacity(self.config.snn_steps);
        for step in 0..self.config.snn_steps {
            let active_1 = (step + snap.gpu_temp_c.max(0.0) as usize) % N_NEURONS;
            let active_2 = (active_1 + 5) % N_NEURONS;
            spike_train.push(vec![active_1, active_2]);
        }

        if spike_train.iter().all(|step| step.is_empty()) {
            return Err(HybridError::SilentNetwork {
                steps: self.config.snn_steps,
            });
        }

        let potentials = vec![0.5; N_NEURONS];
        let iz_potentials = vec![0.0; IZ_NEURONS];
        let embedding = self
            .projector
            .project(&spike_train, &potentials, &iz_potentials)?;
        let olmoe_out = self.olmoe.forward(&embedding)?;

        let steps_f = self.config.snn_steps as f32;
        let mut counts = vec![0usize; N_NEURONS];
        for step in &spike_train {
            for &idx in step {
                if idx < N_NEURONS {
                    counts[idx] += 1;
                }
            }
        }
        let firing_rates: Vec<f32> = counts.iter().map(|&c| c as f32 / steps_f).collect();

        Ok(HybridOutput {
            spike_train,
            firing_rates,
            membrane_potentials: potentials,
            embedding,
            expert_weights: Some(olmoe_out.expert_weights),
            selected_experts: Some(olmoe_out.selected_experts),
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

    pub fn reset(&mut self) {
        self.projector.reset_membrane();
        self.olmoe.reset_state();
        self.global_step = 0;
    }

    pub fn global_step(&self) -> i64 {
        self.global_step
    }

    pub fn olmoe_loaded(&self) -> bool {
        self.olmoe.is_loaded()
    }

    pub fn config(&self) -> &HybridConfig {
        &self.config
    }

    pub fn projector_mut(&mut self) -> &mut Projector {
        &mut self.projector
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{OlmoeExecutionMode, ProjectionMode};

    fn default_model() -> HybridModel {
        HybridModel::new(HybridConfig::default()).expect("default model init failed")
    }

    #[test]
    fn test_model_creation() {
        let model = default_model();
        assert_eq!(model.global_step(), 0);
        assert!(!model.olmoe_loaded());
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
    fn test_reset_clears_olmoe_spiking_state() {
        let cfg = HybridConfig {
            olmoe_execution_mode: OlmoeExecutionMode::SpikingSim,
            ..Default::default()
        };
        let mut model = HybridModel::new(cfg).unwrap();
        let strong_embedding = vec![1.0_f32; EMBEDDING_DIM];

        for _ in 0..8 {
            model.olmoe.forward(&strong_embedding).unwrap();
        }

        assert!(model.olmoe.has_state_activity());
        model.reset();
        assert!(!model.olmoe.has_state_activity());
    }

    #[test]
    fn test_invalid_snn_steps_zero() {
        let cfg = HybridConfig {
            snn_steps: 0,
            ..Default::default()
        };
        assert!(HybridModel::new(cfg).is_err());
    }

    #[test]
    fn test_invalid_top_k_exceeds_experts() {
        let cfg = HybridConfig {
            num_experts: 4,
            top_k_experts: 8,
            ..Default::default()
        };
        assert!(HybridModel::new(cfg).is_err());
    }

    #[test]
    fn test_train_step_non_negative_loss() {
        let mut model = default_model();
        let snap = TelemetrySnapshot {
            gpu_temp_c: 72.0,
            gpu_power_w: 280.0,
            cpu_tctl_c: 65.0,
            workload_throughput: 10.0,
            workload_efficiency: 0.7,
            ..Default::default()
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
            let cfg = HybridConfig {
                projection_mode: mode,
                ..Default::default()
            };
            let mut model = HybridModel::new(cfg).unwrap();
            let out = model.forward(&TelemetrySnapshot::default()).unwrap();
            assert_eq!(out.embedding.len(), EMBEDDING_DIM, "mode: {mode:?}");
        }
    }
}
