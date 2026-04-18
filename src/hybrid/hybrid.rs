//! `HybridModel` orchestrates the standalone dummy-spike -> projector -> OLMoE path.

use super::olmoe::OLMoE;
use super::projector::Projector;
use crate::error::{HybridError, Result};
use crate::gpu::{GpuAccelerator, GpuBuffer, GpuError, GpuResult};
use crate::types::{HybridConfig, HybridOutput, OlmoeExecutionMode, TelemetrySnapshot, EMBEDDING_DIM};
use std::fs::OpenOptions;
use std::io::Write;
use std::path::Path;

const N_NEURONS: usize = 2048;
const IZ_NEURONS: usize = 5;
const GPU_ROUTING_TELEMETRY_HEADER: &str =
    "token_idx,best_score,best_walker,spike_count,mean_adaptation,active_fraction";
const GPU_ROUTING_TELEMETRY_PATH: &str = "snn_gpu_routing_telemetry.csv";

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
        Self::new_with_projector_neurons(config, N_NEURONS)
    }

    pub fn new_with_projector_neurons(
        config: HybridConfig,
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
    ) -> Result<HybridOutput> {
        self.global_step += 1;

        let embedding = self
            .projector
            .project(spike_train, potentials, iz_potentials)?;
        let olmoe_out = self.olmoe.forward(&embedding)?;

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

        Ok(HybridOutput {
            spike_train: spike_train.to_vec(),
            firing_rates,
            membrane_potentials: potentials.to_vec(),
            embedding,
            expert_weights: Some(olmoe_out.expert_weights),
            selected_experts: Some(olmoe_out.selected_experts),
            reasoning: None,
        })
    }

    /// GPU-only temporal simulation with GIF (Generalized Integrate-and-Fire).
    /// Phase 1: reset + load_synapse_weights + project_snapshot_current
    /// Phase 2: gif_step_weighted_tick (with adaptation, dynamic threshold, weighted synapses)
    /// Downloads membrane + adaptation; uses existing projector (GIF-compatible via SpikingTernary).
    /// Fails fast with GpuError::NoGpu if GPU unavailable (no CPU fallback).
    pub fn prepare_gpu_temporal(&mut self, accelerator: &mut GpuAccelerator) -> GpuResult<()> {
        let neuron_count = self.projector.input_neurons();
        accelerator.ensure_temporal_state(neuron_count)?;
        self.ensure_temporal_synapse_weights(accelerator, neuron_count)?;
        accelerator.reset_temporal_state()
    }

    /// Execute exactly one GPU temporal tick using an explicit per-neuron input vector.
    /// Leaves all temporal state resident so repeated calls form a stateful temporal loop.
    pub fn tick_gpu_temporal(
        &mut self,
        accelerator: &mut GpuAccelerator,
        input_spikes: &[f32],
    ) -> GpuResult<u32> {
        let neuron_count = self.projector.input_neurons();
        if input_spikes.len() != neuron_count {
            return Err(GpuError::MemoryError(format!(
                "gpu temporal input length mismatch: expected {neuron_count}, got {}",
                input_spikes.len()
            )));
        }

        accelerator.ensure_temporal_state(neuron_count)?;
        self.ensure_temporal_synapse_weights(accelerator, neuron_count)?;
        accelerator.upload_temporal_input_spikes(input_spikes)?;
        accelerator.gif_step_weighted_tick(neuron_count)
    }

    pub fn forward_gpu_temporal(
        &mut self,
        accelerator: &mut GpuAccelerator,
        snap: &TelemetrySnapshot,
    ) -> GpuResult<HybridOutput> {
        let neuron_count = self.projector.input_neurons();

        accelerator.ensure_temporal_state(neuron_count)?;
        self.ensure_temporal_synapse_weights(accelerator, neuron_count)?;
        accelerator.reset_temporal_state()?;

        // Phase 1: Project snapshot to input_current (can also drive input_spikes)
        accelerator.project_snapshot_current(snap, neuron_count)?;

        // Optional: set some baseline input_spikes from snapshot or previous output for recurrence.
        // For now, the kernel will use loaded weights + any non-zero input_spikes.

        // Phase 2: Run N GIF steps with on-device two-pass SAAQ reduction (no per-tick downloads).
        // gif_step_weighted_tick now internally calls saaq_find_best_walker.
        // Spikes/membrane/adaptation stay in VRAM. Only best_walker (4 bytes) is downloaded.
        let mut spike_train: Vec<Vec<usize>> = Vec::with_capacity(self.config.snn_steps);
        let mut best_walker = 0u32;

        for _ in 0..self.config.snn_steps {
            let walker = accelerator.gif_step_weighted_tick(neuron_count)?; // now returns best_walker from SAAQ
            best_walker = walker; // last one wins for telemetry (or collect if multi-walker SAAQ)

            // Minimal spike download only (or optimize further to stay on-device for projector)
            let spikes = accelerator.temporal_spikes_to_vec(neuron_count)?;
            let active_neurons: Vec<usize> = spikes
                .iter()
                .enumerate()
                .filter(|(_, v)| **v != 0)
                .map(|(i, _)| i)
                .collect();
            spike_train.push(active_neurons);
        }

        // SAAQ now on-device; no full adaptation/membrane download needed for score
        let membrane = accelerator.temporal_membrane_to_vec(neuron_count)?;
        let potentials: Vec<f32> = membrane.iter().map(|&v| v.clamp(0.0, 1.0)).collect();

        // Use existing projector/OLMoE path (CPU-side). Projector supports GIF via SpikingTernary mode.
        let iz_potentials = vec![0.0f32; IZ_NEURONS];
        let output = self
            .forward_activity(&spike_train, &potentials, &iz_potentials)
            .map_err(|e| GpuError::LaunchFailed(format!("forward_activity failed: {e}")))?;

        // SAAQ + GIF sparsity telemetry (spike_count, adaptation stats for history-aware SAT routing,
        // Julia symbolic regression, and 16GB VRAM/power optimization on 2048 neurons)
        let total_spikes: usize = spike_train.iter().map(|s| s.len()).sum();
        let spike_count = total_spikes;
        let active_fraction = if neuron_count > 0 && self.config.snn_steps > 0 {
            (total_spikes as f32) / (neuron_count as f32 * self.config.snn_steps as f32)
        } else {
            0.0
        };
        let mean_adaptation = 0.25f32; // TODO: pull from on-device SAAQ or adaptation buffer in future zero-copy pass
        let _ = append_gpu_routing_telemetry_row(
            Path::new(GPU_ROUTING_TELEMETRY_PATH),
            self.global_step as usize,
            0, // best_score placeholder (SAAQ now on-device)
            best_walker as i32,
            spike_count,
            mean_adaptation,
            active_fraction,
        );

        Ok(output)
    }

    fn ensure_temporal_synapse_weights(
        &mut self,
        accelerator: &mut GpuAccelerator,
        neuron_count: usize,
    ) -> GpuResult<()> {
        if self.olmoe.is_loaded() {
            let signature = format!(
                "{}::{}",
                self.olmoe.model_path(),
                self.config.gpu_synapse_tensor_name
            );
            match self
                .olmoe
                .registered_gpu_synapse_weights(&self.config.gpu_synapse_tensor_name)
            {
                Ok(weights) => {
                    accelerator.load_synapse_weights_f16_registered(&signature, weights)?;
                    return Ok(());
                }
                Err(HybridError::UnsupportedFormat(_)) | Err(HybridError::MissingTensor { .. }) => {
                    // Some GGUF checkpoints keep the routing bridge usable but do not expose a
                    // square F16 attention tensor for direct GPU synapse upload.
                }
                Err(e) => return Err(e),
            }
        }

        let fallback_signature = format!("synthetic-f32::{neuron_count}");
        if accelerator.synapse_signature() == Some(fallback_signature.as_str()) {
            return Ok(());
        }

        // Stub mode keeps a deterministic fallback path without re-uploading every forward.
        let synthetic_weights = vec![0.0f32; neuron_count * neuron_count];
        accelerator.load_synapse_weights_named(&fallback_signature, &synthetic_weights)?;
        Ok(())
    }

    fn synthetic_activity(
        &self,
        snap: &TelemetrySnapshot,
    ) -> (Vec<Vec<usize>>, Vec<f32>, Vec<f32>) {
        let temp_offset = snap.gpu_temp_c.max(0.0).round() as usize % N_NEURONS;

        let spike_train = (0..self.config.snn_steps)
            .map(|step| {
                let lead = (step + temp_offset) % N_NEURONS;
                let trail = (lead + 5) % N_NEURONS;
                vec![lead, trail]
            })
            .collect();

        let potentials = vec![0.25 + 0.5 * snap.thermal_stress(); N_NEURONS];
        let iz_potentials = vec![0.0; IZ_NEURONS];

        (spike_train, potentials, iz_potentials)
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
            0,   // spike_count (extend SAT path later with GIF stats)
            0.0, // mean_adaptation
            0.0, // active_fraction
        )
    }

    pub fn reset(&mut self) {
        self.projector.reset_membrane();
        self.olmoe.reset_state();
        self.global_step = 0;
    }

    pub fn global_step(&self) -> i64 {
        self.global_step
    }

    pub fn olmoe_execution_mode(&self) -> OlmoeExecutionMode {
        self.config.olmoe_execution_mode
    }

    pub fn olmoe_loaded(&self) -> bool {
        self.olmoe.is_loaded()
    }

    pub fn checkpoint_architecture(&self) -> &str {
        self.olmoe.architecture()
    }

    pub fn checkpoint_hidden_size(&self) -> usize {
        self.olmoe.hidden_size()
    }

    pub fn checkpoint_source_hidden_size(&self) -> usize {
        self.olmoe.source_hidden_size()
    }

    pub fn checkpoint_num_experts(&self) -> usize {
        self.olmoe.checkpoint_num_experts()
    }

    pub fn checkpoint_expert_used_count(&self) -> Option<usize> {
        self.olmoe.expert_used_count()
    }

    pub fn routing_tensor_name(&self) -> &str {
        self.olmoe.routing_tensor_name()
    }

    /// Extract the embedding vector for a single token ID from the GGUF `token_embd.weight` tensor.
    pub fn extract_token_embedding(&mut self, token_id: usize) -> Result<Vec<f32>> {
        self.olmoe.extract_token_embedding(token_id)
    }

    /// Number of parameters / experts.
    pub fn config(&self) -> &HybridConfig {
        &self.config
    }

    pub fn projector_mut(&mut self) -> &mut Projector {
        &mut self.projector
    }
}

fn append_gpu_routing_telemetry_row(
    path: &Path,
    token_idx: usize,
    best_score: i32,
    best_walker: i32,
    spike_count: usize,
    mean_adaptation: f32,
    active_fraction: f32,
) -> GpuResult<()> {
    let needs_header = !path.exists()
        || std::fs::metadata(path)
            .map(|metadata| metadata.len() == 0)
            .unwrap_or(true);

    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .map_err(|e| GpuError::MemoryError(format!("CSV Write Failed: {e}")))?;

    if needs_header {
        writeln!(file, "{GPU_ROUTING_TELEMETRY_HEADER}")
            .map_err(|e| GpuError::MemoryError(format!("CSV Write Failed: {e}")))?;
    }

    writeln!(
        file,
        "{token_idx},{best_score},{best_walker},{spike_count},{mean_adaptation:.4},{active_fraction:.4}"
    )
    .map_err(|e| GpuError::MemoryError(format!("CSV Write Failed: {e}")))?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::funnel::FUNNEL_HIDDEN_NEURONS;
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
            let cfg = HybridConfig {
                projection_mode: mode,
                ..Default::default()
            };
            let mut model = HybridModel::new(cfg).unwrap();
            let out = model.forward(&TelemetrySnapshot::default()).unwrap();
            assert_eq!(out.embedding.len(), EMBEDDING_DIM, "mode: {mode:?}");
        }
    }

    #[test]
    fn test_forward_activity_supports_custom_projector_width() {
        let mut model =
            HybridModel::new_with_projector_neurons(HybridConfig::default(), FUNNEL_HIDDEN_NEURONS)
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
        let mut model = HybridModel::new(HybridConfig::default()).unwrap();
        let snap = TelemetrySnapshot::default();

        let err = model
            .forward_gpu_temporal(&mut accelerator, &snap)
            .unwrap_err();
        assert!(matches!(err, GpuError::NoGpu));
    }

    #[test]
    fn test_tick_gpu_temporal_requires_gpu() {
        let mut accelerator = GpuAccelerator::new_stub_for_tests();
        let mut model = HybridModel::new(HybridConfig::default()).unwrap();
        let input = vec![0.1_f32; N_NEURONS];

        let err = model
            .tick_gpu_temporal(&mut accelerator, &input)
            .unwrap_err();
        assert!(matches!(err, GpuError::NoGpu));
    }

    #[test]
    fn test_tick_gpu_temporal_rejects_wrong_input_length() {
        let mut accelerator = GpuAccelerator::new_stub_for_tests();
        let mut model = HybridModel::new(HybridConfig::default()).unwrap();
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
