//! GPU temporal loop helpers for [`Model`](Model).

use super::Model;
use super::{
    core::{IZ_NEURONS, N_NEURONS, resolve_gpu_routing_telemetry_path},
    telemetry_io::append_gpu_routing_telemetry_row,
};
use crate::funnel::active_neuron_indices;
use crate::gpu::{GpuAccelerator, GpuError, GpuResult};
use crate::types::{ModelOutput, TelemetrySnapshot};
impl Model {
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
    ) -> GpuResult<ModelOutput> {
        let neuron_count = self.projector.input_neurons();

        accelerator.ensure_temporal_state(neuron_count)?;
        self.ensure_temporal_synapse_weights(accelerator, neuron_count)?;
        accelerator.reset_temporal_state()?;
        accelerator.project_snapshot_current(snap, neuron_count)?;

        let mut spike_train: Vec<Vec<usize>> = Vec::with_capacity(self.config.snn_steps);
        let mut best_walker = 0u32;

        for _ in 0..self.config.snn_steps {
            let walker = accelerator.gif_step_weighted_tick(neuron_count)?;
            best_walker = walker;

            let spikes = accelerator.temporal_spikes_to_vec(neuron_count)?;
            let active_neurons = active_neuron_indices(&spikes);
            spike_train.push(active_neurons);
        }

        let membrane = accelerator.temporal_membrane_to_vec(neuron_count)?;
        let potentials: Vec<f32> = membrane.iter().map(|&v| v.clamp(0.0, 1.0)).collect();

        let iz_potentials = vec![0.0f32; IZ_NEURONS];
        let output = self
            .forward_activity(&spike_train, &potentials, &iz_potentials)
            .map_err(|e| GpuError::LaunchFailed(format!("forward_activity failed: {e}")))?;

        let total_spikes: usize = spike_train.iter().map(|s| s.len()).sum();
        let active_fraction = if neuron_count > 0 && self.config.snn_steps > 0 {
            (total_spikes as f32) / (neuron_count as f32 * self.config.snn_steps as f32)
        } else {
            0.0
        };
        let mean_adaptation = 0.25f32;
        let target = resolve_gpu_routing_telemetry_path(&self.config);
        let _ = append_gpu_routing_telemetry_row(
            &target,
            self.global_step as usize,
            0,
            best_walker as i32,
            total_spikes,
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
        // Capability is decided by the adapter/probe layer via
        // `real_gpu_synapse_tensor_name()`. The runtime must not re-discover
        // F16 support from `self.config.gpu_synapse_tensor_name`: doing so
        // can hit quantized (IQ/Q) tensors on non-F16 GGUF checkpoints and
        // abort the SAAQ campaign. If no real F16 tensor is validated, fall
        // through to the Q8_0 dequantized path or the synthetic-fallback below.
        if let Some(tensor_name) = self
            .router
            .real_gpu_synapse_tensor_name()
            .map(str::to_owned)
        {
            let signature = format!("{}::{tensor_name}", self.router.model_path());
            let weights = self
                .router
                .registered_gpu_synapse_weights(&tensor_name)
                .map_err(|e| {
                    GpuError::MemoryError(format!("GGUF synapse registration failed: {e}"))
                })?;
            accelerator.load_synapse_weights_f16_registered(&signature, weights)?;
            return Ok(());
        }

        // Q8_0 dequantized path: only invoked when the adapter confirmed that
        // the preferred tensor is Q8_0 with width divisible by 32.  We check
        // the GPU signature before dequantizing to avoid the allocation cost
        // on repeated calls.
        if let Some(tensor_name) = self
            .router
            .dequantized_q8_0_synapse_tensor_name()
            .map(str::to_owned)
        {
            let fallback_signature = format!("synthetic-f32::{neuron_count}");
            if accelerator.synapse_signature() == Some(fallback_signature.as_str()) {
                return Ok(());
            }
            let signature = format!(
                "dequantized-q8_0::{}::{tensor_name}",
                self.router.model_path()
            );
            if accelerator.synapse_signature() != Some(signature.as_str()) {
                let weights = self
                    .router
                    .dequantized_q8_0_synapse_weights(&tensor_name)
                    .map_err(|e| {
                        GpuError::MemoryError(format!("Q8_0 dequantization failed: {e}"))
                    })?;
                let expected = neuron_count
                    .checked_mul(neuron_count)
                    .ok_or_else(|| GpuError::MemoryError("neuron_count² overflows usize".into()))?;
                if weights.len() == expected {
                    accelerator.load_synapse_weights_named(&signature, &weights)?;
                    return Ok(());
                }
                // Tensor element count does not match neuron_count²; fall
                // through to the synthetic-fallback below.
            } else {
                return Ok(());
            }
        }

        // Q5_K dequantized path: only invoked when the adapter confirmed that
        // the preferred tensor is Q5_K with width divisible by 256.  We check
        // the GPU signature before dequantizing to avoid the allocation cost
        // on repeated calls.
        if let Some(tensor_name) = self
            .router
            .dequantized_q5_k_synapse_tensor_name()
            .map(str::to_owned)
        {
            let fallback_signature = format!("synthetic-f32::{neuron_count}");
            if accelerator.synapse_signature() == Some(fallback_signature.as_str()) {
                return Ok(());
            }
            let signature = format!(
                "dequantized-q5_k::{}::{tensor_name}",
                self.router.model_path()
            );
            if accelerator.synapse_signature() != Some(signature.as_str()) {
                let weights = self
                    .router
                    .dequantized_q5_k_synapse_weights(&tensor_name)
                    .map_err(|e| {
                        GpuError::MemoryError(format!("Q5_K dequantization failed: {e}"))
                    })?;
                let expected = neuron_count
                    .checked_mul(neuron_count)
                    .ok_or_else(|| GpuError::MemoryError("neuron_count² overflows usize".into()))?;
                if weights.len() == expected {
                    accelerator.load_synapse_weights_named(&signature, &weights)?;
                    return Ok(());
                }
                // Tensor element count does not match neuron_count²; fall
                // through to the synthetic-fallback below.
            } else {
                return Ok(());
            }
        }

        let fallback_signature = format!("synthetic-f32::{neuron_count}");
        if accelerator.synapse_signature() == Some(fallback_signature.as_str()) {
            return Ok(());
        }

        let synthetic_weights = vec![0.0f32; neuron_count * neuron_count];
        accelerator.load_synapse_weights_named(&fallback_signature, &synthetic_weights)?;
        Ok(())
    }

    pub(super) fn synthetic_activity(
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
}
