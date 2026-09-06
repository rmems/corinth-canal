// SPDX-License-Identifier: Apache-2.0 OR MIT
//! GPU temporal loop helpers for [`Model`](Model).

use super::Model;
use super::{
    core::{IZ_NEURONS, N_NEURONS, resolve_gpu_routing_telemetry_path},
    telemetry_io::append_gpu_routing_telemetry_row,
};
use crate::funnel::active_neuron_indices;
use crate::gpu::{GpuAccelerator, GpuError, GpuResult};
use crate::moe::SYNTHETIC_FALLBACK_SOURCE;
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

        if self.load_dequant_synapse(
            "q8_0",
            |r| r.dequantized_q8_0_synapse_tensor_name(),
            |r, n| r.dequantized_q8_0_synapse_weights(n),
            accelerator,
            neuron_count,
        )? {
            return Ok(());
        }
        if self.load_dequant_synapse(
            "q5_k",
            |r| r.dequantized_q5_k_synapse_tensor_name(),
            |r, n| r.dequantized_q5_k_synapse_weights(n),
            accelerator,
            neuron_count,
        )? {
            return Ok(());
        }
        if self.load_dequant_synapse(
            "q6_k",
            |r| r.dequantized_q6_k_synapse_tensor_name(),
            |r, n| r.dequantized_q6_k_synapse_weights(n),
            accelerator,
            neuron_count,
        )? {
            return Ok(());
        }
        if self.load_dequant_synapse(
            "iq3_m",
            |r| r.dequantized_iq3_m_synapse_tensor_name(),
            |r, n| r.dequantized_iq3_m_synapse_weights(n),
            accelerator,
            neuron_count,
        )? {
            return Ok(());
        }
        if self.load_dequant_synapse(
            "int4",
            |r| r.dequantized_int4_synapse_tensor_name(),
            |r, n| r.dequantized_int4_synapse_weights(n),
            accelerator,
            neuron_count,
        )? {
            return Ok(());
        }
        if self.load_dequant_synapse(
            "routing-f32",
            |r| r.routing_f32_synapse_tensor_name(),
            |r, n| r.routing_f32_synapse_weights(n),
            accelerator,
            neuron_count,
        )? {
            return Ok(());
        }

        // Nothing above produced weights, so the GIF layer is about to run with
        // an all-zero synapse matrix — i.e. zero recurrent drive. That is a
        // legitimate state for a synthetic run, but it is indistinguishable in
        // the artifacts from a real checkpoint whose tensors failed to load,
        // because `synapse_source` was decided at router-load time and is not
        // revisited here.
        let declared_source = self.router.synapse_source().to_owned();
        if declared_source != SYNTHETIC_FALLBACK_SOURCE {
            tracing::warn!(
                declared_synapse_source = %declared_source,
                neuron_count,
                "no GPU synapse weights could be loaded; falling back to an all-zero                  synapse matrix even though the router reports a real source. The run                  will complete with zero recurrent drive and run_manifest.json will                  still name the declared source."
            );
        } else {
            tracing::debug!(
                neuron_count,
                "using the synthetic all-zero synapse matrix (no checkpoint mapped)"
            );
        }

        let fallback_signature = format!("synthetic-f32::{neuron_count}");
        if accelerator.synapse_signature() == Some(fallback_signature.as_str()) {
            return Ok(());
        }
        let synthetic_weights = vec![0.0f32; neuron_count * neuron_count];
        accelerator.load_synapse_weights_named(&fallback_signature, &synthetic_weights)?;
        Ok(())
    }

    /// Attempt to load a dequantized synapse tensor for the given quantization
    /// format. Returns `Ok(true)` if weights were loaded or already present on
    /// the GPU, `Ok(false)` if no tensor is available for this format.
    fn load_dequant_synapse(
        &mut self,
        label: &str,
        get_name: fn(&crate::moe::Router) -> Option<&str>,
        get_weights: fn(&crate::moe::Router, &str) -> crate::error::Result<Vec<f32>>,
        accelerator: &mut GpuAccelerator,
        neuron_count: usize,
    ) -> GpuResult<bool> {
        let tensor_name = match get_name(&self.router) {
            Some(n) => n.to_owned(),
            None => return Ok(false),
        };
        let fallback_signature = format!("synthetic-f32::{neuron_count}");
        let signature = format!(
            "dequantized-{label}::{}::{tensor_name}",
            self.router.model_path()
        );
        if accelerator.synapse_signature() == Some(signature.as_str()) {
            return Ok(true);
        }
        if accelerator.synapse_signature() == Some(fallback_signature.as_str()) {
            return Ok(false);
        }
        let weights = get_weights(&self.router, &tensor_name)
            .map_err(|e| GpuError::MemoryError(format!("{label} dequantization failed: {e}")))?;
        let (src_rows, src_cols) = self
            .router
            .synapse_tensor_row_major_shape(&tensor_name)
            .map_err(|e| {
                GpuError::MemoryError(format!("synapse tensor shape lookup failed: {e}"))
            })?;
        let final_weights = if src_rows == neuron_count && src_cols == neuron_count {
            weights
        } else {
            Self::resample_weights_to_square(&weights, neuron_count, src_rows, src_cols)
        };
        accelerator.load_synapse_weights_named(&signature, &final_weights)?;
        Ok(true)
    }

    /// Resample a non-square weight tensor into a `[neuron_count × neuron_count]`
    /// matrix using bilinear interpolation over logical rows/columns.
    ///
    /// `src_rows` / `src_cols` must match the GGUF tensor layout (row-major:
    /// `dims[0]` contiguous columns per row, `dims[1]` rows), as produced by the
    /// checkpoint Q8_0 / Q5_K dequantizers.
    ///
    /// GGUF tensors like Gemma4 `[2816, 4096]` or LlamaMoe `[3072, 3072]` don't
    /// match the SNN's fixed 2048-neuron grid.  Rather than falling through to
    /// all-zero synthetic weights (which produce zero GIF drive), we resample the
    /// real weight structure so the neuron population inherits the trained gate
    /// distribution.
    fn resample_weights_to_square(
        src: &[f32],
        n: usize,
        src_rows: usize,
        src_cols: usize,
    ) -> Vec<f32> {
        let total = src.len();
        if total == 0 || n == 0 || src_rows == 0 || src_cols == 0 {
            return vec![0.0; n * n];
        }

        let safe_get = |r: usize, c: usize| -> f32 {
            let idx = r * src_cols + c;
            if idx < total { src[idx] } else { 0.0 }
        };

        let mut out = vec![0.0f32; n * n];
        let row_scale = if n > 1 {
            (src_rows.saturating_sub(1)) as f64 / (n - 1) as f64
        } else {
            0.0
        };
        let col_scale = if n > 1 {
            (src_cols.saturating_sub(1)) as f64 / (n - 1) as f64
        } else {
            0.0
        };

        for r in 0..n {
            let src_r = r as f64 * row_scale;
            let r0 = src_r.floor() as usize;
            let r1 = (r0 + 1).min(src_rows.saturating_sub(1));
            let tr = (src_r - r0 as f64) as f32;
            for c in 0..n {
                let src_c = c as f64 * col_scale;
                let c0 = src_c.floor() as usize;
                let c1 = (c0 + 1).min(src_cols.saturating_sub(1));
                let tc = (src_c - c0 as f64) as f32;

                let v00 = safe_get(r0, c0);
                let v01 = safe_get(r0, c1);
                let v10 = safe_get(r1, c0);
                let v11 = safe_get(r1, c1);
                out[r * n + c] = v00 * (1.0 - tr) * (1.0 - tc)
                    + v01 * (1.0 - tr) * tc
                    + v10 * tr * (1.0 - tc)
                    + v11 * tr * tc;
            }
        }
        out
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

#[cfg(test)]
mod tests {
    use super::Model;

    #[test]
    fn resample_weights_to_square_preserves_square_grid() {
        let src = vec![0.0, 1.0, 2.0, 3.0];
        let out = Model::resample_weights_to_square(&src, 2, 2, 2);
        assert_eq!(out, src);
    }

    #[test]
    fn resample_weights_to_square_uses_rectangular_source_shape() {
        let src = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let out = Model::resample_weights_to_square(&src, 2, 2, 4);
        assert_eq!(out, vec![0.0, 3.0, 4.0, 7.0]);
    }
}
