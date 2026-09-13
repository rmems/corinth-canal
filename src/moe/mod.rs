// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Public MoE router API backed by a family-aware GGUF and Safetensors bridge.
//!
//! Private helpers live in:
//! - `moe/gguf/` for GGUF parsing + mapped tensor access (via `checkpoint` façade)
//! - `moe/checkpoint.rs` compatibility re-export of `gguf/`
//! - `moe/adapter.rs` for model-family detection and tensor selection
//! - `moe/routing.rs` for routing math and embedding resampling
//! - `moe/safetensors.rs` for Safetensors header inspection, manifests, and tensor loading

mod adapter;
pub(crate) use adapter::SYNTHETIC_FALLBACK_SOURCE;
mod checkpoint;
mod ggml;
mod gguf;
mod routing;
pub mod safetensors;

use self::adapter::{ModelAdapter, SynapseSource, resolve_adapter, resolve_safetensors_adapter};
use self::checkpoint::{
    MappedGgufCheckpoint, extract_named_token_embedding_from_checkpoint, probe_and_map_checkpoint,
};
// Constants used by adapter/routing helpers in this module tree.
use self::ggml::{
    GGML_TYPE_F16, GGML_TYPE_F32, GGML_TYPE_IQ3_M_BLOCK, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K,
    GGML_TYPE_Q8_0,
};
// GGUF test fixture helpers (`moe/tests.rs`) pull these via `use super::*`.
// GGUF fixture builders in `tests.rs` pull these via `use super::*`.
#[cfg(test)]
use self::ggml::{
    GGML_TYPE_IQ3_S, GGML_TYPE_Q4_0_4_4, GGUF_MAGIC, GGUF_VALUE_TYPE_STRING,
    GGUF_VALUE_TYPE_UINT32, GGUF_VERSION,
};
pub use self::ggml::{ggml_type_label, synapse_dequant_path_supported};
use self::routing::{
    checkpoint_gate_scores, normalize_l2, normalize_to_internal_embedding_dim, resample_embedding,
    safetensors_gate_scores, softmax, synthetic_gate_scores, top_k_indices,
};
use self::safetensors::MappedSafetensorsCheckpoint;
use crate::error::{HybridError, Result};
pub use crate::types::RoutingMode;
use crate::types::{EMBEDDING_DIM, ModelFamily};

/// Diagnostic snapshot of the GGUF tensor that the adapter wants to use as
/// the GPU synapse weight source for this router. Returned by
/// [`Router::preferred_gpu_synapse_tensor_descriptor`]; only consumed
/// by `examples/synapse_diagnostic.rs` today, but exposed publicly so
/// future runner / manifest stamping can reuse it without re-mapping the
/// checkpoint. Carries no live tensor data.
#[derive(Debug, Clone, serde::Serialize)]
pub struct GpuSynapseTensorDescriptor {
    pub name: String,
    pub ggml_type_id: u32,
    pub ggml_type_label: &'static str,
    pub dims: Vec<usize>,
    /// `true` iff the runtime currently has a code path that can consume
    /// this `ggml_type` as GPU synapse weights. `F16` uses the registered
    /// direct-load path; supported quantized tensors use dequantized F32
    /// paths, while unsupported tensors can still use a checkpoint-backed
    /// routing-gate fallback.
    pub has_dequant_path: bool,
}

impl RouterMetadata {
    fn synthetic(family: ModelFamily, num_experts: usize, top_k: usize) -> Self {
        Self {
            family,
            architecture: "stub".into(),
            hidden_size: EMBEDDING_DIM,
            num_layers: 0,
            num_experts: num_experts.max(1),
            expert_used_count: top_k.max(1),
            quantization: "stub".into(),
            routing_tensor_name: "synthetic".into(),
            preferred_gpu_synapse_tensor_name: None,
            synapse_source: SYNTHETIC_FALLBACK_SOURCE.into(),
            real_gpu_synapse_tensor_name: None,
        }
    }

    fn from_adapter(adapter: &ModelAdapter) -> Self {
        // Touch active dequant field map so none of the Option tensor names
        // are dead storage (values still flow through the named accessors).
        let _active = adapter.active_synapse_tensor_name();
        Self {
            family: adapter.family,
            architecture: adapter.architecture.clone(),
            hidden_size: adapter.hidden_size,
            num_layers: adapter.num_layers,
            num_experts: adapter.num_experts,
            expert_used_count: adapter.expert_used_count,
            quantization: adapter.quantization.clone(),
            routing_tensor_name: adapter.routing_tensor.clone(),
            preferred_gpu_synapse_tensor_name: adapter.preferred_gpu_synapse_tensor.clone(),
            synapse_source: adapter.synapse_source_label().into(),
            real_gpu_synapse_tensor_name: adapter.real_gpu_synapse_tensor.clone(),
        }
    }
}

enum CheckpointBackend {
    Gguf(MappedGgufCheckpoint),
    Safetensors(MappedSafetensorsCheckpoint),
}

pub struct Router {
    metadata: RouterMetadata,
    adapter: Option<ModelAdapter>,
    model_path: String,
    num_experts: usize,
    top_k: usize,
    loaded: bool,
    routing_mode: RoutingMode,
    expert_membranes: Vec<f32>,
    hidden_membranes: Vec<f32>,
    threshold: f32,
    decay: f32,
    checkpoint: Option<CheckpointBackend>,
}

#[derive(Debug, Clone, Default)]
pub struct RouterMetadata {
    pub family: ModelFamily,
    pub architecture: String,
    pub hidden_size: usize,
    pub num_layers: usize,
    pub num_experts: usize,
    pub expert_used_count: usize,
    pub quantization: String,
    pub routing_tensor_name: String,
    pub preferred_gpu_synapse_tensor_name: Option<String>,
    pub synapse_source: String,
    pub real_gpu_synapse_tensor_name: Option<String>,
}

#[derive(Debug, Clone)]
pub struct RouterOutput {
    pub expert_weights: Vec<f32>,
    pub selected_experts: Vec<usize>,
    pub hidden: Vec<f32>,
}

impl Router {
    pub fn load(model_path: &str, num_experts: usize, top_k: usize) -> Result<Self> {
        Self::load_with_family_and_mode(
            model_path,
            num_experts,
            top_k,
            None,
            RoutingMode::StubUniform,
        )
    }

    pub fn load_with_mode(
        model_path: &str,
        num_experts: usize,
        top_k: usize,
        routing_mode: RoutingMode,
    ) -> Result<Self> {
        Self::load_with_family_and_mode(model_path, num_experts, top_k, None, routing_mode)
    }

    pub fn load_with_family_and_mode(
        model_path: &str,
        num_experts: usize,
        top_k: usize,
        family_override: Option<ModelFamily>,
        routing_mode: RoutingMode,
    ) -> Result<Self> {
        if model_path.is_empty() {
            let inferred_experts = num_experts.max(1);
            let inferred_top_k = top_k.max(1).min(inferred_experts);
            return Ok(Self {
                model_path: String::new(),
                num_experts: inferred_experts,
                top_k: inferred_top_k,
                loaded: false,
                metadata: RouterMetadata::synthetic(
                    family_override.unwrap_or(ModelFamily::Olmoe),
                    inferred_experts,
                    inferred_top_k,
                ),
                adapter: None,
                routing_mode,
                expert_membranes: vec![0.0; inferred_experts],
                hidden_membranes: vec![0.0; EMBEDDING_DIM],
                threshold: 0.75,
                decay: 0.91,
                checkpoint: None,
            });
        }

        let (metadata, checkpoint, adapter) = Self::probe_and_map(model_path, family_override)?;
        let effective_num_experts = if num_experts == 0 {
            metadata.num_experts
        } else {
            num_experts
        };
        if effective_num_experts > metadata.num_experts {
            return Err(HybridError::InvalidConfig(format!(
                "num_experts ({effective_num_experts}) exceeds checkpoint expert_count ({})",
                metadata.num_experts
            )));
        }

        let effective_top_k = if top_k == 0 {
            metadata.expert_used_count.max(1).min(effective_num_experts)
        } else {
            top_k.max(1).min(effective_num_experts)
        };

        Ok(Self {
            model_path: model_path.to_owned(),
            num_experts: effective_num_experts,
            top_k: effective_top_k,
            loaded: true,
            metadata,
            adapter: Some(adapter),
            routing_mode,
            expert_membranes: vec![0.0; effective_num_experts],
            hidden_membranes: vec![0.0; EMBEDDING_DIM],
            threshold: 0.75,
            decay: 0.91,
            checkpoint: Some(checkpoint),
        })
    }

    pub fn probe_model(path: &str, family_override: Option<ModelFamily>) -> Result<RouterMetadata> {
        let (metadata, _checkpoint, _adapter) = Self::probe_and_map(path, family_override)?;
        Ok(metadata)
    }

    pub fn forward(&mut self, embedding: &[f32]) -> Result<RouterOutput> {
        if embedding.len() != EMBEDDING_DIM {
            return Err(HybridError::InputLengthMismatch {
                expected: EMBEDDING_DIM,
                got: embedding.len(),
            });
        }

        match self.routing_mode {
            RoutingMode::StubUniform => Ok(self.stub_output()),
            RoutingMode::DenseSim => self.simulate_moe_routing(embedding),
            RoutingMode::SpikingSim => self.spiking_moe_routing(embedding),
        }
    }

    pub fn extract_token_embedding(&mut self, token_id: usize) -> Result<Vec<f32>> {
        let adapter = self
            .adapter
            .as_ref()
            .ok_or_else(|| HybridError::ModelLoad {
                path: self.model_path.clone(),
                reason: "checkpoint not loaded".into(),
            })?;
        let checkpoint = self
            .checkpoint
            .as_mut()
            .ok_or_else(|| HybridError::ModelLoad {
                path: self.model_path.clone(),
                reason: "checkpoint not loaded".into(),
            })?;
        let embedding = match checkpoint {
            CheckpointBackend::Gguf(cp) => extract_named_token_embedding_from_checkpoint(
                cp,
                &adapter.token_embedding_tensor,
                &self.model_path,
                token_id,
            )?,
            CheckpointBackend::Safetensors(cp) => cp.extract_token_embedding(
                &adapter.token_embedding_tensor,
                &self.model_path,
                token_id,
            )?,
        };
        Ok(normalize_to_internal_embedding_dim(&embedding))
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn registered_gpu_synapse_weights(&mut self, tensor_name: &str) -> Result<&[u16]> {
        let checkpoint = self
            .checkpoint
            .as_mut()
            .ok_or_else(|| HybridError::ModelLoad {
                path: self.model_path.clone(),
                reason: "checkpoint not loaded".into(),
            })?;
        match checkpoint {
            CheckpointBackend::Gguf(cp) => cp.registered_f16_tensor(tensor_name, &self.model_path),
            CheckpointBackend::Safetensors(_) => Err(HybridError::UnsupportedFormat(
                "Safetensors checkpoint does not support GPU synapse registration".into(),
            )),
        }
    }

    /// Returns the tensor name to use for Q8_0 dequantized synapse loading,
    /// or `None` if the checkpoint does not have a compatible Q8_0 synapse
    /// tensor (e.g. the adapter chose F16 or synthetic fallback instead).
    pub fn dequantized_q8_0_synapse_tensor_name(&self) -> Option<&str> {
        self.adapter.as_ref().and_then(|a| {
            (a.synapse_source == SynapseSource::DequantizedQ8_0)
                .then(|| a.active_synapse_tensor_name())
                .flatten()
        })
    }

    /// Dequantize the named Q8_0 tensor to a flat `Vec<f32>` that can be
    /// passed to [`GpuAccelerator::load_synapse_weights_named`].
    #[cfg(any(feature = "cuda", test))]
    pub(crate) fn dequantized_q8_0_synapse_weights(&self, tensor_name: &str) -> Result<Vec<f32>> {
        let checkpoint = self
            .checkpoint
            .as_ref()
            .ok_or_else(|| HybridError::ModelLoad {
                path: self.model_path.clone(),
                reason: "checkpoint not loaded".into(),
            })?;
        match checkpoint {
            CheckpointBackend::Gguf(cp) => cp.dequantize_q8_0_tensor(tensor_name, &self.model_path),
            CheckpointBackend::Safetensors(_) => Err(HybridError::UnsupportedFormat(
                "Safetensors checkpoint does not support Q8_0 dequantization".into(),
            )),
        }
    }

    /// Returns the tensor name to use for Q5_K dequantized synapse loading,
    /// or `None` if the checkpoint does not have a compatible Q5_K synapse
    /// tensor (e.g. the adapter chose F16, Q8_0, or synthetic fallback instead).
    pub fn dequantized_q5_k_synapse_tensor_name(&self) -> Option<&str> {
        self.adapter.as_ref().and_then(|a| {
            (a.synapse_source == SynapseSource::DequantizedQ5K)
                .then(|| a.active_synapse_tensor_name())
                .flatten()
        })
    }

    /// Dequantize the named Q5_K tensor to a flat `Vec<f32>` that can be
    /// passed to [`GpuAccelerator::load_synapse_weights_named`].
    #[cfg(any(feature = "cuda", test))]
    pub(crate) fn dequantized_q5_k_synapse_weights(&self, tensor_name: &str) -> Result<Vec<f32>> {
        let checkpoint = self
            .checkpoint
            .as_ref()
            .ok_or_else(|| HybridError::ModelLoad {
                path: self.model_path.clone(),
                reason: "checkpoint not loaded".into(),
            })?;
        match checkpoint {
            CheckpointBackend::Gguf(cp) => cp.dequantize_q5_k_tensor(tensor_name, &self.model_path),
            CheckpointBackend::Safetensors(_) => Err(HybridError::UnsupportedFormat(
                "Safetensors checkpoint does not support Q5_K dequantization".into(),
            )),
        }
    }

    #[cfg(any(feature = "cuda", test))]
    pub(crate) fn dequantized_q6_k_synapse_tensor_name(&self) -> Option<&str> {
        self.adapter.as_ref().and_then(|a| {
            (a.synapse_source == SynapseSource::DequantizedQ6K)
                .then(|| a.active_synapse_tensor_name())
                .flatten()
        })
    }

    #[cfg(any(feature = "cuda", test))]
    pub(crate) fn dequantized_q6_k_synapse_weights(&self, tensor_name: &str) -> Result<Vec<f32>> {
        let checkpoint = self
            .checkpoint
            .as_ref()
            .ok_or_else(|| HybridError::ModelLoad {
                path: self.model_path.clone(),
                reason: "checkpoint not loaded".into(),
            })?;
        match checkpoint {
            CheckpointBackend::Gguf(cp) => cp.dequantize_q6_k_tensor(tensor_name, &self.model_path),
            CheckpointBackend::Safetensors(_) => Err(HybridError::UnsupportedFormat(
                "Safetensors checkpoint does not support Q6_K dequantization".into(),
            )),
        }
    }

    /// Tensor name for IQ3_M dequantized synapse loading, if the adapter selected that path.
    #[cfg(any(feature = "cuda", test))]
    pub(crate) fn dequantized_iq3_m_synapse_tensor_name(&self) -> Option<&str> {
        self.adapter.as_ref().and_then(|a| {
            (a.synapse_source == SynapseSource::DequantizedIQ3M)
                .then(|| a.active_synapse_tensor_name())
                .flatten()
        })
    }

    /// Dequantize the named IQ3_M tensor to a flat `Vec<f32>` for GPU synapse load.
    #[cfg(any(feature = "cuda", test))]
    pub(crate) fn dequantized_iq3_m_synapse_weights(&self, tensor_name: &str) -> Result<Vec<f32>> {
        let checkpoint = self
            .checkpoint
            .as_ref()
            .ok_or_else(|| HybridError::ModelLoad {
                path: self.model_path.clone(),
                reason: "checkpoint not loaded".into(),
            })?;
        match checkpoint {
            CheckpointBackend::Gguf(cp) => {
                cp.dequantize_iq3_m_tensor(tensor_name, &self.model_path)
            }
            CheckpointBackend::Safetensors(_) => Err(HybridError::UnsupportedFormat(
                "Safetensors checkpoint does not support IQ3_M dequantization".into(),
            )),
        }
    }

    /// Tensor name for Int4 dequantized Safetensors routing/synapse, if selected.
    #[cfg(any(feature = "cuda", test))]
    pub(crate) fn dequantized_int4_synapse_tensor_name(&self) -> Option<&str> {
        self.adapter.as_ref().and_then(|a| {
            (a.synapse_source == SynapseSource::DequantizedInt4)
                .then(|| a.active_synapse_tensor_name())
                .flatten()
        })
    }

    /// Unpack INT4/I4/U4 Safetensors routing/synapse weights to f32 for GPU load.
    #[cfg(any(feature = "cuda", test))]
    pub(crate) fn dequantized_int4_synapse_weights(&self, tensor_name: &str) -> Result<Vec<f32>> {
        let checkpoint = self
            .checkpoint
            .as_ref()
            .ok_or_else(|| HybridError::ModelLoad {
                path: self.model_path.clone(),
                reason: "checkpoint not loaded".into(),
            })?;
        match checkpoint {
            CheckpointBackend::Safetensors(cp) => {
                cp.extract_tensor_f32(tensor_name, &self.model_path)
            }
            CheckpointBackend::Gguf(_) => Err(HybridError::UnsupportedFormat(
                "GGUF checkpoint does not use Safetensors Int4 synapse path".into(),
            )),
        }
    }

    #[cfg(any(feature = "cuda", test))]
    pub(crate) fn routing_f32_synapse_tensor_name(&self) -> Option<&str> {
        self.adapter.as_ref().and_then(|a| {
            (a.synapse_source == SynapseSource::RoutingF32)
                .then(|| a.active_synapse_tensor_name())
                .flatten()
        })
    }

    #[cfg(any(feature = "cuda", test))]
    pub(crate) fn routing_f32_synapse_weights(&self, tensor_name: &str) -> Result<Vec<f32>> {
        let checkpoint = self
            .checkpoint
            .as_ref()
            .ok_or_else(|| HybridError::ModelLoad {
                path: self.model_path.clone(),
                reason: "checkpoint not loaded".into(),
            })?;
        match checkpoint {
            CheckpointBackend::Gguf(cp) => {
                Ok(cp.f32_tensor(tensor_name, &self.model_path)?.to_vec())
            }
            CheckpointBackend::Safetensors(cp) => {
                cp.extract_tensor_f32(tensor_name, &self.model_path)
            }
        }
    }

    /// `(src_rows, src_cols)` for GPU synapse resample.
    ///
    /// - **GGUF:** llama.cpp layout — `dims[0]` = contiguous columns per row,
    ///   `dims[1]` = row count (or one row if 1-D). Matches dequant helpers.
    /// - **Safetensors/HF:** C-order — `shape[0]` = rows, `shape[1]` = row
    ///   length (same as [`MappedSafetensorsCheckpoint::extract_token_embedding`]).
    #[cfg(any(feature = "cuda", test))]
    pub(crate) fn synapse_tensor_row_major_shape(
        &self,
        tensor_name: &str,
    ) -> Result<(usize, usize)> {
        let checkpoint = self
            .checkpoint
            .as_ref()
            .ok_or_else(|| HybridError::ModelLoad {
                path: self.model_path.clone(),
                reason: "checkpoint not loaded".into(),
            })?;
        match checkpoint {
            CheckpointBackend::Gguf(cp) => {
                let info = cp.tensor_info(tensor_name, &self.model_path)?;
                let dims = &info.dims;
                if dims.is_empty() {
                    return Err(HybridError::UnsupportedFormat(format!(
                        "tensor '{tensor_name}' has no dimensions"
                    )));
                }
                let src_cols = dims[0];
                let src_rows = dims.get(1).copied().unwrap_or(1);
                Ok((src_rows, src_cols))
            }
            CheckpointBackend::Safetensors(cp) => {
                let info =
                    cp.tensor_info(tensor_name)
                        .ok_or_else(|| HybridError::MissingTensor {
                            name: tensor_name.to_owned(),
                            path: self.model_path.clone(),
                        })?;
                let dims = info.1;
                if dims.is_empty() {
                    return Err(HybridError::UnsupportedFormat(format!(
                        "tensor '{tensor_name}' has no dimensions"
                    )));
                }
                let src_rows = dims[0];
                let src_cols = dims.get(1).copied().unwrap_or(1);
                Ok((src_rows, src_cols))
            }
        }
    }

    fn probe_and_map(
        path: &str,
        family_override: Option<ModelFamily>,
    ) -> Result<(RouterMetadata, CheckpointBackend, ModelAdapter)> {
        let is_safetensors = std::path::Path::new(path).is_dir()
            || !std::path::Path::new(path)
                .extension()
                .map(|ext| ext.eq_ignore_ascii_case("gguf"))
                .unwrap_or(false);

        if is_safetensors {
            let checkpoint = MappedSafetensorsCheckpoint::from_directory(path)?;
            let adapter = resolve_safetensors_adapter(
                &checkpoint.metadata,
                &checkpoint,
                family_override,
                path,
            )?;
            let metadata = RouterMetadata::from_adapter(&adapter);
            return Ok((
                metadata,
                CheckpointBackend::Safetensors(checkpoint),
                adapter,
            ));
        }

        let (_raw_metadata, checkpoint) = probe_and_map_checkpoint(path)?;
        let adapter = resolve_adapter(checkpoint.metadata(), &checkpoint, family_override, path)?;
        let metadata = RouterMetadata::from_adapter(&adapter);
        Ok((metadata, CheckpointBackend::Gguf(checkpoint), adapter))
    }

    fn simulate_moe_routing(&self, embedding: &[f32]) -> Result<RouterOutput> {
        let gate_scores = self.compute_gate_scores(embedding)?;
        let expert_weights = softmax(&gate_scores);
        let selected_experts = top_k_indices(&expert_weights, self.top_k);
        let selected_mass: f32 = selected_experts
            .iter()
            .map(|&idx| expert_weights[idx])
            .sum();
        let hidden: Vec<f32> = embedding.iter().map(|&v| v * selected_mass).collect();

        Ok(RouterOutput {
            expert_weights,
            selected_experts,
            hidden,
        })
    }

    fn spiking_moe_routing(&mut self, embedding: &[f32]) -> Result<RouterOutput> {
        let gate_scores = self.compute_gate_scores(embedding)?;
        let n = self.num_experts;
        let mut membrane_scores = Vec::with_capacity(n);
        let mut expert_spikes = vec![0.0f32; n];

        for expert_id in 0..n {
            self.expert_membranes[expert_id] =
                self.expert_membranes[expert_id] * self.decay + gate_scores[expert_id] * 0.18;

            let spike = if self.expert_membranes[expert_id] > self.threshold {
                self.expert_membranes[expert_id] -= self.threshold;
                1.0
            } else if self.expert_membranes[expert_id] < -self.threshold {
                self.expert_membranes[expert_id] += self.threshold;
                -1.0
            } else {
                0.0
            };

            expert_spikes[expert_id] = spike;
            membrane_scores.push(self.expert_membranes[expert_id] + spike * self.threshold);
        }

        let expert_weights = softmax(&membrane_scores);
        let selected_experts = top_k_indices(&expert_weights, self.top_k);
        let active_mass: f32 = selected_experts
            .iter()
            .map(|&expert_id| expert_spikes[expert_id] * expert_weights[expert_id])
            .sum();

        let mut hidden = vec![0.0f32; EMBEDDING_DIM];
        for (idx, value) in hidden.iter_mut().enumerate() {
            let input = embedding[idx] * active_mass;
            self.hidden_membranes[idx] = self.hidden_membranes[idx] * self.decay + input;
            let spike = if self.hidden_membranes[idx] > self.threshold {
                self.hidden_membranes[idx] -= self.threshold;
                1.0
            } else if self.hidden_membranes[idx] < -self.threshold {
                self.hidden_membranes[idx] += self.threshold;
                -1.0
            } else {
                0.0
            };
            *value = spike * 0.3;
        }

        Ok(RouterOutput {
            expert_weights,
            selected_experts,
            hidden,
        })
    }

    fn compute_gate_scores(&self, embedding: &[f32]) -> Result<Vec<f32>> {
        if let (Some(checkpoint), Some(adapter)) = (&self.checkpoint, &self.adapter) {
            let mut routed_embedding = resample_embedding(embedding, adapter.hidden_size);
            normalize_l2(&mut routed_embedding);
            return match checkpoint {
                CheckpointBackend::Gguf(cp) => checkpoint_gate_scores(
                    cp,
                    &self.model_path,
                    &adapter.routing_tensor,
                    self.num_experts,
                    &routed_embedding,
                ),
                CheckpointBackend::Safetensors(cp) => safetensors_gate_scores(
                    cp,
                    &self.model_path,
                    &adapter.routing_tensor,
                    self.num_experts,
                    &routed_embedding,
                ),
            };
        }

        Ok(synthetic_gate_scores(self.num_experts, embedding))
    }

    fn stub_output(&self) -> RouterOutput {
        let n = self.num_experts.max(1);
        RouterOutput {
            expert_weights: vec![1.0 / n as f32; n],
            selected_experts: (0..self.top_k.min(n)).collect(),
            hidden: vec![0.0; EMBEDDING_DIM],
        }
    }

    pub fn is_loaded(&self) -> bool {
        self.loaded
    }

    pub fn reset_state(&mut self) {
        self.expert_membranes.fill(0.0);
        self.hidden_membranes.fill(0.0);
    }

    pub fn model_path(&self) -> &str {
        &self.model_path
    }

    pub fn family(&self) -> ModelFamily {
        self.metadata.family
    }

    pub fn architecture(&self) -> &str {
        &self.metadata.architecture
    }

    pub fn quantization(&self) -> &str {
        &self.metadata.quantization
    }

    pub fn hidden_size(&self) -> usize {
        self.metadata.hidden_size
    }

    pub fn num_layers(&self) -> usize {
        self.metadata.num_layers
    }

    pub fn checkpoint_num_experts(&self) -> usize {
        self.metadata.num_experts
    }

    pub fn checkpoint_expert_used_count(&self) -> usize {
        self.metadata.expert_used_count
    }

    pub fn routing_tensor_name(&self) -> &str {
        &self.metadata.routing_tensor_name
    }

    pub fn preferred_gpu_synapse_tensor_name(&self) -> Option<&str> {
        self.metadata.preferred_gpu_synapse_tensor_name.as_deref()
    }

    pub fn real_gpu_synapse_tensor_name(&self) -> Option<&str> {
        self.adapter
            .as_ref()
            .and_then(|adapter| adapter.real_gpu_synapse_tensor.as_deref())
    }

    pub fn synapse_source(&self) -> &str {
        &self.metadata.synapse_source
    }

    /// Diagnostic descriptor for the preferred GPU synapse tensor.
    ///
    /// Returns `None` when the router has no checkpoint mapped (synthetic
    /// stub), when the adapter found no candidate tensor, or when the
    /// candidate tensor cannot be located in the mapped checkpoint. Used by
    /// `examples/synapse_diagnostic.rs` to surface the fallback reason for
    /// quantized GGUF models without dereferencing tensor payload bytes.
    pub fn preferred_gpu_synapse_tensor_descriptor(&self) -> Option<GpuSynapseTensorDescriptor> {
        let name = self.metadata.preferred_gpu_synapse_tensor_name.as_deref()?;
        let checkpoint = self.checkpoint.as_ref()?;
        match checkpoint {
            CheckpointBackend::Gguf(cp) => {
                let info = cp.tensor_info(name, &self.model_path).ok()?;
                Some(GpuSynapseTensorDescriptor {
                    name: name.to_owned(),
                    ggml_type_id: info.ggml_type,
                    ggml_type_label: ggml_type_label(info.ggml_type),
                    dims: info.dims.clone(),
                    has_dequant_path: synapse_dequant_path_supported(info.ggml_type),
                })
            }
            CheckpointBackend::Safetensors(_) => None,
        }
    }

    pub fn num_experts(&self) -> usize {
        self.num_experts
    }

    pub fn routing_mode(&self) -> RoutingMode {
        self.routing_mode
    }

    #[cfg(test)]
    pub(crate) fn has_state_activity(&self) -> bool {
        self.expert_membranes.iter().any(|&value| value != 0.0)
            || self.hidden_membranes.iter().any(|&value| value != 0.0)
    }
}

#[cfg(test)]
mod tests;
