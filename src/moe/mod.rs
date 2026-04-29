//! Public MoE router API backed by a family-aware GGUF bridge.
//!
//! Private helpers live in:
//! - `moe/checkpoint.rs` for GGUF parsing + mapped tensor access
//! - `moe/adapter.rs` for model-family detection and tensor selection
//! - `moe/routing.rs` for routing math and embedding resampling

mod adapter;
mod checkpoint;
mod routing;

use self::adapter::{resolve_adapter, ModelAdapter};
use self::checkpoint::{
    extract_named_token_embedding_from_checkpoint, probe_and_map_checkpoint, MappedGgufCheckpoint,
};
use self::routing::{
    checkpoint_gate_scores, normalize_l2, normalize_to_internal_embedding_dim, resample_embedding,
    softmax, synthetic_gate_scores, top_k_indices,
};
use crate::error::{HybridError, Result};
pub use crate::types::RoutingMode;
use crate::types::{ModelFamily, EMBEDDING_DIM};

pub(super) const GGUF_MAGIC: [u8; 4] = [b'G', b'G', b'U', b'F'];
pub(super) const GGUF_VERSION: u32 = 3;
pub(super) const GGML_TYPE_F32: u32 = 0;
pub(super) const GGML_TYPE_F16: u32 = 1;
pub(super) const GGML_TYPE_Q8_0: u32 = 8;
pub(super) const GGML_TYPE_Q5_K: u32 = 13;
pub(super) const GGML_TYPE_IQ3_S: u32 = 21;
pub(super) const GGUF_VALUE_TYPE_UINT8: u32 = 0;
pub(super) const GGUF_VALUE_TYPE_INT8: u32 = 1;
pub(super) const GGUF_VALUE_TYPE_UINT16: u32 = 2;
pub(super) const GGUF_VALUE_TYPE_INT16: u32 = 3;
pub(super) const GGUF_VALUE_TYPE_UINT32: u32 = 4;
pub(super) const GGUF_VALUE_TYPE_INT32: u32 = 5;
pub(super) const GGUF_VALUE_TYPE_FLOAT32: u32 = 6;
pub(super) const GGUF_VALUE_TYPE_BOOL: u32 = 7;
pub(super) const GGUF_VALUE_TYPE_STRING: u32 = 8;
pub(super) const GGUF_VALUE_TYPE_ARRAY: u32 = 9;
pub(super) const GGUF_VALUE_TYPE_UINT64: u32 = 10;
pub(super) const GGUF_VALUE_TYPE_INT64: u32 = 11;
pub(super) const GGUF_VALUE_TYPE_FLOAT64: u32 = 12;

/// Diagnostic snapshot of the GGUF tensor that the adapter wants to use as
/// the GPU synapse weight source for this router. Returned by
/// [`OlmoeRouter::preferred_gpu_synapse_tensor_descriptor`]; only consumed
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
    /// this `ggml_type` as GPU synapse weights. Today only `F16` qualifies
    /// (see `OlmoeRouter::registered_gpu_synapse_weights`); every other
    /// type falls back to synthetic synapses.
    pub has_dequant_path: bool,
}

/// Map a GGUF `ggml_type` u32 to a short human label. Returns `"unknown"`
/// for type ids the diagnostic doesn't recognize. The numeric ids are
/// stable in `ggml.h`, so hard-coding them here keeps the diagnostic
/// readable for the full SAAQ 1.5 lineup without adding new
/// `pub(super) const`s that aren't otherwise referenced.
pub fn ggml_type_label(ggml_type: u32) -> &'static str {
    match ggml_type {
        GGML_TYPE_F32 => "F32",
        GGML_TYPE_F16 => "F16",
        2 => "Q4_0",
        3 => "Q4_1",
        6 => "Q5_0",
        7 => "Q5_1",
        GGML_TYPE_Q8_0 => "Q8_0",
        9 => "Q8_1",
        10 => "Q2_K",
        11 => "Q3_K",
        12 => "Q4_K",
        GGML_TYPE_Q5_K => "Q5_K",
        14 => "Q6_K",
        15 => "Q8_K",
        16 => "IQ2_XXS",
        17 => "IQ2_XS",
        18 => "IQ3_XXS",
        19 => "IQ1_S",
        20 => "IQ4_NL",
        GGML_TYPE_IQ3_S => "IQ3_S",
        22 => "IQ2_S",
        23 => "IQ4_XS",
        24 => "I8",
        25 => "I16",
        26 => "I32",
        27 => "I64",
        28 => "F64",
        29 => "IQ1_M",
        30 => "BF16",
        _ => "unknown",
    }
}

/// Returns `true` iff the runtime can consume `ggml_type` as the source
/// for the GPU synapse tensor today. Mirrors the F16-only contract of
/// `OlmoeRouter::registered_gpu_synapse_weights` /
/// `MappedGgufCheckpoint::registered_f16_tensor`. Every other type falls
/// back to synthetic synapses.
pub fn synapse_dequant_path_supported(ggml_type: u32) -> bool {
    ggml_type == GGML_TYPE_F16
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
            synapse_source: "synthetic-fallback".into(),
            real_gpu_synapse_tensor_name: None,
        }
    }

    fn from_adapter(adapter: &ModelAdapter) -> Self {
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

pub struct OlmoeRouter {
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
    checkpoint: Option<MappedGgufCheckpoint>,
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
pub struct OlmoeOutput {
    pub expert_weights: Vec<f32>,
    pub selected_experts: Vec<usize>,
    pub hidden: Vec<f32>,
}

impl OlmoeRouter {
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

        let (metadata, checkpoint) = Self::probe_and_map(model_path, family_override)?;
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
        let adapter = resolve_adapter(
            checkpoint.metadata(),
            &checkpoint,
            family_override,
            model_path,
        )?;

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
        let (metadata, _checkpoint) = Self::probe_and_map(path, family_override)?;
        Ok(metadata)
    }

    pub fn forward(&mut self, embedding: &[f32]) -> Result<OlmoeOutput> {
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
        let embedding = extract_named_token_embedding_from_checkpoint(
            checkpoint,
            &adapter.token_embedding_tensor,
            &self.model_path,
            token_id,
        )?;
        Ok(normalize_to_internal_embedding_dim(&embedding))
    }

    pub(crate) fn registered_gpu_synapse_weights(&mut self, tensor_name: &str) -> Result<&[u16]> {
        let checkpoint = self
            .checkpoint
            .as_mut()
            .ok_or_else(|| HybridError::ModelLoad {
                path: self.model_path.clone(),
                reason: "checkpoint not loaded".into(),
            })?;
        checkpoint.registered_f16_tensor(tensor_name, &self.model_path)
    }

    fn probe_and_map(
        path: &str,
        family_override: Option<ModelFamily>,
    ) -> Result<(RouterMetadata, MappedGgufCheckpoint)> {
        let (_raw_metadata, checkpoint) = probe_and_map_checkpoint(path)?;
        let adapter = resolve_adapter(checkpoint.metadata(), &checkpoint, family_override, path)?;
        let metadata = RouterMetadata::from_adapter(&adapter);
        Ok((metadata, checkpoint))
    }

    fn simulate_moe_routing(&self, embedding: &[f32]) -> Result<OlmoeOutput> {
        let gate_scores = self.compute_gate_scores(embedding)?;
        let expert_weights = softmax(&gate_scores);
        let selected_experts = top_k_indices(&expert_weights, self.top_k);
        let selected_mass: f32 = selected_experts
            .iter()
            .map(|&idx| expert_weights[idx])
            .sum();
        let hidden: Vec<f32> = embedding.iter().map(|&v| v * selected_mass).collect();

        Ok(OlmoeOutput {
            expert_weights,
            selected_experts,
            hidden,
        })
    }

    fn spiking_moe_routing(&mut self, embedding: &[f32]) -> Result<OlmoeOutput> {
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

        Ok(OlmoeOutput {
            expert_weights,
            selected_experts,
            hidden,
        })
    }

    fn compute_gate_scores(&self, embedding: &[f32]) -> Result<Vec<f32>> {
        if let (Some(checkpoint), Some(adapter)) = (&self.checkpoint, &self.adapter) {
            let mut routed_embedding = resample_embedding(embedding, adapter.hidden_size);
            normalize_l2(&mut routed_embedding);
            return checkpoint_gate_scores(
                checkpoint,
                &self.model_path,
                &adapter.routing_tensor,
                self.num_experts,
                &routed_embedding,
            );
        }

        Ok(synthetic_gate_scores(self.num_experts, embedding))
    }

    fn stub_output(&self) -> OlmoeOutput {
        let n = self.num_experts.max(1);
        OlmoeOutput {
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
        let info = checkpoint.tensor_info(name, &self.model_path).ok()?;
        Some(GpuSynapseTensorDescriptor {
            name: name.to_owned(),
            ggml_type_id: info.ggml_type,
            ggml_type_label: ggml_type_label(info.ggml_type),
            dims: info.dims.clone(),
            has_dequant_path: synapse_dequant_path_supported(info.ggml_type),
        })
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
mod tests {
    use super::*;
    use std::io::Write;
    use std::path::PathBuf;

    fn write_temp_file(bytes: &[u8], label: &str) -> PathBuf {
        let path = std::env::temp_dir().join(format!(
            "corinth_canal_{label}_{}.gguf",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let mut file = std::fs::File::create(&path).unwrap();
        file.write_all(bytes).unwrap();
        path
    }

    fn push_u32(out: &mut Vec<u8>, value: u32) {
        out.extend_from_slice(&value.to_le_bytes());
    }

    fn push_u64(out: &mut Vec<u8>, value: u64) {
        out.extend_from_slice(&value.to_le_bytes());
    }

    fn push_string(out: &mut Vec<u8>, value: &str) {
        push_u64(out, value.len() as u64);
        out.extend_from_slice(value.as_bytes());
    }

    fn push_kv_u32(out: &mut Vec<u8>, key: &str, value: u32) {
        push_string(out, key);
        push_u32(out, GGUF_VALUE_TYPE_UINT32);
        push_u32(out, value);
    }

    fn push_kv_string(out: &mut Vec<u8>, key: &str, value: &str) {
        push_string(out, key);
        push_u32(out, GGUF_VALUE_TYPE_STRING);
        push_string(out, value);
    }

    fn build_test_gguf(tensors: Vec<(&str, Vec<usize>, u32, Vec<u8>)>, alignment: u32) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&GGUF_MAGIC);
        push_u32(&mut out, GGUF_VERSION);
        push_u64(&mut out, tensors.len() as u64);
        push_u64(&mut out, 7);
        push_kv_u32(&mut out, "general.alignment", alignment);
        push_kv_u32(&mut out, "general.file_type", 1);
        push_kv_string(&mut out, "general.architecture", "olmoe");
        push_kv_u32(&mut out, "olmoe.embedding_length", EMBEDDING_DIM as u32);
        push_kv_u32(&mut out, "olmoe.block_count", 16);
        push_kv_u32(&mut out, "olmoe.expert_count", 64);
        push_kv_u32(&mut out, "olmoe.expert_used_count", 8);

        let mut data_offset = 0usize;
        let mut tensor_payloads = Vec::new();
        for (name, dims, ggml_type, payload) in tensors {
            push_string(&mut out, name);
            push_u32(&mut out, dims.len() as u32);
            for dim in &dims {
                push_u64(&mut out, *dim as u64);
            }
            push_u32(&mut out, ggml_type);
            push_u64(&mut out, data_offset as u64);
            data_offset += payload.len();
            tensor_payloads.push(payload);
        }

        while out.len() % alignment as usize != 0 {
            out.push(0);
        }
        for payload in tensor_payloads {
            out.extend_from_slice(&payload);
        }

        out
    }

    fn build_real_size_checkpoint(gate_payload: Vec<u8>) -> Vec<u8> {
        let attn_q_payload = vec![0u8; EMBEDDING_DIM * EMBEDDING_DIM * 2];
        build_test_gguf(
            vec![
                (
                    "blk.0.ffn_gate_inp.weight",
                    vec![EMBEDDING_DIM, 64],
                    GGML_TYPE_F32,
                    gate_payload,
                ),
                (
                    "blk.0.attn_q.weight",
                    vec![EMBEDDING_DIM, EMBEDDING_DIM],
                    GGML_TYPE_F16,
                    attn_q_payload,
                ),
                (
                    "token_embd.weight",
                    vec![EMBEDDING_DIM, 32],
                    GGML_TYPE_F16,
                    vec![0u8; EMBEDDING_DIM * 32 * 2],
                ),
            ],
            32,
        )
    }

    fn build_quantized_synapse_checkpoint(gate_payload: Vec<u8>) -> Vec<u8> {
        build_test_gguf(
            vec![
                (
                    "blk.0.ffn_gate_inp.weight",
                    vec![EMBEDDING_DIM, 64],
                    GGML_TYPE_F32,
                    gate_payload,
                ),
                (
                    "blk.0.attn_q.weight",
                    vec![EMBEDDING_DIM, EMBEDDING_DIM],
                    GGML_TYPE_IQ3_S,
                    Vec::new(),
                ),
                (
                    "token_embd.weight",
                    vec![EMBEDDING_DIM, 32],
                    GGML_TYPE_F16,
                    vec![0u8; EMBEDDING_DIM * 32 * 2],
                ),
            ],
            32,
        )
    }

    fn stub() -> OlmoeRouter {
        OlmoeRouter::load_with_mode("", 8, 1, RoutingMode::StubUniform)
            .expect("stub load should succeed")
    }

    #[test]
    fn test_stub_mode_loads() {
        let model = stub();
        assert!(!model.is_loaded());
        assert_eq!(model.quantization(), "stub");
    }

    #[test]
    fn test_stub_forward_uniform_weights() {
        let mut model = stub();
        let out = model.forward(&vec![0.1; EMBEDDING_DIM]).unwrap();
        for weight in &out.expert_weights {
            assert!((*weight - 0.125).abs() < 1e-5);
        }
    }

    #[test]
    fn test_dense_sim_uses_real_gate_weights() {
        let mut gate = vec![0.0f32; EMBEDDING_DIM * 64];
        for (expert, value) in gate.iter_mut().take(64).enumerate() {
            *value = if expert == 0 { 8.0 } else { -8.0 };
        }
        let gate_bytes: Vec<u8> = gate.iter().flat_map(|value| value.to_le_bytes()).collect();
        let path = write_temp_file(&build_real_size_checkpoint(gate_bytes), "dense-real");

        let mut model =
            OlmoeRouter::load_with_mode(path.to_str().unwrap(), 8, 2, RoutingMode::DenseSim)
                .unwrap();
        let mut embedding = vec![0.0f32; EMBEDDING_DIM];
        embedding[0] = 1.0;
        let out = model.forward(&embedding).unwrap();
        assert_eq!(out.selected_experts[0], 0);
        assert_eq!(model.family(), ModelFamily::Olmoe);
        assert_eq!(model.routing_tensor_name(), "blk.0.ffn_gate_inp.weight");

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn test_quantized_synapse_probe_uses_synthetic_fallback() {
        let gate_payload = vec![0u8; EMBEDDING_DIM * 64 * size_of::<f32>()];
        let path = write_temp_file(
            &build_quantized_synapse_checkpoint(gate_payload),
            "iq3-s-synapse",
        );

        let metadata = OlmoeRouter::probe_model(path.to_str().unwrap(), None).unwrap();
        assert_eq!(
            metadata.preferred_gpu_synapse_tensor_name.as_deref(),
            Some("blk.0.attn_q.weight")
        );
        assert_eq!(metadata.real_gpu_synapse_tensor_name, None);
        assert_eq!(metadata.synapse_source, "synthetic-fallback");

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn test_quantized_attn_q_does_not_advertise_real_gpu_synapse_tensor() {
        let gate_payload = vec![0u8; EMBEDDING_DIM * 64 * 4];
        let attn_q_payload = vec![0u8; 16];

        let checkpoint = build_test_gguf(
            vec![
                (
                    "blk.0.ffn_gate_inp.weight",
                    vec![EMBEDDING_DIM, 64],
                    GGML_TYPE_F32,
                    gate_payload,
                ),
                (
                    "blk.0.attn_q.weight",
                    vec![EMBEDDING_DIM, EMBEDDING_DIM],
                    GGML_TYPE_IQ3_S,
                    attn_q_payload,
                ),
                (
                    "token_embd.weight",
                    vec![EMBEDDING_DIM, 32],
                    GGML_TYPE_F16,
                    vec![0u8; EMBEDDING_DIM * 32 * 2],
                ),
            ],
            32,
        );

        let path = write_temp_file(&checkpoint, "quantized-attn-q");
        let metadata = OlmoeRouter::probe_model(path.to_str().unwrap(), None).unwrap();

        assert_eq!(
            metadata.preferred_gpu_synapse_tensor_name.as_deref(),
            Some("blk.0.attn_q.weight")
        );
        assert_eq!(metadata.real_gpu_synapse_tensor_name, None);
        assert_eq!(metadata.synapse_source, "synthetic-fallback");

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn test_preferred_synapse_descriptor_iq3s_has_no_dequant_path() {
        let gate_payload = vec![0u8; EMBEDDING_DIM * 64 * size_of::<f32>()];
        let path = write_temp_file(
            &build_quantized_synapse_checkpoint(gate_payload),
            "iq3-s-descriptor",
        );

        let model =
            OlmoeRouter::load_with_mode(path.to_str().unwrap(), 0, 0, RoutingMode::StubUniform)
                .unwrap();
        let descriptor = model
            .preferred_gpu_synapse_tensor_descriptor()
            .expect("preferred descriptor must be exposed for quantized attn_q");

        assert_eq!(descriptor.name, "blk.0.attn_q.weight");
        assert_eq!(descriptor.ggml_type_id, GGML_TYPE_IQ3_S);
        assert_eq!(descriptor.ggml_type_label, "IQ3_S");
        assert_eq!(descriptor.dims, vec![EMBEDDING_DIM, EMBEDDING_DIM]);
        assert!(!descriptor.has_dequant_path);
        assert_eq!(model.real_gpu_synapse_tensor_name(), None);
        assert_eq!(model.synapse_source(), "synthetic-fallback");

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn test_preferred_synapse_descriptor_f16_has_dequant_path() {
        let gate_payload = vec![0u8; EMBEDDING_DIM * 64 * size_of::<f32>()];
        let path = write_temp_file(&build_real_size_checkpoint(gate_payload), "f16-descriptor");

        let model =
            OlmoeRouter::load_with_mode(path.to_str().unwrap(), 0, 0, RoutingMode::StubUniform)
                .unwrap();
        let descriptor = model
            .preferred_gpu_synapse_tensor_descriptor()
            .expect("preferred descriptor must be exposed for F16 attn_q");

        assert_eq!(descriptor.name, "blk.0.attn_q.weight");
        assert_eq!(descriptor.ggml_type_id, GGML_TYPE_F16);
        assert_eq!(descriptor.ggml_type_label, "F16");
        assert_eq!(descriptor.dims, vec![EMBEDDING_DIM, EMBEDDING_DIM]);
        assert!(descriptor.has_dequant_path);
        assert_eq!(
            model.real_gpu_synapse_tensor_name(),
            Some("blk.0.attn_q.weight")
        );
        assert_eq!(model.synapse_source(), "real");

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn test_ggml_type_label_covers_lineup_quants() {
        // Sanity: the labels we surface in synapse_diagnostic.json should
        // never read "unknown" for the SAAQ 1.5 lineup's known quant types.
        for &(ty, expected) in &[
            (0u32, "F32"),
            (1u32, "F16"),
            (8u32, "Q8_0"),
            (12u32, "Q4_K"),
            (13u32, "Q5_K"),
            (14u32, "Q6_K"),
            (20u32, "IQ4_NL"),
            (21u32, "IQ3_S"),
        ] {
            assert_eq!(ggml_type_label(ty), expected, "ggml_type={ty}");
        }
        assert_eq!(ggml_type_label(9999), "unknown");
        assert!(synapse_dequant_path_supported(GGML_TYPE_F16));
        for &ty in &[0u32, 8, 12, 13, 14, 20, 21] {
            assert!(!synapse_dequant_path_supported(ty), "ggml_type={ty}");
        }
    }

    #[test]
    fn test_spiking_sim_state_can_reset() {
        let mut model = OlmoeRouter::load_with_mode("", 8, 2, RoutingMode::SpikingSim).unwrap();
        let _ = model.forward(&vec![1.0; EMBEDDING_DIM]).unwrap();
        assert!(model.has_state_activity());
        model.reset_state();
        assert!(!model.has_state_activity());
    }

    #[test]
    fn test_real_checkpoint_probe_via_env() {
        let Some(path) = std::env::var("GGUF_CHECKPOINT_PATH").ok() else {
            return;
        };

        let metadata = OlmoeRouter::probe_model(&path, None).unwrap();
        assert!(!metadata.architecture.is_empty());
        assert!(metadata.hidden_size > 0);
        assert!(metadata.num_experts > 0);
        assert!(!metadata.routing_tensor_name.is_empty());
    }
}
