// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Model-family adapter resolution for the GGUF and Safetensors router host.

use super::checkpoint::{GgufMetadata, GgufTensorInfo, MappedGgufCheckpoint};
use super::safetensors::{MappedSafetensorsCheckpoint, SafetensorsMetadata};
use super::{
    GGML_TYPE_F16, GGML_TYPE_F32, GGML_TYPE_IQ3_M_BLOCK, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K,
    GGML_TYPE_Q8_0,
};
use crate::error::{HybridError, Result};
use crate::types::ModelFamily;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum SynapseSource {
    Real,
    DequantizedQ8_0,
    DequantizedQ5K,
    DequantizedQ6K,
    DequantizedIQ3M,
    RoutingF32,
    DequantizedInt4,
    SyntheticFallback,
}

#[derive(Debug, Clone)]
pub(super) struct ModelAdapter {
    pub(super) family: ModelFamily,
    pub(super) architecture: String,
    pub(super) hidden_size: usize,
    pub(super) num_layers: usize,
    pub(super) num_experts: usize,
    pub(super) expert_used_count: usize,
    pub(super) token_embedding_tensor: String,
    pub(super) routing_tensor: String,
    pub(super) preferred_gpu_synapse_tensor: Option<String>,
    pub(super) real_gpu_synapse_tensor: Option<String>,
    pub(super) dequant_q8_0_synapse_tensor: Option<String>,
    pub(super) dequant_q5_k_synapse_tensor: Option<String>,
    pub(super) dequant_q6_k_synapse_tensor: Option<String>,
    pub(super) dequant_iq3_m_synapse_tensor: Option<String>,
    pub(super) routing_f32_synapse_tensor: Option<String>,
    pub(super) dequant_int4_synapse_tensor: Option<String>,
    pub(super) synapse_source: SynapseSource,
    pub(super) quantization: String,
}

/// The `synapse_source` string used when no checkpoint synapse tensor is in
/// play. Defined once here because it is a value `run_manifest.json` persists
/// and CLAUDE.md pins; it was previously spelled out separately in
/// `RouterMetadata::synthetic` and again in the temporal fallback.
pub(crate) const SYNTHETIC_FALLBACK_SOURCE: &str = "synthetic-fallback";

impl ModelAdapter {
    pub(super) fn synapse_source_label(&self) -> &'static str {
        match self.synapse_source {
            SynapseSource::Real => "real",
            SynapseSource::DequantizedQ8_0 => "dequantized-q8_0",
            SynapseSource::DequantizedQ5K => "dequantized-q5_k",
            SynapseSource::DequantizedQ6K => "dequantized-q6_k",
            SynapseSource::DequantizedIQ3M => "dequantized-iq3_m",
            SynapseSource::RoutingF32 => "routing-f32",
            SynapseSource::DequantizedInt4 => "dequantized-int4",
            SynapseSource::SyntheticFallback => SYNTHETIC_FALLBACK_SOURCE,
        }
    }

    /// Tensor name selected for the active synapse source (reads every dequant
    /// field arm so none are dead storage).
    pub(super) fn active_synapse_tensor_name(&self) -> Option<&str> {
        match self.synapse_source {
            SynapseSource::Real => self.real_gpu_synapse_tensor.as_deref(),
            SynapseSource::DequantizedQ8_0 => self.dequant_q8_0_synapse_tensor.as_deref(),
            SynapseSource::DequantizedQ5K => self.dequant_q5_k_synapse_tensor.as_deref(),
            SynapseSource::DequantizedQ6K => self.dequant_q6_k_synapse_tensor.as_deref(),
            SynapseSource::DequantizedIQ3M => self.dequant_iq3_m_synapse_tensor.as_deref(),
            SynapseSource::RoutingF32 => self.routing_f32_synapse_tensor.as_deref(),
            SynapseSource::DequantizedInt4 => self.dequant_int4_synapse_tensor.as_deref(),
            SynapseSource::SyntheticFallback => self.preferred_gpu_synapse_tensor.as_deref(),
        }
    }
}

fn is_token_embedding_supported(info: &GgufTensorInfo, hidden_size: usize) -> bool {
    info.dims.len() == 2
        && info.dims[0] == hidden_size
        && info.dims[1] > 0
        && match info.ggml_type {
            GGML_TYPE_F32 | GGML_TYPE_F16 => true,
            GGML_TYPE_Q8_0 => info.dims[0].is_multiple_of(32),
            GGML_TYPE_Q5_K | GGML_TYPE_Q6_K => info.dims[0].is_multiple_of(256),
            _ => false,
        }
}

fn resolve_token_embedding_tensor(
    checkpoint: &MappedGgufCheckpoint,
    hidden_size: usize,
    path: &str,
) -> Result<String> {
    if checkpoint.has_tensor("token_embd.weight") {
        let info = checkpoint.tensor_info("token_embd.weight", path)?;
        if is_token_embedding_supported(info, hidden_size) {
            return Ok("token_embd.weight".to_owned());
        }
        return Err(HybridError::UnsupportedFormat(format!(
            "token embedding tensor 'token_embd.weight' in '{path}' has unsupported shape/type: ggml_type={} dims={:?} (expected [hidden_size={hidden_size}, vocab_size>0])",
            info.ggml_type, info.dims
        )));
    }
    if checkpoint.has_tensor("tok_embeddings.weight") {
        let info = checkpoint.tensor_info("tok_embeddings.weight", path)?;
        if is_token_embedding_supported(info, hidden_size) {
            return Ok("tok_embeddings.weight".to_owned());
        }
        return Err(HybridError::UnsupportedFormat(format!(
            "token embedding tensor 'tok_embeddings.weight' in '{path}' has unsupported shape/type: ggml_type={} dims={:?} (expected [hidden_size={hidden_size}, vocab_size>0])",
            info.ggml_type, info.dims
        )));
    }
    Err(HybridError::MissingTensor {
        name: "token_embd.weight".into(),
        path: path.to_owned(),
    })
}

/// Resolve the GGUF gate tensor name and reject shapes the runtime cannot index.
///
/// Gate scoring (`routing_weight_index`) requires one dimension to equal
/// `hidden_size` (metadata `embedding_length`, matching the resampled embedding
/// length in `Router::compute_gate_scores`) and the other to be ≥ `num_experts`.
/// Do **not** relax this to `min(d0, d1) >= num_experts`: that previously
/// accepted tensors that then failed on the first gate-score pass.
fn resolve_routing_tensor(
    checkpoint: &MappedGgufCheckpoint,
    hidden_size: usize,
    num_experts: usize,
    path: &str,
) -> Result<String> {
    let routing_tensor = checkpoint
        .find_first_tensor_with_suffix("ffn_gate_inp.weight")
        .or_else(|| checkpoint.find_first_tensor_with_suffix("ffn_gate.weight"))
        .ok_or_else(|| HybridError::MissingTensor {
            name: "ffn_gate_inp.weight".into(),
            path: path.to_owned(),
        })?
        .to_owned();
    let routing_info = checkpoint.tensor_info(&routing_tensor, path)?;
    if routing_info.ggml_type != GGML_TYPE_F32 || routing_info.dims.len() != 2 {
        return Err(HybridError::UnsupportedFormat(format!(
            "routing tensor '{routing_tensor}' must be rank-2 F32 in '{path}', got dims={:?} ggml_type={}",
            routing_info.dims, routing_info.ggml_type
        )));
    }
    let (d0, d1) = (routing_info.dims[0], routing_info.dims[1]);
    // Same orientation contract as `routing_weight_index` (routing.rs).
    let has_hidden_size_dim = d0 == hidden_size || d1 == hidden_size;
    if !has_hidden_size_dim {
        return Err(HybridError::UnsupportedFormat(format!(
            "routing tensor '{routing_tensor}' in '{path}' has unsupported orientation dims={d0}x{d1}; expected one dimension to equal hidden_size={hidden_size} and the other to be at least {num_experts}"
        )));
    }
    let routing_experts = if d0 == hidden_size { d1 } else { d0 };
    if routing_experts < num_experts {
        return Err(HybridError::UnsupportedFormat(format!(
            "routing tensor '{routing_tensor}' in '{path}' only exposes {routing_experts} experts, expected at least {num_experts}"
        )));
    }
    Ok(routing_tensor)
}

struct GgufTopology {
    hidden_size: usize,
    num_layers: usize,
    num_experts: usize,
    expert_used_count: usize,
}

fn resolve_gguf_topology(
    metadata: &GgufMetadata,
    architecture: &str,
    path: &str,
) -> Result<GgufTopology> {
    let hidden_size = metadata
        .numeric(&format!("{architecture}.embedding_length"))
        .ok_or_else(|| {
            HybridError::UnsupportedFormat(format!(
                "missing '{architecture}.embedding_length' in '{path}'"
            ))
        })?;
    let num_layers = metadata
        .numeric(&format!("{architecture}.block_count"))
        .ok_or_else(|| {
            HybridError::UnsupportedFormat(format!(
                "missing '{architecture}.block_count' in '{path}'"
            ))
        })?;
    let num_experts = metadata
        .numeric(&format!("{architecture}.expert_count"))
        .ok_or_else(|| {
            HybridError::UnsupportedFormat(format!(
                "missing '{architecture}.expert_count' in '{path}'"
            ))
        })?;
    let expert_used_count = metadata
        .numeric(&format!("{architecture}.expert_used_count"))
        .unwrap_or(1);
    Ok(GgufTopology {
        hidden_size,
        num_layers,
        num_experts,
        expert_used_count,
    })
}

pub(super) fn resolve_adapter(
    metadata: &GgufMetadata,
    checkpoint: &MappedGgufCheckpoint,
    family_override: Option<ModelFamily>,
    path: &str,
) -> Result<ModelAdapter> {
    let architecture = metadata.architecture().to_owned();
    let family = infer_family(&architecture, family_override, path)?;
    let topology = resolve_gguf_topology(metadata, &architecture, path)?;
    let token_embedding_tensor =
        resolve_token_embedding_tensor(checkpoint, topology.hidden_size, path)?;

    let routing_tensor =
        resolve_routing_tensor(checkpoint, topology.hidden_size, topology.num_experts, path)?;

    let preferred_gpu_synapse_tensor = checkpoint
        .has_tensor("blk.0.attn_q.weight")
        .then(|| "blk.0.attn_q.weight".to_owned());
    let selection = select_gguf_synapse(
        preferred_gpu_synapse_tensor.as_deref(),
        &routing_tensor,
        checkpoint,
        topology.hidden_size,
        path,
    );

    Ok(ModelAdapter {
        family,
        architecture,
        hidden_size: topology.hidden_size,
        num_layers: topology.num_layers,
        num_experts: topology.num_experts,
        expert_used_count: topology.expert_used_count,
        token_embedding_tensor,
        routing_tensor,
        preferred_gpu_synapse_tensor,
        synapse_source: selection.synapse_source,
        real_gpu_synapse_tensor: selection.real_gpu_synapse_tensor,
        dequant_q8_0_synapse_tensor: selection.dequant_q8_0_synapse_tensor,
        dequant_q5_k_synapse_tensor: selection.dequant_q5_k_synapse_tensor,
        dequant_q6_k_synapse_tensor: selection.dequant_q6_k_synapse_tensor,
        dequant_iq3_m_synapse_tensor: selection.dequant_iq3_m_synapse_tensor,
        routing_f32_synapse_tensor: selection.routing_f32_synapse_tensor,
        dequant_int4_synapse_tensor: None,
        quantization: metadata.quantization().to_owned(),
    })
}

/// Family-specific Safetensors routing/gate tensor names, in probe order.
///
/// Layouts: GPT-OSS router, DeepSeek dense prefix, MiniMax block_sparse_moe,
/// Gemma4 language_model router, Step moe.gate (after dense prefix), Granite
/// block_sparse_moe.router.layer, Qwen/Cohere language_model gates, generic.
fn safetensors_routing_candidates(family: ModelFamily) -> &'static [&'static str] {
    match family {
        ModelFamily::GptOss => &[
            "model.layers.0.mlp.router.weight",
            "model.layers.0.mlp.gate.weight",
        ],
        // DeepSeek-V3 / Moonlight: first_k_dense_replace often makes layer 0 dense.
        ModelFamily::Moonlight16BA3B | ModelFamily::DeepSeek2 => &[
            "model.layers.1.mlp.gate.weight",
            "model.layers.0.mlp.gate.weight",
            "model.layers.0.mlp.router.weight",
            "model.layers.1.block_sparse_moe.gate.weight",
            "model.layers.0.block_sparse_moe.gate.weight",
        ],
        // MiniMax-M2/M2.7: block_sparse_moe.gate.weight
        ModelFamily::MiniMax => &[
            "model.layers.0.block_sparse_moe.gate.weight",
            "model.layers.1.block_sparse_moe.gate.weight",
            "model.layers.0.mlp.gate.weight",
            "model.layers.0.mlp.router.weight",
        ],
        // google/gemma-4-*: language_model.layers.*.router.proj.weight
        ModelFamily::Gemma4 => &[
            "model.language_model.layers.0.router.proj.weight",
            "model.language_model.layers.1.router.proj.weight",
            "model.layers.0.mlp.gate.weight",
            "model.layers.0.mlp.router.weight",
        ],
        // Qwen3.6 multimodal MoE: language_model.layers.*.mlp.gate.weight
        ModelFamily::Qwen3Moe => &[
            "model.language_model.layers.0.mlp.gate.weight",
            "model.language_model.layers.1.mlp.gate.weight",
            "model.layers.0.mlp.gate.weight",
            "model.layers.0.mlp.router.weight",
        ],
        // Command A+ / Cohere2Vision: language_model MoE gates
        ModelFamily::Cohere => &[
            "model.language_model.layers.0.mlp.gate.weight",
            "model.language_model.layers.1.mlp.gate.weight",
            "model.layers.0.mlp.gate.weight",
            "model.layers.0.mlp.router.weight",
        ],
        // Step-3.5-Flash: moe_layers_enum often starts at layer 3 (dense 0..2).
        ModelFamily::Step => &[
            "model.layers.3.moe.gate.weight",
            "model.layers.0.moe.gate.weight",
            "model.layers.1.moe.gate.weight",
            "model.layers.2.moe.gate.weight",
            "model.layers.0.mlp.gate.weight",
            "model.layers.0.mlp.router.weight",
        ],
        // ibm-granite A800M: block_sparse_moe.router.layer.weight
        ModelFamily::Granite31A800M => &[
            "model.layers.0.block_sparse_moe.router.layer.weight",
            "model.layers.0.block_sparse_moe.gate.weight",
            "model.layers.0.moe.gate.weight",
            "model.layers.0.mlp.gate.weight",
            "model.layers.0.mlp.router.weight",
        ],
        // microsoft/Phi-tiny-MoE (SlimMoE): block_sparse_moe.gate.weight
        ModelFamily::SlimMoe => &[
            "model.layers.0.block_sparse_moe.gate.weight",
            "model.layers.0.moe.gate.weight",
            "model.layers.0.mlp.gate.weight",
            "model.layers.0.mlp.router.weight",
        ],
        // LiquidAI LFM2-MoE: feed_forward.gate (not mlp.gate); MoE layers often mid-stack.
        ModelFamily::Lfm2Moe => &[
            "model.layers.0.feed_forward.gate.weight",
            "model.layers.1.feed_forward.gate.weight",
            "model.layers.10.feed_forward.gate.weight",
            "model.layers.0.mlp.gate.weight",
            "model.layers.0.mlp.router.weight",
        ],
        // microsoft/GRIN-MoE: first block_sparse router is layer 1 (no layer-0 gate).
        ModelFamily::Grin => &[
            "model.layers.1.block_sparse_moe.gate.weight",
            "model.layers.0.block_sparse_moe.gate.weight",
            "model.layers.0.mlp.gate.weight",
            "model.layers.0.mlp.router.weight",
        ],
        _ => &[
            "model.layers.0.mlp.gate.weight",
            "model.layers.0.mlp.router.weight",
            "model.layers.1.mlp.gate.weight",
            "model.layers.0.block_sparse_moe.gate.weight",
            "model.layers.1.block_sparse_moe.gate.weight",
            "model.layers.0.moe.gate.weight",
            "model.language_model.layers.0.mlp.gate.weight",
            "model.layers.0.feed_forward.gate.weight",
        ],
    }
}

/// Pick the first existing routing/gate tensor for a Safetensors MoE pack.
fn resolve_safetensors_routing_tensor(
    family: ModelFamily,
    checkpoint: &MappedSafetensorsCheckpoint,
    path: &str,
) -> Result<String> {
    for name in safetensors_routing_candidates(family) {
        if checkpoint.tensor_info(name).is_some() {
            return Ok((*name).to_owned());
        }
    }
    Err(HybridError::MissingTensor {
        name: format!(
            "routing tensor for {family:?} (tried family-specific gate/router candidates)"
        ),
        path: path.to_owned(),
    })
}

/// Embedding tensor name for Safetensors packs (multimodal uses language_model prefix).
fn resolve_safetensors_token_embedding(
    family: ModelFamily,
    checkpoint: &MappedSafetensorsCheckpoint,
    path: &str,
) -> Result<String> {
    let mut candidates: Vec<&str> = Vec::new();
    match family {
        // Multimodal MoE packs nest the LM under language_model.
        ModelFamily::Gemma4 | ModelFamily::Qwen3Moe | ModelFamily::Cohere => {
            candidates.push("model.language_model.embed_tokens.weight");
            candidates.push("model.embed_tokens.weight");
        }
        _ => {
            candidates.push("model.embed_tokens.weight");
            candidates.push("model.language_model.embed_tokens.weight");
        }
    }
    for name in candidates {
        if checkpoint.tensor_info(name).is_some() {
            return Ok(name.to_owned());
        }
    }
    Err(HybridError::MissingTensor {
        name: "model.embed_tokens.weight".to_owned(),
        path: path.to_owned(),
    })
}

/// Priority-ordered GGUF synapse source selection.
///
/// First matching candidate wins; all lower-priority Option fields stay `None`.
struct SynapseSelection {
    synapse_source: SynapseSource,
    real_gpu_synapse_tensor: Option<String>,
    dequant_q8_0_synapse_tensor: Option<String>,
    dequant_q5_k_synapse_tensor: Option<String>,
    dequant_q6_k_synapse_tensor: Option<String>,
    dequant_iq3_m_synapse_tensor: Option<String>,
    routing_f32_synapse_tensor: Option<String>,
}

impl Default for SynapseSelection {
    fn default() -> Self {
        Self {
            synapse_source: SynapseSource::RoutingF32,
            real_gpu_synapse_tensor: None,
            dequant_q8_0_synapse_tensor: None,
            dequant_q5_k_synapse_tensor: None,
            dequant_q6_k_synapse_tensor: None,
            dequant_iq3_m_synapse_tensor: None,
            routing_f32_synapse_tensor: None,
        }
    }
}

fn select_real_synapse(
    name: &str,
    info: &GgufTensorInfo,
    hidden_size: usize,
) -> Option<SynapseSelection> {
    (info.ggml_type == GGML_TYPE_F16 && info.dims == [hidden_size, hidden_size]).then(|| {
        SynapseSelection {
            synapse_source: SynapseSource::Real,
            real_gpu_synapse_tensor: Some(name.to_owned()),
            ..Default::default()
        }
    })
}

fn select_quantized_synapse(name: &str, info: &GgufTensorInfo) -> Option<SynapseSelection> {
    if info.dims.len() != 2 {
        return None;
    }
    let d0 = info.dims[0];
    match info.ggml_type {
        GGML_TYPE_Q8_0 if d0.is_multiple_of(32) => Some(SynapseSelection {
            synapse_source: SynapseSource::DequantizedQ8_0,
            dequant_q8_0_synapse_tensor: Some(name.to_owned()),
            ..Default::default()
        }),
        GGML_TYPE_Q5_K if d0.is_multiple_of(256) => Some(SynapseSelection {
            synapse_source: SynapseSource::DequantizedQ5K,
            dequant_q5_k_synapse_tensor: Some(name.to_owned()),
            ..Default::default()
        }),
        GGML_TYPE_Q6_K if d0.is_multiple_of(256) => Some(SynapseSelection {
            synapse_source: SynapseSource::DequantizedQ6K,
            dequant_q6_k_synapse_tensor: Some(name.to_owned()),
            ..Default::default()
        }),
        // IQ3_M *block* dequant is only for the internal GGML_TYPE_IQ3_M_BLOCK id
        // (synthetic fixtures). Wire type 31 is Q4_0_4_4 in the GGUF enum — never
        // select it here (Codex). HF “IQ3_M” GGUFs are mixed-type presets.
        GGML_TYPE_IQ3_M_BLOCK if d0.is_multiple_of(256) => Some(SynapseSelection {
            synapse_source: SynapseSource::DequantizedIQ3M,
            dequant_iq3_m_synapse_tensor: Some(name.to_owned()),
            ..Default::default()
        }),
        _ => None,
    }
}

fn select_preferred_synapse(
    name: &str,
    info: &GgufTensorInfo,
    hidden_size: usize,
) -> Option<SynapseSelection> {
    select_real_synapse(name, info, hidden_size).or_else(|| select_quantized_synapse(name, info))
}

fn select_gguf_synapse(
    preferred: Option<&str>,
    routing_tensor: &str,
    checkpoint: &MappedGgufCheckpoint,
    hidden_size: usize,
    path: &str,
) -> SynapseSelection {
    if let Some(name) = preferred
        && let Ok(info) = checkpoint.tensor_info(name, path)
        && let Some(selection) = select_preferred_synapse(name, info, hidden_size)
    {
        return selection;
    }

    // RoutingF32 fallback: reached when no preferred tensor is present or none matched a dequant path.
    SynapseSelection {
        synapse_source: SynapseSource::RoutingF32,
        routing_f32_synapse_tensor: Some(routing_tensor.to_owned()),
        ..Default::default()
    }
}

pub(super) fn resolve_safetensors_adapter(
    metadata: &SafetensorsMetadata,
    checkpoint: &MappedSafetensorsCheckpoint,
    family_override: Option<ModelFamily>,
    path: &str,
) -> Result<ModelAdapter> {
    let architecture = metadata.architecture.clone();
    let family = infer_family_safetensors(&architecture, family_override, path)?;
    let hidden_size = metadata.hidden_size;
    let num_layers = metadata.num_layers;
    let num_experts = metadata.num_experts;
    let expert_used_count = metadata.expert_used_count;

    let token_embedding_tensor = resolve_safetensors_token_embedding(family, checkpoint, path)?;
    let routing_tensor = resolve_safetensors_routing_tensor(family, checkpoint, path)?;

    // Validate routing tensor is rank-2 and exposes enough experts.
    let routing_info =
        checkpoint
            .tensor_info(&routing_tensor)
            .ok_or_else(|| HybridError::MissingTensor {
                name: routing_tensor.clone(),
                path: path.to_owned(),
            })?;
    if routing_info.1.len() != 2 {
        return Err(HybridError::UnsupportedFormat(format!(
            "routing tensor '{routing_tensor}' must be rank-2 in '{path}', got dims={:?}",
            routing_info.1
        )));
    }
    let routing_experts = routing_info.1[0].min(routing_info.1[1]);
    if routing_experts < num_experts {
        return Err(HybridError::UnsupportedFormat(format!(
            "routing tensor '{routing_tensor}' in '{path}' only exposes {routing_experts} experts, expected at least {num_experts}"
        )));
    }

    // Safetensors path: use the resolved routing tensor as preferred synapse;
    // detect Int4 quantized weights. Note: float dtypes (F16/BF16/F32)
    // remain synthetic fallback because the GPU synapse loader only
    // supports GGUF-registered tensors today.
    let preferred_gpu_synapse_tensor = Some(routing_tensor.clone());
    let real_gpu_synapse_tensor = None;
    let dequant_int4_synapse_tensor = preferred_gpu_synapse_tensor.as_ref().and_then(|name| {
        let info = checkpoint.tensor_info(name)?;
        matches!(info.0, "INT4" | "I4" | "U4").then(|| name.clone())
    });
    let synapse_source = if dequant_int4_synapse_tensor.is_some() {
        SynapseSource::DequantizedInt4
    } else {
        SynapseSource::SyntheticFallback
    };

    Ok(ModelAdapter {
        family,
        architecture,
        hidden_size,
        num_layers,
        num_experts,
        expert_used_count,
        token_embedding_tensor,
        routing_tensor,
        preferred_gpu_synapse_tensor,
        synapse_source,
        real_gpu_synapse_tensor,
        dequant_q8_0_synapse_tensor: None,
        dequant_q5_k_synapse_tensor: None,
        dequant_q6_k_synapse_tensor: None,
        dequant_iq3_m_synapse_tensor: None,
        routing_f32_synapse_tensor: None,
        dequant_int4_synapse_tensor,
        quantization: "safetensors".into(),
    })
}

fn check_family_compatibility(expected: ModelFamily, inferred: ModelFamily) -> bool {
    expected == inferred
        || matches!(
            (expected, inferred),
            (ModelFamily::Moonlight16BA3B, ModelFamily::DeepSeek2)
                | (ModelFamily::DeepSeek2, ModelFamily::Moonlight16BA3B)
        )
}

/// Discriminator for architecture-string tables (GGUF vs Safetensors/HF names).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ArchFormat {
    Gguf,
    Safetensors,
}

impl ArchFormat {
    fn label(self) -> &'static str {
        match self {
            Self::Gguf => "GGUF",
            Self::Safetensors => "Safetensors",
        }
    }
}

/// Per-family architecture aliases for GGUF `general.architecture` and
/// Safetensors/HF `architectures[]` class names.
///
/// Adding a family (or an alias) is a single table row — no second match arm.
/// Empty `safetensors` means that family is GGUF-only for inference today
/// (override + Moonlight/DeepSeek2 compatibility still apply separately).
struct FamilyArchNames {
    family: ModelFamily,
    gguf: &'static [&'static str],
    safetensors: &'static [&'static str],
}

/// Single source of truth for architecture → [`ModelFamily`] maps.
///
/// Order is not significant for lookup; keep rows aligned with
/// [`ModelFamily`] variant order for reviewability. `NemotronLegacy` is a
/// serde alias of Nemotron and is not listed (same arch strings as Nemotron).
const FAMILY_ARCHES: &[FamilyArchNames] = &[
    FamilyArchNames {
        family: ModelFamily::Olmoe,
        gguf: &["olmoe"],
        // allenai/Emo_1b14b_1T (emo_1b14b_cloud) ships EmoForCausalLM.
        safetensors: &["OlmoeForCausalLM", "EmoForCausalLM"],
    },
    FamilyArchNames {
        family: ModelFamily::Qwen3Moe,
        gguf: &["qwen3moe"],
        // Qwen3.6-35B-A3B multimodal MoE ships Qwen3_5MoeForConditionalGeneration.
        safetensors: &["Qwen3MoeForCausalLM", "Qwen3_5MoeForConditionalGeneration"],
    },
    FamilyArchNames {
        family: ModelFamily::Gemma4,
        gguf: &["gemma4"],
        // Shipped google/gemma-4-* packages use ConditionalGeneration + text_config.
        safetensors: &["Gemma4ForCausalLM", "Gemma4ForConditionalGeneration"],
    },
    FamilyArchNames {
        family: ModelFamily::DeepSeek2,
        gguf: &["deepseek2"],
        // DeepseekV4ForCausalLM left unmapped: V4 has num_hash_layers / hash_moe
        // bootstrap gates; probing layers.0/1 as top-k MoE silently misroutes.
        // Re-add only with hash-aware layer selection (not on this PR).
        safetensors: &["DeepseekV2ForCausalLM"],
    },
    FamilyArchNames {
        family: ModelFamily::LlamaMoe,
        gguf: &["llama"],
        // Llama 4 Scout / multimodal MoE packs use Llama4ForConditionalGeneration.
        safetensors: &["LlamaMoeForCausalLM", "Llama4ForConditionalGeneration"],
    },
    FamilyArchNames {
        family: ModelFamily::Moonlight16BA3B,
        // Kimi-VL-A3B GGUF packages are classified under the Moonlight-16B family
        // (Moonshot AI lineage), not Qwen3/DeepSeek2.
        gguf: &["moonlight", "kimi", "kimi_vl_a3b", "kimi_vl_a3b_q6_k"],
        // DeepseekV3 is the HF architecture tag used by Moonlight-16B-A3B ST packs.
        safetensors: &["MoonlightForCausalLM", "DeepseekV3ForCausalLM"],
    },
    FamilyArchNames {
        family: ModelFamily::Granite31A800M,
        gguf: &["granite", "granitemoe"],
        // Dense GraniteForCausalLM intentionally omitted (MoE A800M only).
        safetensors: &["GraniteMoeForCausalLM"],
    },
    FamilyArchNames {
        family: ModelFamily::Nemotron,
        gguf: &["nemotronh"],
        safetensors: &["NemotronHForCausalLM"],
    },
    FamilyArchNames {
        family: ModelFamily::Lfm2Moe,
        gguf: &["lfm2", "lfm2moe"],
        safetensors: &["Lfm2MoeForCausalLM"],
    },
    FamilyArchNames {
        family: ModelFamily::SlimMoe,
        gguf: &["phimoe", "slimmoe"],
        safetensors: &["PhiMoEForCausalLM"],
    },
    FamilyArchNames {
        family: ModelFamily::Zaya,
        gguf: &["zaya"],
        // ST left unsupported: official ZAYA1 uses an MLP router, not
        // model.layers.0.mlp.gate.weight (Codex P2). GGUF path remains.
        safetensors: &[],
    },
    FamilyArchNames {
        family: ModelFamily::Glm4,
        gguf: &["glm4", "glm4moe"],
        // Dense Glm4ForCausalLM omitted (one-expert false positive). MoE-Lite
        // is the shipped tag for glm47_flash_cloud (zai-org/GLM-4.7-Flash).
        safetensors: &["Glm4MoeForCausalLM", "Glm4MoeLiteForCausalLM"],
    },
    FamilyArchNames {
        family: ModelFamily::GptOss,
        gguf: &["gptoss"],
        safetensors: &["GptOssForCausalLM"],
    },
    FamilyArchNames {
        family: ModelFamily::Step,
        gguf: &["step", "step3"],
        // Step-3.5-Flash ships Step3p5ForCausalLM.
        safetensors: &["Step3ForCausalLM", "Step3p5ForCausalLM"],
    },
    FamilyArchNames {
        family: ModelFamily::MiniMax,
        gguf: &["minimax"],
        safetensors: &["MiniMaxForCausalLM", "MiniMaxM2ForCausalLM"],
    },
    FamilyArchNames {
        family: ModelFamily::Cohere,
        gguf: &["cohere"],
        // Command A+ uses Cohere2Vision + text_config backbone.
        safetensors: &["CohereForCausalLM", "Cohere2VisionForConditionalGeneration"],
    },
    FamilyArchNames {
        family: ModelFamily::Grin,
        gguf: &["grin", "grinmoe"],
        // microsoft/GRIN-MoE reports "GRIN-MoE" as architectures[0].
        safetensors: &["GrinMoeForCausalLM", "GRIN-MoE"],
    },
    FamilyArchNames {
        family: ModelFamily::Skyworks,
        gguf: &["skyworks", "skyworkmoe"],
        // ST left unsupported: documented Skywork-MoE-Base-FP8 ships F8_*
        // routers, but extract_tensor_f32 has no F8 dequant arm yet (Codex P2).
        // GGUF path remains; re-add SkyworkForCausalLM once FP8 extraction lands.
        safetensors: &[],
    },
    FamilyArchNames {
        family: ModelFamily::Trinity,
        gguf: &["trinity"],
        // arcee-ai Trinity Nano is AFMoE (AfmoeForCausalLM).
        safetensors: &["TrinityForCausalLM", "AfmoeForCausalLM"],
    },
    FamilyArchNames {
        family: ModelFamily::Grok,
        gguf: &["grok"],
        // xai-org/grok-1 ships Grok1* class names, not GrokForCausalLM.
        // Grok2ForCausalLM (grok_2_fp8_cloud / TevunahAi/grok-2-FP8) left
        // unmapped until extract_tensor_f32 supports F8_* (same as Skywork).
        safetensors: &[
            "GrokForCausalLM",
            "Grok1ForCausalLM",
            "Grok1ModelForCausalLM",
        ],
    },
];

fn map_architecture(architecture: &str, format: ArchFormat) -> Option<ModelFamily> {
    FAMILY_ARCHES.iter().find_map(|entry| {
        let names = match format {
            ArchFormat::Gguf => entry.gguf,
            ArchFormat::Safetensors => entry.safetensors,
        };
        names.contains(&architecture).then_some(entry.family)
    })
}

/// Unified family inference for GGUF and Safetensors architecture strings.
///
/// Format-specific arch tables live in [`map_architecture`]; override
/// compatibility and error formatting are shared here.
fn infer_family_for_format(
    architecture: &str,
    family_override: Option<ModelFamily>,
    path: &str,
    format: ArchFormat,
) -> Result<ModelFamily> {
    let inferred = map_architecture(architecture, format).ok_or_else(|| {
        HybridError::UnsupportedFormat(format!(
            "unsupported {} architecture '{architecture}' in '{path}'",
            format.label()
        ))
    })?;

    if let Some(expected) = family_override {
        if !check_family_compatibility(expected, inferred) {
            return Err(HybridError::InvalidConfig(format!(
                "model_family override {:?} does not match {} architecture '{architecture}'",
                expected,
                format.label()
            )));
        }
        Ok(expected)
    } else {
        Ok(inferred)
    }
}

/// GGUF entry point (kept for call sites and unit tests).
fn infer_family(
    architecture: &str,
    family_override: Option<ModelFamily>,
    path: &str,
) -> Result<ModelFamily> {
    infer_family_for_format(architecture, family_override, path, ArchFormat::Gguf)
}

/// Safetensors entry point (kept for call sites and unit tests).
fn infer_family_safetensors(
    architecture: &str,
    family_override: Option<ModelFamily>,
    path: &str,
) -> Result<ModelFamily> {
    infer_family_for_format(architecture, family_override, path, ArchFormat::Safetensors)
}

#[cfg(test)]
mod tests {
    use super::infer_family;
    use crate::types::ModelFamily;

    #[test]
    fn infer_family_supports_zaya_and_glm4_architectures() {
        assert_eq!(
            infer_family("zaya", None, "test.gguf").unwrap(),
            ModelFamily::Zaya
        );
        assert_eq!(
            infer_family("glm4", None, "test.gguf").unwrap(),
            ModelFamily::Glm4
        );
        assert_eq!(
            infer_family("glm4moe", None, "test.gguf").unwrap(),
            ModelFamily::Glm4
        );
    }

    #[test]
    fn infer_family_supports_granite_architecture() {
        assert_eq!(
            infer_family("granite", None, "test.gguf").unwrap(),
            ModelFamily::Granite31A800M
        );
    }

    #[test]
    fn infer_family_supports_moonlight_architecture() {
        assert_eq!(
            infer_family("moonlight", None, "test.gguf").unwrap(),
            ModelFamily::Moonlight16BA3B
        );
    }

    #[test]
    fn infer_family_supports_granitemoe_architecture() {
        assert_eq!(
            infer_family("granitemoe", None, "test.gguf").unwrap(),
            ModelFamily::Granite31A800M
        );
    }

    #[test]
    fn infer_family_supports_nemotron_architecture() {
        assert_eq!(
            infer_family("nemotronh", None, "test.gguf").unwrap(),
            ModelFamily::Nemotron
        );
    }

    #[test]
    fn infer_family_supports_new_cloud_architectures() {
        assert_eq!(
            infer_family("gptoss", None, "test.gguf").unwrap(),
            ModelFamily::GptOss
        );
        assert_eq!(
            infer_family("step3", None, "test.gguf").unwrap(),
            ModelFamily::Step
        );
        assert_eq!(
            infer_family("grinmoe", None, "test.gguf").unwrap(),
            ModelFamily::Grin
        );
        assert_eq!(
            infer_family("skyworks", None, "test.gguf").unwrap(),
            ModelFamily::Skyworks
        );
        assert_eq!(
            infer_family("trinity", None, "test.gguf").unwrap(),
            ModelFamily::Trinity
        );
        assert_eq!(
            infer_family("grok", None, "test.gguf").unwrap(),
            ModelFamily::Grok
        );
    }

    #[test]
    fn infer_family_safetensors_moonlight_and_granite_moe_tags() {
        assert_eq!(
            super::infer_family_safetensors("DeepseekV3ForCausalLM", None, "test.safetensors")
                .unwrap(),
            ModelFamily::Moonlight16BA3B
        );
        assert_eq!(
            super::infer_family_safetensors("MoonlightForCausalLM", None, "test.safetensors")
                .unwrap(),
            ModelFamily::Moonlight16BA3B
        );
        assert_eq!(
            super::infer_family_safetensors("GraniteMoeForCausalLM", None, "test.safetensors")
                .unwrap(),
            ModelFamily::Granite31A800M
        );
        // Dense Granite is not the MoE A800M family.
        assert!(
            super::infer_family_safetensors("GraniteForCausalLM", None, "test.safetensors")
                .is_err()
        );
    }

    #[test]
    fn infer_family_safetensors_shipped_cloud_architecture_aliases() {
        use super::infer_family_safetensors as st;
        assert_eq!(
            st("Step3p5ForCausalLM", None, "t").unwrap(),
            ModelFamily::Step
        );
        assert_eq!(
            st("MiniMaxM2ForCausalLM", None, "t").unwrap(),
            ModelFamily::MiniMax
        );
        assert_eq!(st("GRIN-MoE", None, "t").unwrap(), ModelFamily::Grin);
        assert_eq!(
            st("AfmoeForCausalLM", None, "t").unwrap(),
            ModelFamily::Trinity
        );
        assert_eq!(
            st("Grok1ForCausalLM", None, "t").unwrap(),
            ModelFamily::Grok
        );
        assert_eq!(
            st("Glm4MoeLiteForCausalLM", None, "t").unwrap(),
            ModelFamily::Glm4
        );
        assert_eq!(
            st("Llama4ForConditionalGeneration", None, "t").unwrap(),
            ModelFamily::LlamaMoe
        );
        assert_eq!(
            st("Gemma4ForConditionalGeneration", None, "t").unwrap(),
            ModelFamily::Gemma4
        );
        assert_eq!(
            st("Cohere2VisionForConditionalGeneration", None, "t").unwrap(),
            ModelFamily::Cohere
        );
        assert_eq!(
            st("Qwen3_5MoeForConditionalGeneration", None, "t").unwrap(),
            ModelFamily::Qwen3Moe
        );
        assert_eq!(st("EmoForCausalLM", None, "t").unwrap(), ModelFamily::Olmoe);
        // V4 hash-MoE, Grok2-FP8, dense GLM4, Zaya, Skywork-FP8: fail closed.
        assert!(st("DeepseekV4ForCausalLM", None, "t").is_err());
        assert!(st("Grok2ForCausalLM", None, "t").is_err());
        assert!(st("Glm4ForCausalLM", None, "t").is_err());
        assert!(st("ZayaForCausalLM", None, "t").is_err());
        assert!(st("SkyworkForCausalLM", None, "t").is_err());
        assert!(st("SkyworkMoeForCausalLM", None, "t").is_err());
    }

    #[test]
    fn safetensors_routing_candidates_cover_codex_layouts() {
        use super::safetensors_routing_candidates as c;
        assert!(
            c(ModelFamily::Granite31A800M)
                .contains(&"model.layers.0.block_sparse_moe.router.layer.weight")
        );
        assert!(c(ModelFamily::Step).contains(&"model.layers.3.moe.gate.weight"));
        assert!(
            c(ModelFamily::Qwen3Moe).contains(&"model.language_model.layers.0.mlp.gate.weight")
        );
        assert!(c(ModelFamily::Cohere).contains(&"model.language_model.layers.0.mlp.gate.weight"));
        assert!(c(ModelFamily::Lfm2Moe).contains(&"model.layers.0.feed_forward.gate.weight"));
        assert!(c(ModelFamily::Lfm2Moe).contains(&"model.layers.10.feed_forward.gate.weight"));
        assert!(c(ModelFamily::Grin).contains(&"model.layers.1.block_sparse_moe.gate.weight"));
        assert!(c(ModelFamily::SlimMoe).contains(&"model.layers.0.block_sparse_moe.gate.weight"));
    }

    #[test]
    fn test_family_compatibility_overrides() {
        // Test Moonlight/DeepSeek2 compatibility checks for GGUF path
        assert_eq!(
            infer_family("moonlight", Some(ModelFamily::DeepSeek2), "test.gguf").unwrap(),
            ModelFamily::DeepSeek2
        );
        assert_eq!(
            infer_family("deepseek2", Some(ModelFamily::Moonlight16BA3B), "test.gguf").unwrap(),
            ModelFamily::Moonlight16BA3B
        );

        // Verify that incompatible overrides still error
        assert!(infer_family("olmoe", Some(ModelFamily::DeepSeek2), "test.gguf").is_err());

        // Test Moonlight/DeepSeek2 compatibility checks for Safetensors path
        assert_eq!(
            super::infer_family_safetensors(
                "DeepseekV2ForCausalLM",
                Some(ModelFamily::Moonlight16BA3B),
                "test.safetensors"
            )
            .unwrap(),
            ModelFamily::Moonlight16BA3B
        );
        assert_eq!(
            super::infer_family_safetensors("DeepseekV2ForCausalLM", None, "test.safetensors")
                .unwrap(),
            ModelFamily::DeepSeek2
        );
        assert!(
            super::infer_family_safetensors(
                "OlmoeForCausalLM",
                Some(ModelFamily::DeepSeek2),
                "test.safetensors"
            )
            .is_err()
        );
    }
}
