//! OLMoE-style routing interface backed by either a GGUF or local Qwen safetensors checkpoint.

use super::qwen_local::{
    QwenLocalCheckpoint, load_qwen_local_checkpoint, probe_qwen_local_checkpoint,
};
use crate::error::{HybridError, Result};
use crate::types::{EMBEDDING_DIM, OlmoeExecutionMode};
use memmap2::{MmapMut, MmapOptions};
use std::cmp::Ordering;
use std::collections::HashMap;
use std::ffi::c_void;
use std::fs::OpenOptions;
use std::path::Path;
use std::slice;

const OLMOE_HIDDEN: usize = 2048;
const OLMOE_NUM_EXPERTS: usize = 64;
const OLMOE_NUM_LAYERS: usize = 16;
const ROUTING_TENSOR_CANDIDATES: [&str; 2] = ["blk.0.ffn_gate_inp.weight", "blk.0.ffn_gate.weight"];
const GGUF_MAGIC: [u8; 4] = [b'G', b'G', b'U', b'F'];
const GGUF_VERSION: u32 = 3;
const GGML_TYPE_F32: u32 = 0;
const GGML_TYPE_F16: u32 = 1;
const GGML_TYPE_Q8_0: u32 = 8;
const GGML_Q8_0_BLOCK_SIZE: usize = 32;
const GGML_Q8_0_BLOCK_BYTES: usize = 34;

#[inline(always)]
fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits as u32) & 0x8000) << 16;
    let exp = ((bits as u32) & 0x7C00) >> 10;
    let mant = ((bits as u32) & 0x03FF) << 13;
    let val = if exp == 0 {
        mant
    } else if exp == 31 {
        0x7F800000 | mant
    } else {
        ((exp + 127 - 15) << 23) | mant
    };
    f32::from_bits(sign | val)
}

fn decode_q8_0_row(
    raw: &[u8],
    source_hidden_size: usize,
    active_hidden_size: usize,
) -> Result<Vec<f32>> {
    if !source_hidden_size.is_multiple_of(GGML_Q8_0_BLOCK_SIZE) {
        return Err(HybridError::UnsupportedFormat(format!(
            "Q8_0 row width {source_hidden_size} is not divisible by {}",
            GGML_Q8_0_BLOCK_SIZE
        )));
    }

    let blocks_per_row = source_hidden_size / GGML_Q8_0_BLOCK_SIZE;
    let expected_bytes = blocks_per_row * GGML_Q8_0_BLOCK_BYTES;
    if raw.len() < expected_bytes {
        return Err(HybridError::UnsupportedFormat(format!(
            "Q8_0 row is truncated: expected {expected_bytes} bytes, got {}",
            raw.len()
        )));
    }

    let mut out = Vec::with_capacity(active_hidden_size);
    for block_idx in 0..blocks_per_row {
        let base = block_idx * GGML_Q8_0_BLOCK_BYTES;
        let scale = f16_to_f32(u16::from_le_bytes([raw[base], raw[base + 1]]));
        for q in &raw[base + 2..base + GGML_Q8_0_BLOCK_BYTES] {
            if out.len() == active_hidden_size {
                return Ok(out);
            }
            out.push((*q as i8 as f32) * scale);
        }
    }

    Ok(out)
}

const GGUF_VALUE_TYPE_UINT8: u32 = 0;
const GGUF_VALUE_TYPE_INT8: u32 = 1;
const GGUF_VALUE_TYPE_UINT16: u32 = 2;
const GGUF_VALUE_TYPE_INT16: u32 = 3;
const GGUF_VALUE_TYPE_UINT32: u32 = 4;
const GGUF_VALUE_TYPE_INT32: u32 = 5;
const GGUF_VALUE_TYPE_FLOAT32: u32 = 6;
const GGUF_VALUE_TYPE_BOOL: u32 = 7;
const GGUF_VALUE_TYPE_STRING: u32 = 8;
const GGUF_VALUE_TYPE_ARRAY: u32 = 9;
const GGUF_VALUE_TYPE_UINT64: u32 = 10;
const GGUF_VALUE_TYPE_INT64: u32 = 11;
const GGUF_VALUE_TYPE_FLOAT64: u32 = 12;

pub struct OLMoE {
    model_path: String,
    num_experts: usize,
    top_k: usize,
    loaded: bool,
    metadata: OlmoeMetadata,
    routing_tensor_name: String,
    execution_mode: OlmoeExecutionMode,
    expert_membranes: Vec<f32>,
    hidden_membranes: Vec<f32>,
    threshold: f32,
    decay: f32,
    checkpoint: Option<LoadedCheckpoint>,
}

#[derive(Debug, Clone, Default)]
pub(crate) struct OlmoeMetadata {
    pub architecture: String,
    pub hidden_size: usize,
    pub source_hidden_size: usize,
    pub num_layers: usize,
    pub num_experts: usize,
    pub expert_used_count: Option<usize>,
    pub quantization: String,
    pub checkpoint_format: String,
}

#[derive(Debug, Clone)]
pub struct OlmoeCheckpointMetadata {
    pub architecture: String,
    pub hidden_size: usize,
    pub source_hidden_size: usize,
    pub num_layers: usize,
    pub num_experts: usize,
    pub expert_used_count: Option<usize>,
    pub quantization: String,
    pub routing_tensor_name: String,
    pub checkpoint_format: String,
}

#[derive(Debug, Clone)]
pub struct OlmoeOutput {
    pub expert_weights: Vec<f32>,
    pub selected_experts: Vec<usize>,
    pub hidden: Vec<f32>,
}

#[derive(Debug)]
struct MappedOlmoeCheckpoint {
    mmap: MmapMut,
    tensors: HashMap<String, GgufTensorInfo>,
    registered_gpu_synapse: Option<RegisteredTensorSliceU16>,
}

#[derive(Debug)]
enum LoadedCheckpoint {
    Gguf(MappedOlmoeCheckpoint),
    QwenLocal(QwenLocalCheckpoint),
}

#[derive(Debug, Clone)]
struct GgufTensorInfo {
    dims: Vec<usize>,
    ggml_type: u32,
    relative_offset: usize,
    absolute_offset: usize,
    n_elements: usize,
}

#[derive(Debug)]
struct RegisteredTensorSliceU16 {
    tensor_name: String,
    _region: RegisteredCudaRegion,
    ptr: *const u16,
    len: usize,
}

#[derive(Debug)]
struct RegisteredCudaRegion {
    ptr: *mut c_void,
}

struct ParsedCheckpointLayout {
    metadata: OlmoeMetadata,
    tensors: HashMap<String, GgufTensorInfo>,
    routing_tensor_name: String,
}

struct GgufCursor<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl OLMoE {
    pub fn load(model_path: &str, num_experts: usize, top_k: usize) -> Result<Self> {
        Self::load_with_mode_and_local(
            model_path,
            "",
            num_experts,
            top_k,
            OlmoeExecutionMode::StubUniform,
        )
    }

    pub fn load_with_mode(
        model_path: &str,
        num_experts: usize,
        top_k: usize,
        execution_mode: OlmoeExecutionMode,
    ) -> Result<Self> {
        Self::load_with_mode_and_local(model_path, "", num_experts, top_k, execution_mode)
    }

    pub fn load_with_mode_and_local(
        model_path: &str,
        local_checkpoint_dir: &str,
        num_experts: usize,
        top_k: usize,
        execution_mode: OlmoeExecutionMode,
    ) -> Result<Self> {
        let top_k = top_k.max(1).min(num_experts);
        let active_path = resolve_active_model_path(model_path, local_checkpoint_dir);

        if active_path.is_empty() {
            return Ok(Self {
                model_path: String::new(),
                num_experts,
                top_k,
                loaded: false,
                metadata: OlmoeMetadata {
                    architecture: "stub".into(),
                    hidden_size: OLMOE_HIDDEN,
                    source_hidden_size: OLMOE_HIDDEN,
                    num_layers: OLMOE_NUM_LAYERS,
                    num_experts: OLMOE_NUM_EXPERTS,
                    expert_used_count: None,
                    quantization: "stub".into(),
                    checkpoint_format: "stub".into(),
                },
                routing_tensor_name: ROUTING_TENSOR_CANDIDATES[0].into(),
                execution_mode,
                expert_membranes: vec![0.0; num_experts],
                hidden_membranes: vec![0.0; EMBEDDING_DIM],
                threshold: 0.75,
                decay: 0.91,
                checkpoint: None,
            });
        }

        let (metadata, routing_tensor_name, checkpoint) =
            Self::probe_and_map_with_local(model_path, local_checkpoint_dir)?;
        if num_experts > metadata.num_experts {
            return Err(HybridError::InvalidConfig(format!(
                "num_experts ({num_experts}) exceeds checkpoint expert_count ({})",
                metadata.num_experts
            )));
        }
        let hidden_size = metadata.hidden_size;

        Ok(Self {
            model_path: active_path.to_owned(),
            num_experts,
            top_k,
            loaded: true,
            metadata,
            routing_tensor_name,
            execution_mode,
            expert_membranes: vec![0.0; num_experts],
            hidden_membranes: vec![0.0; hidden_size],
            threshold: 0.75,
            decay: 0.91,
            checkpoint: Some(checkpoint),
        })
    }

    pub fn probe_checkpoint(model_path: &str) -> Result<OlmoeCheckpointMetadata> {
        Self::probe_checkpoint_with_local(model_path, "")
    }

    pub fn probe_checkpoint_with_local(
        model_path: &str,
        local_checkpoint_dir: &str,
    ) -> Result<OlmoeCheckpointMetadata> {
        let active_path = resolve_active_model_path(model_path, local_checkpoint_dir);
        if !local_checkpoint_dir.trim().is_empty() || Path::new(model_path).is_dir() {
            return probe_qwen_local_checkpoint(active_path);
        }

        let file = OpenOptions::new()
            .read(true)
            .open(model_path)
            .map_err(|e| HybridError::ModelLoad {
                path: model_path.to_owned(),
                reason: e.to_string(),
            })?;
        let mmap =
            unsafe { MmapOptions::new().map_copy(&file) }.map_err(|e| HybridError::ModelLoad {
                path: model_path.to_owned(),
                reason: format!("copy-on-write mmap failed: {e}"),
            })?;
        let parsed = parse_checkpoint_layout(&mmap, model_path)?;
        Ok(OlmoeCheckpointMetadata {
            architecture: parsed.metadata.architecture,
            hidden_size: parsed.metadata.hidden_size,
            source_hidden_size: parsed.metadata.source_hidden_size,
            num_layers: parsed.metadata.num_layers,
            num_experts: parsed.metadata.num_experts,
            expert_used_count: parsed.metadata.expert_used_count,
            quantization: parsed.metadata.quantization,
            routing_tensor_name: parsed.routing_tensor_name,
            checkpoint_format: parsed.metadata.checkpoint_format,
        })
    }

    pub fn forward(&mut self, embedding: &[f32]) -> Result<OlmoeOutput> {
        if embedding.len() != self.metadata.hidden_size {
            return Err(HybridError::InputLengthMismatch {
                expected: self.metadata.hidden_size,
                got: embedding.len(),
            });
        }

        match self.execution_mode {
            OlmoeExecutionMode::StubUniform => Ok(self.stub_output()),
            OlmoeExecutionMode::DenseSim => self.simulate_moe_routing(embedding),
            OlmoeExecutionMode::SpikingSim => self.spiking_moe_routing(embedding),
        }
    }

    pub fn extract_token_embedding(&mut self, token_id: usize) -> Result<Vec<f32>> {
        let path = self.model_path.clone();
        let hidden_size = self.metadata.hidden_size;
        let source_hidden_size = self.metadata.source_hidden_size;
        let checkpoint = self
            .checkpoint
            .as_ref()
            .ok_or_else(|| HybridError::ModelLoad {
                path: path.clone(),
                reason: "checkpoint not loaded".into(),
            })?;

        if let LoadedCheckpoint::QwenLocal(checkpoint) = checkpoint {
            return checkpoint.extract_token_embedding(&path, token_id, hidden_size);
        }

        let LoadedCheckpoint::Gguf(checkpoint) = checkpoint else {
            unreachable!("QwenLocal handled above");
        };

        let info = checkpoint.tensor_info("token_embd.weight", &path)?.clone();
        let d0 = info.dims[0];
        let d1 = info.dims.get(1).copied().unwrap_or(0);

        match info.ggml_type {
            GGML_TYPE_F32 => {
                let weights = checkpoint.f32_tensor("token_embd.weight", &path)?;
                if d0 == source_hidden_size {
                    if token_id >= d1 {
                        return Err(HybridError::InputLengthMismatch {
                            expected: d1,
                            got: token_id,
                        });
                    }
                    let start = token_id * d0;
                    Ok(weights[start..start + hidden_size].to_vec())
                } else if d1 == source_hidden_size {
                    if token_id >= d0 {
                        return Err(HybridError::InputLengthMismatch {
                            expected: d0,
                            got: token_id,
                        });
                    }
                    Ok((0..hidden_size)
                        .map(|dim| weights[dim * d0 + token_id])
                        .collect())
                } else {
                    Err(HybridError::UnsupportedFormat(format!(
                        "tensor 'token_embd.weight' has unexpected dimensions {:?}",
                        info.dims
                    )))
                }
            }
            GGML_TYPE_F16 => {
                let byte_start = info.absolute_offset;
                let byte_end = byte_start + info.n_elements * 2;
                if byte_end > checkpoint.mmap.len() {
                    return Err(HybridError::ModelLoad {
                        path: path.clone(),
                        reason: "token_embd.weight F16 tensor extends beyond mapped file".into(),
                    });
                }
                let raw = &checkpoint.mmap[byte_start..byte_end];
                let u16s: Vec<u16> = raw
                    .chunks_exact(2)
                    .map(|b| u16::from_le_bytes([b[0], b[1]]))
                    .collect();
                if d0 == source_hidden_size {
                    if token_id >= d1 {
                        return Err(HybridError::InputLengthMismatch {
                            expected: d1,
                            got: token_id,
                        });
                    }
                    let start = token_id * d0;
                    Ok(u16s[start..start + hidden_size]
                        .iter()
                        .map(|&b| f16_to_f32(b))
                        .collect())
                } else if d1 == source_hidden_size {
                    if token_id >= d0 {
                        return Err(HybridError::InputLengthMismatch {
                            expected: d0,
                            got: token_id,
                        });
                    }
                    Ok((0..hidden_size)
                        .map(|dim| f16_to_f32(u16s[dim * d0 + token_id]))
                        .collect())
                } else {
                    Err(HybridError::UnsupportedFormat(format!(
                        "tensor 'token_embd.weight' has unexpected dimensions {:?}",
                        info.dims
                    )))
                }
            }
            GGML_TYPE_Q8_0 => {
                if d0 != source_hidden_size {
                    return Err(HybridError::UnsupportedFormat(format!(
                        "Q8_0 token embeddings must be [hidden_size, vocab], got {:?}",
                        info.dims
                    )));
                }
                if token_id >= d1 {
                    return Err(HybridError::InputLengthMismatch {
                        expected: d1,
                        got: token_id,
                    });
                }

                let blocks_per_row = source_hidden_size / GGML_Q8_0_BLOCK_SIZE;
                let row_bytes = blocks_per_row * GGML_Q8_0_BLOCK_BYTES;
                let byte_start = info.absolute_offset + token_id * row_bytes;
                let byte_end = byte_start + row_bytes;
                if byte_end > checkpoint.mmap.len() {
                    return Err(HybridError::ModelLoad {
                        path: path.clone(),
                        reason: "token_embd.weight Q8_0 tensor extends beyond mapped file".into(),
                    });
                }

                decode_q8_0_row(
                    &checkpoint.mmap[byte_start..byte_end],
                    source_hidden_size,
                    hidden_size,
                )
            }
            other => Err(HybridError::UnsupportedFormat(format!(
                "tensor 'token_embd.weight' has unsupported ggml_type={other}"
            ))),
        }
    }

    pub(crate) fn registered_gpu_synapse_weights(&mut self, tensor_name: &str) -> Result<&[u16]> {
        let path = self.model_path.clone();
        let checkpoint = self
            .checkpoint
            .as_mut()
            .ok_or_else(|| HybridError::ModelLoad {
                path: path.clone(),
                reason: "checkpoint not loaded".into(),
            })?;
        match checkpoint {
            LoadedCheckpoint::Gguf(checkpoint) => {
                checkpoint.registered_f16_tensor(tensor_name, &path)
            }
            LoadedCheckpoint::QwenLocal(_) => Err(HybridError::UnsupportedFormat(format!(
                "local safetensors GPTQ checkpoints do not expose a registered F16 recurrent synapse tensor for '{tensor_name}'"
            ))),
        }
    }

    fn probe_and_map(path: &str) -> Result<(OlmoeMetadata, String, MappedOlmoeCheckpoint)> {
        let file =
            OpenOptions::new()
                .read(true)
                .open(path)
                .map_err(|e| HybridError::ModelLoad {
                    path: path.to_owned(),
                    reason: e.to_string(),
                })?;
        // CUDA host registration rejects the shared file mapping used by `map`,
        // but accepts a writable MAP_PRIVATE/COW file view without requiring any
        // Vec staging or mutating the checkpoint on disk.
        let mmap =
            unsafe { MmapOptions::new().map_copy(&file) }.map_err(|e| HybridError::ModelLoad {
                path: path.to_owned(),
                reason: format!("copy-on-write mmap failed: {e}"),
            })?;

        let parsed = parse_checkpoint_layout(&mmap, path)?;
        let routing = parsed
            .tensors
            .get(parsed.routing_tensor_name.as_str())
            .ok_or_else(|| HybridError::MissingTensor {
                name: parsed.routing_tensor_name.clone(),
                path: path.to_owned(),
            })?;
        validate_routing_tensor(
            path,
            parsed.routing_tensor_name.as_str(),
            routing,
            parsed.metadata.source_hidden_size,
            parsed.metadata.num_experts,
        )?;

        Ok((
            parsed.metadata,
            parsed.routing_tensor_name,
            MappedOlmoeCheckpoint {
                mmap,
                tensors: parsed.tensors,
                registered_gpu_synapse: None,
            },
        ))
    }

    fn probe_and_map_with_local(
        path: &str,
        local_checkpoint_dir: &str,
    ) -> Result<(OlmoeMetadata, String, LoadedCheckpoint)> {
        let active_path = resolve_active_model_path(path, local_checkpoint_dir);
        if !local_checkpoint_dir.trim().is_empty() || Path::new(path).is_dir() {
            let (metadata, routing_tensor_name, checkpoint) =
                load_qwen_local_checkpoint(active_path)?;
            return Ok((
                metadata,
                routing_tensor_name,
                LoadedCheckpoint::QwenLocal(checkpoint),
            ));
        }

        let (metadata, routing_tensor_name, checkpoint) = Self::probe_and_map(path)?;
        Ok((
            metadata,
            routing_tensor_name,
            LoadedCheckpoint::Gguf(checkpoint),
        ))
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

        let mut hidden = vec![0.0f32; self.metadata.hidden_size];
        for (j, h) in hidden.iter_mut().enumerate() {
            let input = embedding[j] * active_mass;
            self.hidden_membranes[j] = self.hidden_membranes[j] * self.decay + input;

            let spike = if self.hidden_membranes[j] > self.threshold {
                self.hidden_membranes[j] -= self.threshold;
                1.0
            } else if self.hidden_membranes[j] < -self.threshold {
                self.hidden_membranes[j] += self.threshold;
                -1.0
            } else {
                0.0
            };

            *h = spike * 0.3;
        }

        Ok(OlmoeOutput {
            expert_weights,
            selected_experts,
            hidden,
        })
    }

    fn compute_gate_scores(&self, embedding: &[f32]) -> Result<Vec<f32>> {
        if let Some(checkpoint) = &self.checkpoint {
            match checkpoint {
                LoadedCheckpoint::Gguf(checkpoint) => {
                    let info =
                        checkpoint.tensor_info(&self.routing_tensor_name, &self.model_path)?;
                    let weights =
                        checkpoint.f32_tensor(&self.routing_tensor_name, &self.model_path)?;
                    let mut gate_scores = Vec::with_capacity(self.num_experts);
                    for expert_id in 0..self.num_experts {
                        let mut score = 0.0f32;
                        for (dim, &value) in embedding.iter().enumerate() {
                            let index = routing_weight_index(
                                info,
                                expert_id,
                                dim,
                                self.metadata.source_hidden_size,
                                self.num_experts,
                            )?;
                            score += weights[index] * value;
                        }
                        gate_scores.push(score);
                    }
                    return Ok(gate_scores);
                }
                LoadedCheckpoint::QwenLocal(checkpoint) => {
                    return checkpoint.compute_gate_scores(
                        &self.model_path,
                        embedding,
                        self.num_experts,
                        self.metadata.source_hidden_size,
                    );
                }
            }
        }

        // Deterministic synthetic fallback for DenseSim/SpikingSim when no checkpoint is present.
        let chunk = (embedding.len() / self.num_experts.max(1)).max(1);
        let mut gate_scores = Vec::with_capacity(self.num_experts);
        for expert_id in 0..self.num_experts {
            let start = (expert_id * chunk) % embedding.len().max(1);
            let end = (start + chunk).min(embedding.len());
            gate_scores.push(embedding[start..end].iter().sum());
        }
        Ok(gate_scores)
    }

    fn stub_output(&self) -> OlmoeOutput {
        let n = self.num_experts;
        let expert_weights = vec![1.0 / n as f32; n];
        let selected_experts = (0..self.top_k).collect();
        let hidden = vec![0.0f32; self.metadata.hidden_size];
        OlmoeOutput {
            expert_weights,
            selected_experts,
            hidden,
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

    pub fn quantization(&self) -> &str {
        &self.metadata.quantization
    }

    pub fn checkpoint_format(&self) -> &str {
        &self.metadata.checkpoint_format
    }

    pub fn architecture(&self) -> &str {
        &self.metadata.architecture
    }

    pub fn hidden_size(&self) -> usize {
        self.metadata.hidden_size
    }

    pub fn source_hidden_size(&self) -> usize {
        self.metadata.source_hidden_size
    }

    pub fn num_layers(&self) -> usize {
        self.metadata.num_layers
    }

    pub fn checkpoint_num_experts(&self) -> usize {
        self.metadata.num_experts
    }

    pub fn expert_used_count(&self) -> Option<usize> {
        self.metadata.expert_used_count
    }

    pub fn num_experts(&self) -> usize {
        self.num_experts
    }

    pub fn routing_tensor_name(&self) -> &str {
        &self.routing_tensor_name
    }

    pub fn routing_source(&self) -> &str {
        if self.loaded {
            "real-first-block"
        } else {
            "synthetic-fallback"
        }
    }

    pub fn execution_mode(&self) -> OlmoeExecutionMode {
        self.execution_mode
    }

    #[cfg(test)]
    pub(crate) fn has_state_activity(&self) -> bool {
        self.expert_membranes.iter().any(|&v| v != 0.0)
            || self.hidden_membranes.iter().any(|&v| v != 0.0)
    }
}

impl MappedOlmoeCheckpoint {
    fn tensor_info<'a>(&'a self, name: &str, path: &str) -> Result<&'a GgufTensorInfo> {
        self.tensors
            .get(name)
            .ok_or_else(|| HybridError::MissingTensor {
                name: name.to_owned(),
                path: path.to_owned(),
            })
    }

    fn f32_tensor<'a>(&'a self, name: &str, path: &str) -> Result<&'a [f32]> {
        let info = self.tensor_info(name, path)?;
        if info.ggml_type != GGML_TYPE_F32 {
            return Err(HybridError::UnsupportedFormat(format!(
                "tensor '{name}' must be F32, got ggml_type={}",
                info.ggml_type
            )));
        }

        let start = info.absolute_offset;
        let end = start + info.n_elements * std::mem::size_of::<f32>();
        if end > self.mmap.len() {
            return Err(HybridError::ModelLoad {
                path: path.to_owned(),
                reason: format!("tensor '{name}' extends beyond mapped file"),
            });
        }

        let ptr = unsafe { self.mmap.as_ptr().add(start) as *const f32 };
        Ok(unsafe { slice::from_raw_parts(ptr, info.n_elements) })
    }

    fn registered_f16_tensor<'a>(&'a mut self, name: &str, path: &str) -> Result<&'a [u16]> {
        let info = self.tensor_info(name, path)?.clone();
        if info.ggml_type != GGML_TYPE_F16 {
            return Err(HybridError::UnsupportedFormat(format!(
                "tensor '{name}' must be F16, got ggml_type={}",
                info.ggml_type
            )));
        }

        if self
            .registered_gpu_synapse
            .as_ref()
            .map(|registered| registered.tensor_name.as_str() == name)
            .unwrap_or(false)
        {
            return Ok(self
                .registered_gpu_synapse
                .as_ref()
                .expect("checked above")
                .as_slice());
        }

        self.registered_gpu_synapse = Some(RegisteredTensorSliceU16::register(
            name,
            &self.mmap,
            info.absolute_offset,
            info.n_elements,
            path,
        )?);

        Ok(self
            .registered_gpu_synapse
            .as_ref()
            .expect("registered above")
            .as_slice())
    }
}

impl RegisteredTensorSliceU16 {
    fn register(
        tensor_name: &str,
        mmap: &MmapMut,
        absolute_offset: usize,
        n_elements: usize,
        path: &str,
    ) -> Result<Self> {
        let byte_len = n_elements
            .checked_mul(std::mem::size_of::<u16>())
            .ok_or_else(|| HybridError::ModelLoad {
                path: path.to_owned(),
                reason: format!("tensor '{tensor_name}' byte length overflow"),
            })?;

        let page_size = page_size_bytes(path)?;
        let aligned_start = absolute_offset / page_size * page_size;
        let aligned_end = align_up(absolute_offset + byte_len, page_size);
        let register_len = aligned_end - aligned_start;

        if aligned_end > mmap.len() {
            return Err(HybridError::ModelLoad {
                path: path.to_owned(),
                reason: format!("tensor '{tensor_name}' registration range exceeds mmap"),
            });
        }

        let register_ptr = unsafe { mmap.as_ptr().add(aligned_start) as *mut c_void };
        cuda_host_register(register_ptr, register_len, path, tensor_name)?;

        let tensor_ptr = unsafe { mmap.as_ptr().add(absolute_offset) as *const u16 };
        Ok(Self {
            tensor_name: tensor_name.to_owned(),
            _region: RegisteredCudaRegion { ptr: register_ptr },
            ptr: tensor_ptr,
            len: n_elements,
        })
    }

    fn as_slice(&self) -> &[u16] {
        unsafe { slice::from_raw_parts(self.ptr, self.len) }
    }
}

impl Drop for RegisteredCudaRegion {
    fn drop(&mut self) {
        let result = unsafe { cust::sys::cuMemHostUnregister(self.ptr) };
        if result != cust::sys::CUresult::CUDA_SUCCESS {
            // The model remains valid; dropping the registration should not panic.
        }
    }
}

fn parse_checkpoint_layout(bytes: &[u8], path: &str) -> Result<ParsedCheckpointLayout> {
    let mut cursor = GgufCursor::new(bytes);

    let magic = cursor.read_exact(4, path)?;
    if magic != GGUF_MAGIC {
        return Err(HybridError::UnsupportedFormat(format!(
            "unrecognised model magic bytes: {magic:?}"
        )));
    }

    let version = cursor.read_u32(path)?;
    if version != GGUF_VERSION {
        return Err(HybridError::UnsupportedFormat(format!(
            "unsupported GGUF version {version}; expected {GGUF_VERSION}"
        )));
    }

    let tensor_count = cursor.read_u64(path)? as usize;
    let kv_count = cursor.read_u64(path)? as usize;

    let mut alignment = 32usize;
    let mut file_type = None;
    let mut architecture = "olmoe".to_owned();
    let mut numeric_metadata = HashMap::new();

    for _ in 0..kv_count {
        let key = cursor.read_string(path)?;
        let value_type = cursor.read_u32(path)?;
        match key.as_str() {
            "general.alignment" => alignment = cursor.read_numeric_as_usize(value_type, path)?,
            "general.file_type" => file_type = Some(cursor.read_numeric_as_u32(value_type, path)?),
            "general.architecture" => {
                architecture = cursor.read_string(path)?;
            }
            _ if is_numeric_gguf_value(value_type) => {
                let value = cursor.read_numeric_as_usize(value_type, path)?;
                numeric_metadata.insert(key, value);
            }
            _ => cursor.skip_value(value_type, path)?,
        }
    }

    let mut tensors = HashMap::with_capacity(tensor_count);
    for _ in 0..tensor_count {
        let name = cursor.read_string(path)?;
        let n_dims = cursor.read_u32(path)? as usize;
        let mut dims = Vec::with_capacity(n_dims);
        for _ in 0..n_dims {
            dims.push(cursor.read_u64(path)? as usize);
        }
        let ggml_type = cursor.read_u32(path)?;
        let relative_offset = cursor.read_u64(path)? as usize;
        let n_elements = dims
            .iter()
            .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
            .ok_or_else(|| HybridError::ModelLoad {
                path: path.to_owned(),
                reason: format!("tensor '{name}' element count overflow"),
            })?;
        tensors.insert(
            name,
            GgufTensorInfo {
                dims,
                ggml_type,
                relative_offset,
                absolute_offset: 0,
                n_elements,
            },
        );
    }

    let tensor_data_offset = align_up(cursor.offset, alignment);
    for tensor in tensors.values_mut() {
        tensor.absolute_offset = tensor_data_offset + tensor.relative_offset;
    }

    let source_hidden_size = numeric_metadata
        .get(format!("{architecture}.embedding_length").as_str())
        .copied()
        .unwrap_or(OLMOE_HIDDEN);
    if source_hidden_size < EMBEDDING_DIM {
        return Err(HybridError::UnsupportedFormat(format!(
            "checkpoint '{path}' hidden size {source_hidden_size} is smaller than required SNN width {EMBEDDING_DIM}"
        )));
    }
    let num_layers = numeric_metadata
        .get(format!("{architecture}.block_count").as_str())
        .copied()
        .unwrap_or(OLMOE_NUM_LAYERS);
    let num_experts = numeric_metadata
        .get(format!("{architecture}.expert_count").as_str())
        .copied()
        .or_else(|| numeric_metadata.get("olmoe.expert_count").copied())
        .unwrap_or(OLMOE_NUM_EXPERTS);
    let expert_used_count = numeric_metadata
        .get(format!("{architecture}.expert_used_count").as_str())
        .copied();
    let routing_tensor_name =
        resolve_routing_tensor_name(path, &tensors, source_hidden_size, num_experts)?;

    Ok(ParsedCheckpointLayout {
        metadata: OlmoeMetadata {
            architecture,
            hidden_size: EMBEDDING_DIM,
            source_hidden_size,
            num_layers,
            num_experts,
            expert_used_count,
            quantization: quantization_label(file_type),
            checkpoint_format: "gguf".into(),
        },
        tensors,
        routing_tensor_name,
    })
}

fn validate_routing_tensor(
    path: &str,
    tensor_name: &str,
    tensor: &GgufTensorInfo,
    hidden_size: usize,
    num_experts: usize,
) -> Result<()> {
    if tensor.ggml_type != GGML_TYPE_F32 {
        return Err(HybridError::UnsupportedFormat(format!(
            "tensor '{tensor_name}' must be F32 in '{path}'"
        )));
    }
    if tensor.dims.len() != 2 {
        return Err(HybridError::UnsupportedFormat(format!(
            "tensor '{tensor_name}' must be rank-2, got {:?}",
            tensor.dims
        )));
    }
    let matches = (tensor.dims[0] == hidden_size && tensor.dims[1] == num_experts)
        || (tensor.dims[0] == num_experts && tensor.dims[1] == hidden_size);
    if !matches {
        return Err(HybridError::UnsupportedFormat(format!(
            "tensor '{tensor_name}' must expose expert axis {num_experts} and hidden axis {hidden_size}, got {:?}",
            tensor.dims
        )));
    }
    Ok(())
}

fn routing_weight_index(
    tensor: &GgufTensorInfo,
    expert_id: usize,
    dim: usize,
    hidden_size: usize,
    num_experts: usize,
) -> Result<usize> {
    let d0 = tensor.dims[0];
    let d1 = tensor.dims[1];

    if d0 == hidden_size && d1 >= num_experts {
        return Ok(dim * d1 + expert_id);
    }
    if d0 >= num_experts && d1 == hidden_size {
        return Ok(expert_id * d1 + dim);
    }

    Err(HybridError::UnsupportedFormat(format!(
        "unsupported routing tensor orientation {:?}",
        tensor.dims
    )))
}

fn softmax(scores: &[f32]) -> Vec<f32> {
    if scores.is_empty() {
        return Vec::new();
    }

    let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exp_scores: Vec<f32> = scores
        .iter()
        .map(|&score| (score - max_score).exp())
        .collect();
    let sum_exp: f32 = exp_scores.iter().sum();
    if sum_exp <= 0.0 || !sum_exp.is_finite() {
        return vec![1.0 / scores.len() as f32; scores.len()];
    }
    exp_scores
        .into_iter()
        .map(|value| value / sum_exp)
        .collect()
}

fn top_k_indices(weights: &[f32], top_k: usize) -> Vec<usize> {
    let mut indexed: Vec<(usize, f32)> = weights.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
    indexed
        .into_iter()
        .take(top_k)
        .map(|(idx, _)| idx)
        .collect()
}

fn is_numeric_gguf_value(value_type: u32) -> bool {
    matches!(
        value_type,
        GGUF_VALUE_TYPE_UINT8
            | GGUF_VALUE_TYPE_INT8
            | GGUF_VALUE_TYPE_UINT16
            | GGUF_VALUE_TYPE_INT16
            | GGUF_VALUE_TYPE_UINT32
            | GGUF_VALUE_TYPE_INT32
            | GGUF_VALUE_TYPE_UINT64
            | GGUF_VALUE_TYPE_INT64
    )
}

fn resolve_routing_tensor_name(
    path: &str,
    tensors: &HashMap<String, GgufTensorInfo>,
    hidden_size: usize,
    num_experts: usize,
) -> Result<String> {
    for candidate in ROUTING_TENSOR_CANDIDATES {
        let Some(tensor) = tensors.get(candidate) else {
            continue;
        };
        if validate_routing_tensor(path, candidate, tensor, hidden_size, num_experts).is_ok() {
            return Ok(candidate.to_owned());
        }
    }

    let available: Vec<String> = ROUTING_TENSOR_CANDIDATES
        .iter()
        .filter_map(|name| {
            tensors
                .get(*name)
                .map(|tensor| format!("{name}:{:?}", tensor.dims))
        })
        .collect();

    Err(HybridError::UnsupportedFormat(format!(
        "no compatible first-block routing tensor found in '{path}' for hidden_size={hidden_size} expert_count={num_experts}; candidates={available:?}"
    )))
}

fn quantization_label(file_type: Option<u32>) -> String {
    match file_type {
        Some(0) => "F32".into(),
        Some(1) => "F16".into(),
        Some(other) => format!("GGUF({other})"),
        None => "GGUF".into(),
    }
}

fn resolve_active_model_path<'a>(model_path: &'a str, local_checkpoint_dir: &'a str) -> &'a str {
    if !local_checkpoint_dir.trim().is_empty() {
        local_checkpoint_dir
    } else {
        model_path
    }
}

fn page_size_bytes(path: &str) -> Result<usize> {
    let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    if page_size <= 0 {
        return Err(HybridError::ModelLoad {
            path: path.to_owned(),
            reason: "sysconf(_SC_PAGESIZE) failed".into(),
        });
    }
    Ok(page_size as usize)
}

fn align_up(value: usize, alignment: usize) -> usize {
    if alignment == 0 {
        value
    } else {
        value.div_ceil(alignment) * alignment
    }
}

fn cuda_host_register(ptr: *mut c_void, len: usize, path: &str, tensor_name: &str) -> Result<()> {
    let result = unsafe { cust::sys::cuMemHostRegister_v2(ptr, len, 0) };
    if result == cust::sys::CUresult::CUDA_SUCCESS {
        return Ok(());
    }

    Err(HybridError::ModelLoad {
        path: path.to_owned(),
        reason: format!("cuMemHostRegister_v2 failed for '{tensor_name}': {result:?}"),
    })
}

impl<'a> GgufCursor<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }

    fn read_exact(&mut self, len: usize, path: &str) -> Result<&'a [u8]> {
        let end = self
            .offset
            .checked_add(len)
            .ok_or_else(|| HybridError::ModelLoad {
                path: path.to_owned(),
                reason: "cursor overflow".into(),
            })?;
        if end > self.bytes.len() {
            return Err(HybridError::ModelLoad {
                path: path.to_owned(),
                reason: "unexpected EOF while parsing GGUF".into(),
            });
        }
        let slice = &self.bytes[self.offset..end];
        self.offset = end;
        Ok(slice)
    }

    fn read_u8(&mut self, path: &str) -> Result<u8> {
        Ok(self.read_exact(1, path)?[0])
    }

    fn read_u16(&mut self, path: &str) -> Result<u16> {
        let bytes = self.read_exact(2, path)?;
        Ok(u16::from_le_bytes([bytes[0], bytes[1]]))
    }

    fn read_u32(&mut self, path: &str) -> Result<u32> {
        let bytes = self.read_exact(4, path)?;
        Ok(u32::from_le_bytes(
            bytes.try_into().expect("slice length is fixed"),
        ))
    }

    fn read_u64(&mut self, path: &str) -> Result<u64> {
        let bytes = self.read_exact(8, path)?;
        Ok(u64::from_le_bytes(
            bytes.try_into().expect("slice length is fixed"),
        ))
    }

    fn read_i16(&mut self, path: &str) -> Result<i16> {
        Ok(self.read_u16(path)? as i16)
    }

    fn read_i32(&mut self, path: &str) -> Result<i32> {
        Ok(self.read_u32(path)? as i32)
    }

    fn read_i64(&mut self, path: &str) -> Result<i64> {
        Ok(self.read_u64(path)? as i64)
    }

    fn read_string(&mut self, path: &str) -> Result<String> {
        let len = self.read_u64(path)? as usize;
        let bytes = self.read_exact(len, path)?;
        String::from_utf8(bytes.to_vec()).map_err(|e| HybridError::ModelLoad {
            path: path.to_owned(),
            reason: format!("invalid UTF-8 in GGUF string: {e}"),
        })
    }

    fn read_numeric_as_u32(&mut self, value_type: u32, path: &str) -> Result<u32> {
        match value_type {
            GGUF_VALUE_TYPE_UINT8 => Ok(self.read_u8(path)? as u32),
            GGUF_VALUE_TYPE_INT8 => Ok(self.read_u8(path)? as i8 as i32 as u32),
            GGUF_VALUE_TYPE_UINT16 => Ok(self.read_u16(path)? as u32),
            GGUF_VALUE_TYPE_INT16 => Ok(self.read_i16(path)? as i32 as u32),
            GGUF_VALUE_TYPE_UINT32 => self.read_u32(path),
            GGUF_VALUE_TYPE_INT32 => Ok(self.read_i32(path)? as u32),
            GGUF_VALUE_TYPE_UINT64 => Ok(self.read_u64(path)? as u32),
            GGUF_VALUE_TYPE_INT64 => Ok(self.read_i64(path)? as u32),
            _ => Err(HybridError::UnsupportedFormat(format!(
                "GGUF numeric conversion from type {value_type} is not supported"
            ))),
        }
    }

    fn read_numeric_as_usize(&mut self, value_type: u32, path: &str) -> Result<usize> {
        Ok(self.read_numeric_as_u32(value_type, path)? as usize)
    }

    fn skip_value(&mut self, value_type: u32, path: &str) -> Result<()> {
        match value_type {
            GGUF_VALUE_TYPE_UINT8 | GGUF_VALUE_TYPE_INT8 | GGUF_VALUE_TYPE_BOOL => {
                self.read_exact(1, path)?;
            }
            GGUF_VALUE_TYPE_UINT16 | GGUF_VALUE_TYPE_INT16 => {
                self.read_exact(2, path)?;
            }
            GGUF_VALUE_TYPE_UINT32 | GGUF_VALUE_TYPE_INT32 | GGUF_VALUE_TYPE_FLOAT32 => {
                self.read_exact(4, path)?;
            }
            GGUF_VALUE_TYPE_UINT64 | GGUF_VALUE_TYPE_INT64 | GGUF_VALUE_TYPE_FLOAT64 => {
                self.read_exact(8, path)?;
            }
            GGUF_VALUE_TYPE_STRING => {
                let _ = self.read_string(path)?;
            }
            GGUF_VALUE_TYPE_ARRAY => {
                let nested_type = self.read_u32(path)?;
                let len = self.read_u64(path)? as usize;
                for _ in 0..len {
                    self.skip_value(nested_type, path)?;
                }
            }
            _ => {
                return Err(HybridError::UnsupportedFormat(format!(
                    "unsupported GGUF value type {value_type}"
                )));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::GpuAccelerator;
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

    fn build_test_gguf(tensors: Vec<(&str, Vec<usize>, u32, Vec<u8>)>, alignment: u32) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&GGUF_MAGIC);
        push_u32(&mut out, GGUF_VERSION);
        push_u64(&mut out, tensors.len() as u64);
        push_u64(&mut out, 3);
        push_kv_u32(&mut out, "general.alignment", alignment);
        push_kv_u32(&mut out, "general.file_type", 1);
        push_kv_u32(&mut out, "olmoe.expert_count", 64);

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
        let attn_q_payload = vec![0u8; OLMOE_HIDDEN * OLMOE_HIDDEN * 2];
        build_test_gguf(
            vec![
                (
                    ROUTING_TENSOR_CANDIDATES[0],
                    vec![EMBEDDING_DIM, OLMOE_NUM_EXPERTS],
                    GGML_TYPE_F32,
                    gate_payload,
                ),
                (
                    "blk.0.attn_q.weight",
                    vec![OLMOE_HIDDEN, OLMOE_HIDDEN],
                    GGML_TYPE_F16,
                    attn_q_payload,
                ),
            ],
            32,
        )
    }

    fn stub() -> OLMoE {
        OLMoE::load_with_mode("", 8, 1, OlmoeExecutionMode::StubUniform)
            .expect("stub load should succeed")
    }

    fn dense_sim_stub() -> OLMoE {
        OLMoE::load_with_mode("", 8, 2, OlmoeExecutionMode::DenseSim)
            .expect("dense sim stub load should succeed")
    }

    fn spiking_sim_stub() -> OLMoE {
        OLMoE::load_with_mode("", 8, 2, OlmoeExecutionMode::SpikingSim)
            .expect("spiking sim stub load should succeed")
    }

    #[test]
    fn test_stub_mode_loads() {
        let model = stub();
        assert!(!model.is_loaded());
        assert_eq!(model.quantization(), "stub");
    }

    #[test]
    fn test_stub_forward_shape() {
        let mut model = stub();
        let embedding = vec![0.0f32; EMBEDDING_DIM];
        let out = model.forward(&embedding).unwrap();
        assert_eq!(out.expert_weights.len(), 8);
        assert_eq!(out.selected_experts.len(), 1);
        assert_eq!(out.hidden.len(), EMBEDDING_DIM);
    }

    #[test]
    fn test_stub_forward_uniform_weights() {
        let mut model = stub();
        let embedding = vec![0.1f32; EMBEDDING_DIM];
        let out = model.forward(&embedding).unwrap();
        for w in &out.expert_weights {
            assert!((*w - 0.125).abs() < 1e-5, "expected uniform 1/8, got {w}");
        }
    }

    #[test]
    fn test_input_length_mismatch() {
        let mut model = stub();
        let bad_embedding = vec![0.0f32; 64];
        assert!(model.forward(&bad_embedding).is_err());
    }

    #[test]
    fn test_dense_sim_in_stub_mode_has_valid_routing() {
        let mut model = dense_sim_stub();
        let embedding: Vec<f32> = (0..EMBEDDING_DIM)
            .map(|i| (i as f32 / EMBEDDING_DIM as f32) * 0.1)
            .collect();
        let out = model.forward(&embedding).unwrap();
        assert_eq!(out.selected_experts.len(), 2);
        let weight_sum: f32 = out.expert_weights.iter().sum();
        assert!(
            (weight_sum - 1.0).abs() < 1e-5,
            "expert weights must sum to 1, got {weight_sum}"
        );
        assert_eq!(out.hidden.len(), EMBEDDING_DIM);
    }

    #[test]
    fn test_spiking_sim_persists_state_and_can_fire() {
        let mut model = spiking_sim_stub();
        let embedding = vec![1.0f32; EMBEDDING_DIM];

        let first = model.forward(&embedding).unwrap();
        assert!(model.expert_membranes.iter().any(|&v| v != 0.0));

        let mut fired = first.hidden.iter().any(|&v| v != 0.0);
        for _ in 0..32 {
            let out = model.forward(&embedding).unwrap();
            if out.hidden.iter().any(|&v| v != 0.0) {
                fired = true;
                break;
            }
        }

        assert!(
            fired,
            "spiking sim should eventually emit ternary hidden events"
        );
    }

    #[test]
    fn test_spiking_sim_reset_clears_state() {
        let mut model = spiking_sim_stub();
        let embedding = vec![1.0f32; EMBEDDING_DIM];

        let _ = model.forward(&embedding).unwrap();
        assert!(model.expert_membranes.iter().any(|&v| v != 0.0));

        model.reset_state();

        assert!(model.expert_membranes.iter().all(|&v| v == 0.0));
        assert!(model.hidden_membranes.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_parse_checkpoint_layout_preserves_tensor_offsets() {
        let bytes = build_test_gguf(
            vec![
                (
                    ROUTING_TENSOR_CANDIDATES[0],
                    vec![EMBEDDING_DIM, OLMOE_NUM_EXPERTS],
                    GGML_TYPE_F32,
                    vec![0u8; EMBEDDING_DIM * OLMOE_NUM_EXPERTS * 4],
                ),
                ("demo.weight", vec![2, 2], GGML_TYPE_F32, vec![0u8; 16]),
            ],
            64,
        );
        let parsed = parse_checkpoint_layout(&bytes, "test.gguf").unwrap();
        let routing = parsed.tensors.get(ROUTING_TENSOR_CANDIDATES[0]).unwrap();
        let tensor = parsed.tensors.get("demo.weight").unwrap();
        assert_eq!(
            tensor.relative_offset,
            routing.n_elements * std::mem::size_of::<f32>()
        );
        assert_eq!(tensor.absolute_offset % 64, 0);
        assert_eq!(tensor.n_elements, 4);
    }

    #[test]
    fn test_probe_and_map_rejects_missing_routing_tensor() {
        let bytes = build_test_gguf(
            vec![(
                "blk.0.attn_q.weight",
                vec![OLMOE_HIDDEN, OLMOE_HIDDEN],
                GGML_TYPE_F16,
                vec![0u8; OLMOE_HIDDEN * OLMOE_HIDDEN * 2],
            )],
            32,
        );
        let path = write_temp_file(&bytes, "missing-routing");
        let err = OLMoE::probe_and_map(path.to_str().unwrap()).unwrap_err();
        assert!(matches!(err, HybridError::UnsupportedFormat(_)));
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn test_probe_and_map_allows_non_f16_synapse_tensor() {
        let gate_payload = vec![0u8; EMBEDDING_DIM * OLMOE_NUM_EXPERTS * 4];
        let bytes = build_test_gguf(
            vec![
                (
                    ROUTING_TENSOR_CANDIDATES[0],
                    vec![EMBEDDING_DIM, OLMOE_NUM_EXPERTS],
                    GGML_TYPE_F32,
                    gate_payload,
                ),
                (
                    "blk.0.attn_q.weight",
                    vec![OLMOE_HIDDEN, OLMOE_HIDDEN],
                    GGML_TYPE_F32,
                    vec![0u8; OLMOE_HIDDEN * OLMOE_HIDDEN * 4],
                ),
            ],
            32,
        );
        let path = write_temp_file(&bytes, "wrong-type");
        let (_, _, mut checkpoint) = OLMoE::probe_and_map(path.to_str().unwrap()).unwrap();
        let err = checkpoint
            .registered_f16_tensor("blk.0.attn_q.weight", path.to_str().unwrap())
            .unwrap_err();
        assert!(matches!(err, HybridError::UnsupportedFormat(_)));
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn test_dense_sim_uses_real_gate_weights() {
        let mut gate = vec![0.0f32; EMBEDDING_DIM * OLMOE_NUM_EXPERTS];
        for (expert, gate_value) in gate.iter_mut().take(OLMOE_NUM_EXPERTS).enumerate() {
            *gate_value = if expert == 0 { 8.0 } else { -8.0 };
        }
        let gate_bytes: Vec<u8> = gate.iter().flat_map(|value| value.to_le_bytes()).collect();
        let path = write_temp_file(&build_real_size_checkpoint(gate_bytes), "dense-real");

        let mut model =
            OLMoE::load_with_mode(path.to_str().unwrap(), 8, 2, OlmoeExecutionMode::DenseSim)
                .unwrap();
        let mut embedding = vec![0.0f32; EMBEDDING_DIM];
        embedding[0] = 1.0;
        let out = model.forward(&embedding).unwrap();
        assert_eq!(out.selected_experts[0], 0);
        let weight_sum: f32 = out.expert_weights.iter().sum();
        assert!((weight_sum - 1.0).abs() < 1e-5);

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn test_spiking_real_gate_path_accumulates_state() {
        let mut gate = vec![0.0f32; EMBEDDING_DIM * OLMOE_NUM_EXPERTS];
        for (expert, gate_value) in gate.iter_mut().take(OLMOE_NUM_EXPERTS).enumerate() {
            *gate_value = if expert == 0 { 8.0 } else { -8.0 };
        }
        let gate_bytes: Vec<u8> = gate.iter().flat_map(|value| value.to_le_bytes()).collect();
        let path = write_temp_file(&build_real_size_checkpoint(gate_bytes), "spiking-real");

        let mut model =
            OLMoE::load_with_mode(path.to_str().unwrap(), 8, 2, OlmoeExecutionMode::SpikingSim)
                .unwrap();
        let mut embedding = vec![0.0f32; EMBEDDING_DIM];
        embedding[0] = 1.0;
        for _ in 0..8 {
            let _ = model.forward(&embedding).unwrap();
        }
        assert!(model.has_state_activity());
        model.reset_state();
        assert!(!model.has_state_activity());

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn test_removed_old_weight_symbol() {
        let source = std::fs::read_to_string(file!()).unwrap();
        let pattern = ["pseudo", "_weight"].concat();
        assert!(!source.contains(&pattern));
    }

    #[test]
    fn test_real_checkpoint_probe_via_env() {
        let Some(path) = std::env::var("OLMOE_PATH").ok() else {
            return;
        };

        let (metadata, routing_tensor_name, checkpoint) = OLMoE::probe_and_map(&path).unwrap();
        assert_eq!(metadata.hidden_size, 2048);
        assert_eq!(metadata.num_experts, 64);
        assert_eq!(routing_tensor_name, ROUTING_TENSOR_CANDIDATES[0]);
        let routing = checkpoint
            .tensor_info(ROUTING_TENSOR_CANDIDATES[0], &path)
            .unwrap();
        assert_eq!(routing.dims, vec![2048, 64]);
        let synapse = checkpoint
            .tensor_info("blk.0.attn_q.weight", &path)
            .unwrap();
        assert_eq!(synapse.dims, vec![2048, 2048]);
    }

    #[test]
    fn test_registered_gpu_upload_via_env() {
        let Some(path) = std::env::var("OLMOE_PATH").ok() else {
            return;
        };
        if !crate::gpu::GpuContext::is_available() {
            return;
        }

        let mut accelerator = GpuAccelerator::new();
        if !accelerator.is_ready() {
            return;
        }

        let mut model = OLMoE::load_with_mode(&path, 8, 1, OlmoeExecutionMode::DenseSim).unwrap();
        accelerator.ensure_temporal_state(OLMOE_HIDDEN).unwrap();
        let weights = model
            .registered_gpu_synapse_weights("blk.0.attn_q.weight")
            .unwrap();
        accelerator
            .load_synapse_weights_f16_registered("env::blk.0.attn_q.weight", weights)
            .unwrap();
        assert_eq!(
            accelerator.synapse_signature(),
            Some("env::blk.0.attn_q.weight")
        );
    }
}
