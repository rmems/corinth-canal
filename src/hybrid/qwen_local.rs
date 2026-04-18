use super::olmoe::{OlmoeCheckpointMetadata, OlmoeMetadata};
use crate::error::{HybridError, Result};
use crate::types::EMBEDDING_DIM;
use memmap2::{Mmap, MmapOptions};
use serde::Deserialize;
use std::collections::{HashMap, HashSet};
use std::fs::OpenOptions;
use std::path::{Path, PathBuf};

pub(crate) const QWEN_ARCHITECTURE: &str = "qwen2_moe";
pub(crate) const QWEN_TOKEN_EMBED_TENSOR_NAME: &str = "model.embed_tokens.weight";
pub(crate) const QWEN_ROUTING_TENSOR_NAME: &str = "model.layers.0.mlp.gate.weight";
const SAFETENSOR_DTYPE_F16: &str = "F16";

#[derive(Debug)]
pub(crate) struct QwenLocalCheckpoint {
    shards: HashMap<String, MappedSafetensorShard>,
    embed_tensor: SafetensorTensorInfo,
    gate_tensor: SafetensorTensorInfo,
    decoded_gate_weights: Vec<f32>,
}

#[derive(Debug)]
struct MappedSafetensorShard {
    mmap: Mmap,
}

#[derive(Debug, Clone)]
struct SafetensorTensorInfo {
    shard_filename: String,
    dtype: String,
    shape: Vec<usize>,
    absolute_offset: usize,
    byte_len: usize,
}

#[derive(Debug, Deserialize)]
struct QwenConfig {
    model_type: String,
    hidden_size: usize,
    num_hidden_layers: usize,
    num_experts: usize,
    num_experts_per_tok: usize,
    #[serde(default)]
    quantization_config: Option<QwenQuantizationConfig>,
}

#[derive(Debug, Deserialize)]
struct QwenQuantizationConfig {
    #[serde(default)]
    quant_method: Option<String>,
    #[serde(default)]
    bits: Option<usize>,
}

#[derive(Debug, Deserialize)]
struct SafetensorIndex {
    weight_map: HashMap<String, String>,
}

#[derive(Debug, Deserialize)]
struct RawSafetensorHeaderEntry {
    #[serde(default)]
    dtype: Option<String>,
    #[serde(default)]
    shape: Option<Vec<usize>>,
    #[serde(default)]
    data_offsets: Option<[usize; 2]>,
}

pub(crate) fn load_qwen_local_checkpoint(
    root_dir: &str,
) -> Result<(OlmoeMetadata, String, QwenLocalCheckpoint)> {
    let root = ensure_checkpoint_dir(root_dir)?;
    let config = read_json::<QwenConfig>(&root.join("config.json"))?;
    validate_qwen_config(&config, root_dir)?;

    let index = read_json::<SafetensorIndex>(&root.join("model.safetensors.index.json"))?;
    let required_tensors = [QWEN_TOKEN_EMBED_TENSOR_NAME, QWEN_ROUTING_TENSOR_NAME];

    let mut required_shards = HashSet::new();
    for tensor_name in required_tensors {
        let shard_filename =
            index
                .weight_map
                .get(tensor_name)
                .ok_or_else(|| HybridError::MissingTensor {
                    name: tensor_name.to_owned(),
                    path: root_dir.to_owned(),
                })?;
        required_shards.insert(shard_filename.clone());
    }

    let mut shards = HashMap::new();
    for shard_filename in required_shards {
        let shard_path = root.join(&shard_filename);
        let file = OpenOptions::new()
            .read(true)
            .open(&shard_path)
            .map_err(|err| HybridError::ModelLoad {
                path: shard_path.display().to_string(),
                reason: err.to_string(),
            })?;
        let mmap =
            unsafe { MmapOptions::new().map(&file) }.map_err(|err| HybridError::ModelLoad {
                path: shard_path.display().to_string(),
                reason: format!("mmap failed: {err}"),
            })?;
        shards.insert(shard_filename, MappedSafetensorShard { mmap });
    }

    let embed_tensor = resolve_tensor_info(
        root_dir,
        QWEN_TOKEN_EMBED_TENSOR_NAME,
        index
            .weight_map
            .get(QWEN_TOKEN_EMBED_TENSOR_NAME)
            .expect("validated above"),
        shards
            .get(
                index
                    .weight_map
                    .get(QWEN_TOKEN_EMBED_TENSOR_NAME)
                    .expect("validated above"),
            )
            .expect("mapped above"),
    )?;
    validate_embedding_tensor(root_dir, &embed_tensor, config.hidden_size)?;

    let gate_tensor = resolve_tensor_info(
        root_dir,
        QWEN_ROUTING_TENSOR_NAME,
        index
            .weight_map
            .get(QWEN_ROUTING_TENSOR_NAME)
            .expect("validated above"),
        shards
            .get(
                index
                    .weight_map
                    .get(QWEN_ROUTING_TENSOR_NAME)
                    .expect("validated above"),
            )
            .expect("mapped above"),
    )?;
    validate_gate_tensor(
        root_dir,
        &gate_tensor,
        config.hidden_size,
        config.num_experts,
    )?;

    let decoded_gate_weights = decode_f16_tensor(
        tensor_bytes(
            shards
                .get(&gate_tensor.shard_filename)
                .expect("mapped above"),
            &gate_tensor,
            root_dir,
        )?,
        &gate_tensor,
        root_dir,
    )?;

    let metadata = OlmoeMetadata {
        architecture: config.model_type,
        hidden_size: EMBEDDING_DIM,
        source_hidden_size: config.hidden_size,
        num_layers: config.num_hidden_layers,
        num_experts: config.num_experts,
        expert_used_count: Some(config.num_experts_per_tok),
        quantization: qwen_quantization_label(config.quantization_config.as_ref()),
        checkpoint_format: "safetensors-gptq".into(),
    };

    Ok((
        metadata,
        QWEN_ROUTING_TENSOR_NAME.to_owned(),
        QwenLocalCheckpoint {
            shards,
            embed_tensor,
            gate_tensor,
            decoded_gate_weights,
        },
    ))
}

pub(crate) fn probe_qwen_local_checkpoint(root_dir: &str) -> Result<OlmoeCheckpointMetadata> {
    let (metadata, routing_tensor_name, _) = load_qwen_local_checkpoint(root_dir)?;
    Ok(OlmoeCheckpointMetadata {
        architecture: metadata.architecture,
        hidden_size: metadata.hidden_size,
        source_hidden_size: metadata.source_hidden_size,
        num_layers: metadata.num_layers,
        num_experts: metadata.num_experts,
        expert_used_count: metadata.expert_used_count,
        quantization: metadata.quantization,
        routing_tensor_name,
        checkpoint_format: metadata.checkpoint_format,
    })
}

impl QwenLocalCheckpoint {
    pub(crate) fn extract_token_embedding(
        &self,
        path: &str,
        token_id: usize,
        hidden_size: usize,
    ) -> Result<Vec<f32>> {
        let info = &self.embed_tensor;
        let raw = tensor_bytes(
            self.shards
                .get(&info.shard_filename)
                .expect("embed shard should stay mapped"),
            info,
            path,
        )?;
        let d0 = info.shape[0];
        let d1 = info.shape.get(1).copied().unwrap_or(0);

        if d1 == hidden_size {
            if token_id >= d0 {
                return Err(HybridError::InputLengthMismatch {
                    expected: d0,
                    got: token_id,
                });
            }
            let row_bytes = hidden_size
                .checked_mul(2)
                .ok_or_else(|| HybridError::ModelLoad {
                    path: path.to_owned(),
                    reason: "embedding row byte length overflow".into(),
                })?;
            let byte_start =
                token_id
                    .checked_mul(row_bytes)
                    .ok_or_else(|| HybridError::ModelLoad {
                        path: path.to_owned(),
                        reason: "embedding row start overflow".into(),
                    })?;
            let byte_end = byte_start + row_bytes;
            return decode_f16_slice(&raw[byte_start..byte_end], hidden_size, path);
        }

        if d0 == hidden_size {
            if token_id >= d1 {
                return Err(HybridError::InputLengthMismatch {
                    expected: d1,
                    got: token_id,
                });
            }
            let mut out = Vec::with_capacity(hidden_size);
            for dim in 0..hidden_size {
                let idx = (dim * d1 + token_id) * 2;
                out.push(f16_to_f32(u16::from_le_bytes([raw[idx], raw[idx + 1]])));
            }
            return Ok(out);
        }

        Err(HybridError::UnsupportedFormat(format!(
            "tensor '{QWEN_TOKEN_EMBED_TENSOR_NAME}' has unexpected dimensions {:?}",
            info.shape
        )))
    }

    pub(crate) fn compute_gate_scores(
        &self,
        path: &str,
        embedding: &[f32],
        num_experts: usize,
        hidden_size: usize,
    ) -> Result<Vec<f32>> {
        let d0 = self.gate_tensor.shape[0];
        let d1 = self.gate_tensor.shape.get(1).copied().unwrap_or(0);
        let mut scores = Vec::with_capacity(num_experts);

        if d0 >= num_experts && d1 == hidden_size {
            for expert_id in 0..num_experts {
                let row_start = expert_id * d1;
                let mut score = 0.0f32;
                for (weight, value) in self.decoded_gate_weights[row_start..row_start + d1]
                    .iter()
                    .zip(embedding.iter())
                {
                    score += *weight * *value;
                }
                scores.push(score);
            }
            return Ok(scores);
        }

        if d0 == hidden_size && d1 >= num_experts {
            for expert_id in 0..num_experts {
                let mut score = 0.0f32;
                for (dim, value) in embedding.iter().enumerate() {
                    score += self.decoded_gate_weights[dim * d1 + expert_id] * *value;
                }
                scores.push(score);
            }
            return Ok(scores);
        }

        Err(HybridError::UnsupportedFormat(format!(
            "tensor '{QWEN_ROUTING_TENSOR_NAME}' has unsupported orientation {:?} in '{path}'",
            self.gate_tensor.shape
        )))
    }
}

fn ensure_checkpoint_dir(root_dir: &str) -> Result<PathBuf> {
    let path = PathBuf::from(root_dir);
    if !path.is_dir() {
        return Err(HybridError::ModelLoad {
            path: root_dir.to_owned(),
            reason: "local checkpoint directory does not exist".into(),
        });
    }
    Ok(path)
}

fn read_json<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<T> {
    let bytes = std::fs::read(path).map_err(|err| HybridError::ModelLoad {
        path: path.display().to_string(),
        reason: err.to_string(),
    })?;
    serde_json::from_slice(&bytes).map_err(|err| HybridError::ModelLoad {
        path: path.display().to_string(),
        reason: format!("invalid JSON: {err}"),
    })
}

fn validate_qwen_config(config: &QwenConfig, path: &str) -> Result<()> {
    if config.model_type != QWEN_ARCHITECTURE {
        return Err(HybridError::UnsupportedFormat(format!(
            "local checkpoint '{path}' has architecture '{}' but only '{QWEN_ARCHITECTURE}' is supported for the safetensors GPTQ path",
            config.model_type
        )));
    }
    if config.hidden_size != EMBEDDING_DIM {
        return Err(HybridError::UnsupportedFormat(format!(
            "local checkpoint '{path}' hidden_size={} does not match required SNN width {}",
            config.hidden_size, EMBEDDING_DIM
        )));
    }
    if config.num_experts == 0 || config.num_experts_per_tok == 0 {
        return Err(HybridError::UnsupportedFormat(format!(
            "local checkpoint '{path}' must expose non-zero expert counts"
        )));
    }
    Ok(())
}

fn qwen_quantization_label(config: Option<&QwenQuantizationConfig>) -> String {
    let Some(config) = config else {
        return "gptq".into();
    };
    match (config.quant_method.as_deref(), config.bits) {
        (Some(method), Some(bits)) => format!("{method}-int{bits}"),
        (Some(method), None) => method.to_owned(),
        (None, Some(bits)) => format!("gptq-int{bits}"),
        (None, None) => "gptq".into(),
    }
}

fn parse_safetensor_header(
    mmap: &Mmap,
    path: &str,
) -> Result<(usize, HashMap<String, RawSafetensorHeaderEntry>)> {
    if mmap.len() < 8 {
        return Err(HybridError::ModelLoad {
            path: path.to_owned(),
            reason: "safetensors shard is truncated".into(),
        });
    }

    let header_len = u64::from_le_bytes(
        mmap[0..8]
            .try_into()
            .expect("slice length is fixed for safetensors header"),
    ) as usize;
    let header_start = 8usize;
    let header_end =
        header_start
            .checked_add(header_len)
            .ok_or_else(|| HybridError::ModelLoad {
                path: path.to_owned(),
                reason: "safetensors header length overflow".into(),
            })?;
    if header_end > mmap.len() {
        return Err(HybridError::ModelLoad {
            path: path.to_owned(),
            reason: "safetensors header extends beyond file".into(),
        });
    }

    let header = serde_json::from_slice::<HashMap<String, RawSafetensorHeaderEntry>>(
        &mmap[header_start..header_end],
    )
    .map_err(|err| HybridError::ModelLoad {
        path: path.to_owned(),
        reason: format!("invalid safetensors header JSON: {err}"),
    })?;
    Ok((header_end, header))
}

fn resolve_tensor_info(
    root_dir: &str,
    tensor_name: &str,
    shard_filename: &str,
    shard: &MappedSafetensorShard,
) -> Result<SafetensorTensorInfo> {
    let shard_path = Path::new(root_dir).join(shard_filename);
    let shard_path_str = shard_path.display().to_string();
    let (data_start, header) = parse_safetensor_header(&shard.mmap, &shard_path_str)?;
    let entry = header
        .get(tensor_name)
        .ok_or_else(|| HybridError::MissingTensor {
            name: tensor_name.to_owned(),
            path: shard_path_str.clone(),
        })?;
    let dtype = entry.dtype.clone().ok_or_else(|| HybridError::ModelLoad {
        path: shard_path_str.clone(),
        reason: format!("tensor '{tensor_name}' is missing dtype"),
    })?;
    let shape = entry.shape.clone().ok_or_else(|| HybridError::ModelLoad {
        path: shard_path_str.clone(),
        reason: format!("tensor '{tensor_name}' is missing shape"),
    })?;
    let [relative_start, relative_end] =
        entry.data_offsets.ok_or_else(|| HybridError::ModelLoad {
            path: shard_path_str.clone(),
            reason: format!("tensor '{tensor_name}' is missing data_offsets"),
        })?;
    if relative_end < relative_start {
        return Err(HybridError::ModelLoad {
            path: shard_path_str.clone(),
            reason: format!("tensor '{tensor_name}' has descending data_offsets"),
        });
    }

    let absolute_offset =
        data_start
            .checked_add(relative_start)
            .ok_or_else(|| HybridError::ModelLoad {
                path: shard_path_str.clone(),
                reason: format!("tensor '{tensor_name}' absolute offset overflow"),
            })?;
    let absolute_end =
        data_start
            .checked_add(relative_end)
            .ok_or_else(|| HybridError::ModelLoad {
                path: shard_path_str.clone(),
                reason: format!("tensor '{tensor_name}' absolute end overflow"),
            })?;
    if absolute_end > shard.mmap.len() {
        return Err(HybridError::ModelLoad {
            path: shard_path_str.clone(),
            reason: format!("tensor '{tensor_name}' extends beyond mapped shard"),
        });
    }

    let n_elements = shape
        .iter()
        .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
        .ok_or_else(|| HybridError::ModelLoad {
            path: shard_path_str.clone(),
            reason: format!("tensor '{tensor_name}' element count overflow"),
        })?;
    let byte_len = relative_end - relative_start;
    let expected_len = n_elements
        .checked_mul(dtype_size_bytes(&dtype, &shard_path_str, tensor_name)?)
        .ok_or_else(|| HybridError::ModelLoad {
            path: shard_path_str.clone(),
            reason: format!("tensor '{tensor_name}' byte length overflow"),
        })?;
    if expected_len != byte_len {
        return Err(HybridError::UnsupportedFormat(format!(
            "tensor '{tensor_name}' in '{shard_path_str}' expected {expected_len} bytes from {n_elements} {dtype} values, got {byte_len}"
        )));
    }

    Ok(SafetensorTensorInfo {
        shard_filename: shard_filename.to_owned(),
        dtype,
        shape,
        absolute_offset,
        byte_len,
    })
}

fn validate_embedding_tensor(
    path: &str,
    info: &SafetensorTensorInfo,
    hidden_size: usize,
) -> Result<()> {
    if info.dtype != SAFETENSOR_DTYPE_F16 {
        return Err(HybridError::UnsupportedFormat(format!(
            "tensor '{QWEN_TOKEN_EMBED_TENSOR_NAME}' in '{path}' must be {SAFETENSOR_DTYPE_F16}, got {}",
            info.dtype
        )));
    }
    if info.shape.len() != 2 {
        return Err(HybridError::UnsupportedFormat(format!(
            "tensor '{QWEN_TOKEN_EMBED_TENSOR_NAME}' must be rank-2, got {:?}",
            info.shape
        )));
    }
    let d0 = info.shape[0];
    let d1 = info.shape[1];
    if d0 != hidden_size && d1 != hidden_size {
        return Err(HybridError::UnsupportedFormat(format!(
            "tensor '{QWEN_TOKEN_EMBED_TENSOR_NAME}' must expose hidden axis {hidden_size}, got {:?}",
            info.shape
        )));
    }
    Ok(())
}

fn validate_gate_tensor(
    path: &str,
    info: &SafetensorTensorInfo,
    hidden_size: usize,
    num_experts: usize,
) -> Result<()> {
    if info.dtype != SAFETENSOR_DTYPE_F16 {
        return Err(HybridError::UnsupportedFormat(format!(
            "tensor '{QWEN_ROUTING_TENSOR_NAME}' in '{path}' must be {SAFETENSOR_DTYPE_F16}, got {}",
            info.dtype
        )));
    }
    if info.shape.len() != 2 {
        return Err(HybridError::UnsupportedFormat(format!(
            "tensor '{QWEN_ROUTING_TENSOR_NAME}' must be rank-2, got {:?}",
            info.shape
        )));
    }
    let matches = (info.shape[0] == num_experts && info.shape[1] == hidden_size)
        || (info.shape[0] == hidden_size && info.shape[1] == num_experts);
    if !matches {
        return Err(HybridError::UnsupportedFormat(format!(
            "tensor '{QWEN_ROUTING_TENSOR_NAME}' must expose expert axis {num_experts} and hidden axis {hidden_size}, got {:?}",
            info.shape
        )));
    }
    Ok(())
}

fn tensor_bytes<'a>(
    shard: &'a MappedSafetensorShard,
    info: &SafetensorTensorInfo,
    path: &str,
) -> Result<&'a [u8]> {
    let byte_end = info.absolute_offset + info.byte_len;
    if byte_end > shard.mmap.len() {
        return Err(HybridError::ModelLoad {
            path: path.to_owned(),
            reason: "tensor extends beyond mapped safetensors shard".into(),
        });
    }
    Ok(&shard.mmap[info.absolute_offset..byte_end])
}

fn decode_f16_tensor(raw: &[u8], info: &SafetensorTensorInfo, path: &str) -> Result<Vec<f32>> {
    decode_f16_slice(raw, element_count(&info.shape, path)?, path)
}

fn decode_f16_slice(raw: &[u8], len: usize, path: &str) -> Result<Vec<f32>> {
    let expected_bytes = len.checked_mul(2).ok_or_else(|| HybridError::ModelLoad {
        path: path.to_owned(),
        reason: "F16 slice byte length overflow".into(),
    })?;
    if raw.len() < expected_bytes {
        return Err(HybridError::ModelLoad {
            path: path.to_owned(),
            reason: format!(
                "F16 slice is truncated: expected {expected_bytes} bytes, got {}",
                raw.len()
            ),
        });
    }
    Ok(raw[..expected_bytes]
        .chunks_exact(2)
        .map(|pair| f16_to_f32(u16::from_le_bytes([pair[0], pair[1]])))
        .collect())
}

fn element_count(shape: &[usize], path: &str) -> Result<usize> {
    shape
        .iter()
        .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
        .ok_or_else(|| HybridError::ModelLoad {
            path: path.to_owned(),
            reason: "tensor element count overflow".into(),
        })
}

fn dtype_size_bytes(dtype: &str, path: &str, tensor_name: &str) -> Result<usize> {
    match dtype {
        SAFETENSOR_DTYPE_F16 => Ok(2),
        other => Err(HybridError::UnsupportedFormat(format!(
            "tensor '{tensor_name}' in '{path}' has unsupported safetensors dtype '{other}'"
        ))),
    }
}

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hybrid::olmoe::OLMoE;
    use crate::types::OlmoeExecutionMode;
    use serde_json::json;
    use std::fs::File;
    use std::io::Write;

    fn unique_temp_dir(label: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "corinth_canal_{label}_{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn write_json(path: &Path, value: serde_json::Value) {
        std::fs::write(path, serde_json::to_vec_pretty(&value).unwrap()).unwrap();
    }

    fn write_safetensor_shard(path: &Path, tensors: Vec<(&str, Vec<usize>, Vec<u8>)>) {
        let mut header = serde_json::Map::new();
        let mut data = Vec::new();
        let mut offset = 0usize;

        for (name, shape, payload) in tensors {
            let end = offset + payload.len();
            header.insert(
                name.to_owned(),
                json!({
                    "dtype": "F16",
                    "shape": shape,
                    "data_offsets": [offset, end],
                }),
            );
            data.extend_from_slice(&payload);
            offset = end;
        }

        let header_bytes = serde_json::to_vec(&header).unwrap();
        let mut file = File::create(path).unwrap();
        file.write_all(&(header_bytes.len() as u64).to_le_bytes())
            .unwrap();
        file.write_all(&header_bytes).unwrap();
        file.write_all(&data).unwrap();
    }

    fn repeat_f16(bits: u16, count: usize) -> Vec<u8> {
        let mut out = Vec::with_capacity(count * 2);
        for _ in 0..count {
            out.extend_from_slice(&bits.to_le_bytes());
        }
        out
    }

    fn build_test_qwen_checkpoint() -> PathBuf {
        let dir = unique_temp_dir("qwen_local");

        write_json(
            &dir.join("config.json"),
            json!({
                "model_type": "qwen2_moe",
                "hidden_size": 2048,
                "num_hidden_layers": 24,
                "num_experts": 60,
                "num_experts_per_tok": 4,
                "quantization_config": {
                    "quant_method": "gptq",
                    "bits": 4
                }
            }),
        );

        let vocab_rows = 4usize;
        let mut embed_payload = Vec::new();
        for token_id in 0..vocab_rows {
            let bits = match token_id {
                0 => 0x0000,
                1 => 0x3c00,
                2 => 0x4000,
                _ => 0x4200,
            };
            embed_payload.extend_from_slice(&repeat_f16(bits, EMBEDDING_DIM));
        }

        let mut gate_payload = Vec::new();
        for expert_id in 0..60usize {
            let head_bits: u16 = match expert_id {
                3 => 0x4000,
                1 => 0x3c00,
                _ => 0xbc00,
            };
            gate_payload.extend_from_slice(&head_bits.to_le_bytes());
            gate_payload.extend_from_slice(&repeat_f16(0x0000, EMBEDDING_DIM - 1));
        }

        write_safetensor_shard(
            &dir.join("model-00001-of-00001.safetensors"),
            vec![
                (
                    QWEN_TOKEN_EMBED_TENSOR_NAME,
                    vec![vocab_rows, EMBEDDING_DIM],
                    embed_payload,
                ),
                (
                    QWEN_ROUTING_TENSOR_NAME,
                    vec![60, EMBEDDING_DIM],
                    gate_payload,
                ),
            ],
        );

        write_json(
            &dir.join("model.safetensors.index.json"),
            json!({
                "metadata": { "total_size": 0 },
                "weight_map": {
                    QWEN_TOKEN_EMBED_TENSOR_NAME: "model-00001-of-00001.safetensors",
                    QWEN_ROUTING_TENSOR_NAME: "model-00001-of-00001.safetensors"
                }
            }),
        );

        dir
    }

    #[test]
    fn test_probe_qwen_local_checkpoint_reports_expected_metadata() {
        let dir = build_test_qwen_checkpoint();
        let metadata = probe_qwen_local_checkpoint(dir.to_str().unwrap()).unwrap();

        assert_eq!(metadata.architecture, "qwen2_moe");
        assert_eq!(metadata.hidden_size, EMBEDDING_DIM);
        assert_eq!(metadata.source_hidden_size, EMBEDDING_DIM);
        assert_eq!(metadata.num_experts, 60);
        assert_eq!(metadata.expert_used_count, Some(4));
        assert_eq!(metadata.quantization, "gptq-int4");
        assert_eq!(metadata.routing_tensor_name, QWEN_ROUTING_TENSOR_NAME);
        assert_eq!(metadata.checkpoint_format, "safetensors-gptq");

        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn test_qwen_local_backend_extracts_embeddings_and_routes() {
        let dir = build_test_qwen_checkpoint();
        let mut model = OLMoE::load_with_mode_and_local(
            "",
            dir.to_str().unwrap(),
            8,
            2,
            OlmoeExecutionMode::DenseSim,
        )
        .unwrap();

        let embedding = model.extract_token_embedding(2).unwrap();
        assert_eq!(embedding.len(), EMBEDDING_DIM);
        assert!(
            embedding
                .iter()
                .take(16)
                .all(|value| (*value - 2.0).abs() < 1e-5)
        );

        let mut routing_embedding = vec![0.0f32; EMBEDDING_DIM];
        routing_embedding[0] = 1.0;
        let output = model.forward(&routing_embedding).unwrap();
        assert_eq!(output.selected_experts[0], 3);
        assert_eq!(output.selected_experts[1], 1);
        assert_eq!(model.routing_tensor_name(), QWEN_ROUTING_TENSOR_NAME);

        let _ = std::fs::remove_dir_all(dir);
    }
}
