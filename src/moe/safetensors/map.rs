// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Memory-mapped Safetensors loading and tensor extraction.

use super::super::checkpoint::f16_to_f32;
use super::config::HfConfig;
use super::json::parse_json_rejecting_duplicate_keys;
use super::manifest::{MAX_HEADER_BYTES, find_index_file, list_safetensors_files, read_index};
use super::model_load;
use super::paths::index_shard_path;
use super::validate::{dtype_size_bytes, expected_tensor_byte_size};
use crate::error::{HybridError, Result};
use memmap2::Mmap;
use serde_json::Value;
use std::fs::{self, File};
use std::io::Read;
use std::path::{Path, PathBuf};

// ── Tensor loading backend ──────────────────────────────────────────────

/// Metadata extracted from a Hugging Face `config.json` for adapter
/// resolution.
#[derive(Debug, Clone)]
pub struct SafetensorsMetadata {
    pub architecture: String,
    pub hidden_size: usize,
    pub num_layers: usize,
    pub num_experts: usize,
    pub expert_used_count: usize,
    pub vocab_size: usize,
}

/// Memory-mapped Safetensors checkpoint with indexed tensor access.
#[derive(Debug)]
pub struct MappedSafetensorsCheckpoint {
    shards: Vec<MappedShard>,
    tensor_map: std::collections::BTreeMap<String, TensorLocation>,
    pub metadata: SafetensorsMetadata,
}

#[derive(Debug, Clone)]
struct TensorLocation {
    shard_idx: usize,
    dtype: String,
    shape: Vec<usize>,
    data_offsets: [usize; 2],
}

#[derive(Debug)]
struct MappedShard {
    // Retained for debug introspection; not read at runtime.
    #[allow(dead_code)]
    path: PathBuf,
    mmap: Mmap,
    header_len: u64,
}

fn build_indexed_shard_map(
    root: &Path,
    index_path: &Path,
) -> Result<(
    Vec<PathBuf>,
    std::collections::BTreeMap<String, TensorLocation>,
)> {
    let raw_index = read_index(index_path)?;
    let mut shard_paths: Vec<PathBuf> = Vec::new();
    let mut tensor_map: std::collections::BTreeMap<String, TensorLocation> =
        std::collections::BTreeMap::new();
    let mut shard_index_by_path: std::collections::BTreeMap<PathBuf, usize> =
        std::collections::BTreeMap::new();
    for (tensor_name, relative) in &raw_index.weight_map {
        let shard_path = index_shard_path(root, index_path, relative)?;
        let shard_idx = *shard_index_by_path
            .entry(shard_path.clone())
            .or_insert_with(|| {
                let idx = shard_paths.len();
                shard_paths.push(shard_path);
                idx
            });
        tensor_map.insert(
            tensor_name.clone(),
            TensorLocation {
                shard_idx,
                dtype: String::new(),
                shape: Vec::new(),
                data_offsets: [0, 0],
            },
        );
    }
    Ok((shard_paths, tensor_map))
}

fn build_unsharded_shard_map(
    shards: &[PathBuf],
) -> Result<(
    Vec<PathBuf>,
    std::collections::BTreeMap<String, TensorLocation>,
)> {
    let mut tensor_map: std::collections::BTreeMap<String, TensorLocation> =
        std::collections::BTreeMap::new();
    for (shard_idx, shard_path) in shards.iter().enumerate() {
        let file = File::open(shard_path).map_err(|e| model_load(shard_path, e.to_string()))?;
        let mmap =
            unsafe { Mmap::map(&file) }.map_err(|e| model_load(shard_path, e.to_string()))?;
        let file_len = mmap.len() as u64;
        let mut len_bytes = [0u8; 8];
        let mut reader = &mmap[..];
        reader
            .read_exact(&mut len_bytes)
            .map_err(|e| model_load(shard_path, format!("read header length: {e}")))?;
        let header_len = u64::from_le_bytes(len_bytes);
        if header_len > MAX_HEADER_BYTES as u64 {
            return Err(model_load(
                shard_path,
                format!("header length {header_len} exceeds limit {MAX_HEADER_BYTES}"),
            ));
        }
        if 8u64
            .checked_add(header_len)
            .is_none_or(|end| end > file_len)
        {
            return Err(model_load(shard_path, "header extends beyond file".into()));
        }
        let header_bytes = &mmap[8..8 + header_len as usize];
        let header = parse_json_rejecting_duplicate_keys(header_bytes, shard_path, "header")?;
        let object = header.as_object().ok_or_else(|| {
            model_load(
                shard_path,
                "Safetensors header must be a JSON object".into(),
            )
        })?;
        for (name, _value) in object {
            if name == "__metadata__" {
                continue;
            }
            tensor_map.insert(
                name.clone(),
                TensorLocation {
                    shard_idx,
                    dtype: String::new(),
                    shape: Vec::new(),
                    data_offsets: [0, 0],
                },
            );
        }
    }
    Ok((shards.to_vec(), tensor_map))
}

impl MappedSafetensorsCheckpoint {
    /// Map a directory containing `config.json` plus either one
    /// `*.safetensors.index.json` (sharded) or one or more `*.safetensors` shards.
    pub fn from_directory(path: &str) -> Result<Self> {
        let root = Path::new(path);
        if !root.is_dir() {
            return Err(HybridError::ModelLoad {
                path: path.to_owned(),
                reason: "path is not a directory".into(),
            });
        }

        let config = Self::read_config(root)?;

        // Try index-first (sharded) layout; fall back to single-file
        let (shard_paths, mut tensor_map) = if let Some(index_path) = find_index_file(root)? {
            build_indexed_shard_map(root, &index_path)?
        } else {
            let shards = list_safetensors_files(root)?;
            if shards.is_empty() {
                return Err(HybridError::ModelLoad {
                    path: path.to_owned(),
                    reason: "no .safetensors files found".into(),
                });
            }
            build_unsharded_shard_map(&shards)?
        };

        // Map each shard and parse its header to resolve tensor dtypes/offsets
        let mut shards: Vec<MappedShard> = Vec::new();
        for (shard_idx, shard_path) in shard_paths.iter().enumerate() {
            let file = File::open(shard_path).map_err(|e| model_load(shard_path, e.to_string()))?;
            let mmap =
                unsafe { Mmap::map(&file) }.map_err(|e| model_load(shard_path, e.to_string()))?;
            let file_len = mmap.len() as u64;
            let mut len_bytes = [0u8; 8];
            let mut reader = &mmap[..];
            reader
                .read_exact(&mut len_bytes)
                .map_err(|e| model_load(shard_path, format!("read header length: {e}")))?;
            let header_len = u64::from_le_bytes(len_bytes);
            if header_len > MAX_HEADER_BYTES as u64 {
                return Err(model_load(
                    shard_path,
                    format!("header length {header_len} exceeds limit {MAX_HEADER_BYTES}"),
                ));
            }
            if 8u64
                .checked_add(header_len)
                .is_none_or(|end| end > file_len)
            {
                return Err(model_load(shard_path, "header extends beyond file".into()));
            }
            let header_bytes = &mmap[8..8 + header_len as usize];
            let header = parse_json_rejecting_duplicate_keys(header_bytes, shard_path, "header")?;
            let object = header.as_object().ok_or_else(|| {
                model_load(
                    shard_path,
                    "Safetensors header must be a JSON object".into(),
                )
            })?;

            let data_len = file_len
                .checked_sub(8)
                .and_then(|len| len.checked_sub(header_len))
                .ok_or_else(|| model_load(shard_path, "data range underflow".into()))?;

            for (name, value) in object {
                if name == "__metadata__" {
                    continue;
                }
                let Some(loc) = tensor_map.get_mut(name) else {
                    continue;
                };
                if loc.shard_idx != shard_idx {
                    continue;
                }
                let tensor = value.as_object().ok_or_else(|| {
                    model_load(
                        shard_path,
                        format!("tensor '{name}' metadata must be an object"),
                    )
                })?;
                let dtype = tensor
                    .get("dtype")
                    .and_then(Value::as_str)
                    .ok_or_else(|| {
                        model_load(shard_path, format!("tensor '{name}' missing dtype"))
                    })?
                    .to_string();
                let shape = tensor
                    .get("shape")
                    .and_then(Value::as_array)
                    .ok_or_else(|| {
                        model_load(shard_path, format!("tensor '{name}' missing shape"))
                    })?
                    .iter()
                    .map(|dim| {
                        dim.as_u64()
                            .and_then(|v| usize::try_from(v).ok())
                            .ok_or_else(|| {
                                model_load(shard_path, format!("tensor '{name}' invalid shape"))
                            })
                    })
                    .collect::<Result<Vec<_>>>()?;
                let offsets = tensor
                    .get("data_offsets")
                    .and_then(Value::as_array)
                    .ok_or_else(|| {
                        model_load(shard_path, format!("tensor '{name}' missing data_offsets"))
                    })?;
                if offsets.len() != 2 {
                    return Err(model_load(
                        shard_path,
                        format!("tensor '{name}' data_offsets must have length 2"),
                    ));
                }
                let start = offsets[0]
                    .as_u64()
                    .and_then(|v| usize::try_from(v).ok())
                    .ok_or_else(|| {
                        model_load(shard_path, format!("tensor '{name}' invalid start offset"))
                    })?;
                let end = offsets[1]
                    .as_u64()
                    .and_then(|v| usize::try_from(v).ok())
                    .ok_or_else(|| {
                        model_load(shard_path, format!("tensor '{name}' invalid end offset"))
                    })?;
                if start > end || end as u64 > data_len {
                    return Err(model_load(
                        shard_path,
                        format!("tensor '{name}' data range invalid"),
                    ));
                }
                let expected = expected_tensor_byte_size(&dtype, &shape, shard_path, name)?;
                if expected != end - start {
                    return Err(model_load(
                        shard_path,
                        format!(
                            "tensor '{name}' byte size mismatch: expected {expected}, got {}",
                            end - start
                        ),
                    ));
                }
                loc.dtype = dtype;
                loc.shape = shape;
                loc.data_offsets = [start, end];
            }

            shards.push(MappedShard {
                path: shard_path.clone(),
                mmap,
                header_len,
            });
        }

        // Validate that all tensors listed in weight_map were found in their declared shard
        for (name, loc) in &tensor_map {
            if loc.dtype.is_empty() {
                return Err(model_load(
                    &shard_paths[loc.shard_idx],
                    format!("tensor '{name}' listed in weight_map but missing from shard header"),
                ));
            }
        }

        let hidden_size = config.resolved_hidden_size().ok_or_else(|| {
            model_load(
                &root.join("config.json"),
                "config.json missing hidden_size (top-level or text_config)".into(),
            )
        })?;
        let num_layers = config.resolved_num_layers().ok_or_else(|| {
            model_load(
                &root.join("config.json"),
                "config.json missing num_hidden_layers (top-level or text_config)".into(),
            )
        })?;
        let vocab_size = config.resolved_vocab_size().ok_or_else(|| {
            model_load(
                &root.join("config.json"),
                "config.json missing vocab_size (top-level or text_config)".into(),
            )
        })?;
        let metadata = SafetensorsMetadata {
            architecture: config.architectures.first().cloned().unwrap_or_default(),
            hidden_size,
            num_layers,
            // Prefer MoE-specific keys (num_local_experts / n_routed_experts)
            // before defaulting to 1 (Codex: Granite/Moonlight ST packs).
            num_experts: config.resolved_num_experts(),
            expert_used_count: config.resolved_expert_used_count(),
            vocab_size,
        };

        Ok(Self {
            shards,
            tensor_map,
            metadata,
        })
    }

    /// Extract a full tensor as `Vec<f32>`, converting BF16/F16 as needed.
    pub fn extract_tensor_f32(&self, name: &str, path: &str) -> Result<Vec<f32>> {
        let loc = self
            .tensor_map
            .get(name)
            .ok_or_else(|| HybridError::MissingTensor {
                name: name.to_owned(),
                path: path.to_owned(),
            })?;
        let shard = &self.shards[loc.shard_idx];
        let data_start = 8 + shard.header_len as usize + loc.data_offsets[0];
        let data_end = 8 + shard.header_len as usize + loc.data_offsets[1];
        let bytes = &shard.mmap[data_start..data_end];

        match loc.dtype.as_str() {
            "F32" => {
                if !bytes.len().is_multiple_of(4) {
                    return Err(HybridError::UnsupportedFormat(format!(
                        "tensor '{name}' F32 data is not 4-byte aligned"
                    )));
                }
                Ok(bytes
                    .chunks_exact(4)
                    .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                    .collect())
            }
            "F16" => {
                if !bytes.len().is_multiple_of(2) {
                    return Err(HybridError::UnsupportedFormat(format!(
                        "tensor '{name}' F16 data is not 2-byte aligned"
                    )));
                }
                Ok(bytes
                    .chunks_exact(2)
                    .map(|b| f16_to_f32(u16::from_le_bytes([b[0], b[1]])))
                    .collect())
            }
            "BF16" => {
                if !bytes.len().is_multiple_of(2) {
                    return Err(HybridError::UnsupportedFormat(format!(
                        "tensor '{name}' BF16 data is not 2-byte aligned"
                    )));
                }
                Ok(bytes
                    .chunks_exact(2)
                    .map(|b| bf16_to_f32(u16::from_le_bytes([b[0], b[1]])))
                    .collect())
            }
            "INT4" | "I4" | "U4" => {
                // Int4: packed format, 2 elements per byte.
                // Compute exact expected element count from shape.
                let expected_elements = loc.shape.iter().product::<usize>();
                let mut out = Vec::with_capacity(expected_elements);
                let is_signed = loc.dtype == "I4";
                for &byte in bytes {
                    let low = byte & 0x0F;
                    let high = byte >> 4;
                    if is_signed {
                        // Sign-extend 4-bit to 8-bit, then convert to f32
                        out.push(((low as i8) << 4 >> 4) as f32);
                        if out.len() < expected_elements {
                            out.push(((high as i8) << 4 >> 4) as f32);
                        }
                    } else {
                        out.push(low as f32);
                        if out.len() < expected_elements {
                            out.push(high as f32);
                        }
                    }
                }
                Ok(out)
            }
            other => Err(HybridError::UnsupportedFormat(format!(
                "tensor '{name}' has unsupported Safetensors dtype '{other}'"
            ))),
        }
    }

    /// Extract a single row (token embedding) as `Vec<f32>`.
    pub fn extract_token_embedding(
        &self,
        name: &str,
        path: &str,
        token_id: usize,
    ) -> Result<Vec<f32>> {
        let loc = self
            .tensor_map
            .get(name)
            .ok_or_else(|| HybridError::MissingTensor {
                name: name.to_owned(),
                path: path.to_owned(),
            })?;
        if loc.shape.len() != 2 {
            return Err(HybridError::UnsupportedFormat(format!(
                "token embedding tensor '{name}' must be rank-2, got {:?}",
                loc.shape
            )));
        }
        let d0 = loc.shape[0];
        let d1 = loc.shape[1];
        if token_id >= d0 {
            return Err(HybridError::InputLengthMismatch {
                expected: d0,
                got: token_id,
            });
        }
        let row_elements = d1;
        let row_bytes = if loc.dtype == "INT4" || loc.dtype == "I4" || loc.dtype == "U4" {
            // Int4: packed format, 2 elements per byte
            row_elements.div_ceil(2)
        } else {
            let element_size = dtype_size_bytes(&loc.dtype).unwrap_or(2);
            row_elements * element_size
        };
        let shard = &self.shards[loc.shard_idx];
        let data_start = 8 + shard.header_len as usize + loc.data_offsets[0];
        let row_start = data_start + token_id * row_bytes;
        let row_end = row_start + row_bytes;
        let bytes = &shard.mmap[row_start..row_end];

        match loc.dtype.as_str() {
            "F32" => Ok(bytes
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect()),
            "F16" => Ok(bytes
                .chunks_exact(2)
                .map(|b| f16_to_f32(u16::from_le_bytes([b[0], b[1]])))
                .collect()),
            "BF16" => Ok(bytes
                .chunks_exact(2)
                .map(|b| bf16_to_f32(u16::from_le_bytes([b[0], b[1]])))
                .collect()),
            "INT4" | "I4" | "U4" => {
                let mut out = Vec::with_capacity(row_elements);
                let is_signed = loc.dtype == "I4";
                for &byte in bytes {
                    let low = byte & 0x0F;
                    let high = byte >> 4;
                    if is_signed {
                        out.push(((low as i8) << 4 >> 4) as f32);
                        if out.len() < row_elements {
                            out.push(((high as i8) << 4 >> 4) as f32);
                        }
                    } else {
                        out.push(low as f32);
                        if out.len() < row_elements {
                            out.push(high as f32);
                        }
                    }
                }
                Ok(out)
            }
            other => Err(HybridError::UnsupportedFormat(format!(
                "token embedding tensor '{name}' has unsupported dtype '{other}'"
            ))),
        }
    }

    pub fn tensor_info(&self, name: &str) -> Option<(&str, &[usize])> {
        self.tensor_map
            .get(name)
            .map(|loc| (loc.dtype.as_str(), loc.shape.as_slice()))
    }

    fn read_config(root: &Path) -> Result<HfConfig> {
        let config_path = root.join("config.json");
        let bytes = fs::read(&config_path).map_err(|e| model_load(&config_path, e.to_string()))?;
        serde_json::from_slice(&bytes)
            .map_err(|e| model_load(&config_path, format!("parse config.json: {e}")))
    }
}

/// Convert a BF16 bit pattern to `f32`.
pub fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}
