// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Mapped GGUF checkpoint access, probe/mmap, and tensor extraction.

use super::super::ggml::{
    GGML_TYPE_F16, GGML_TYPE_F32, GGML_TYPE_IQ3_M_BLOCK, GGML_TYPE_IQ3_S, GGML_TYPE_Q5_K,
    GGML_TYPE_Q6_K, GGML_TYPE_Q8_0,
};
#[cfg(feature = "cuda")]
use super::cuda_register::RegisteredTensorSliceU16;
use super::dequant::{
    dequantize_row_iq3_m, dequantize_row_q5_k, dequantize_row_q6_k, dequantize_row_q8_0,
    f16_to_f32, tensor_row_size,
};
use super::metadata::{GgufMetadata, GgufTensorInfo, parse_checkpoint_layout};
use crate::error::{HybridError, Result};
use memmap2::{MmapMut, MmapOptions};
use std::collections::HashMap;
use std::fs::OpenOptions;
use std::slice;

#[derive(Debug)]
pub(in crate::moe) struct MappedGgufCheckpoint {
    // `registered_gpu_synapse` holds a CUDA host-registration over pages owned
    // by `mmap`, so `cuMemHostUnregister` must run while the mapping is still
    // valid. Declaration order below puts it first, but the explicit `Drop`
    // impl is what actually guarantees it: a future field reorder or refactor
    // would otherwise silently reintroduce unregister-after-munmap.
    #[cfg(feature = "cuda")]
    registered_gpu_synapse: Option<RegisteredTensorSliceU16>,
    mmap: MmapMut,
    tensors: HashMap<String, GgufTensorInfo>,
    metadata: GgufMetadata,
}

#[cfg(feature = "cuda")]
impl Drop for MappedGgufCheckpoint {
    fn drop(&mut self) {
        // Release the CUDA host-registration before `mmap` is dropped, so the
        // pages are still mapped when `cuMemHostUnregister` runs. Taking the
        // Option drops the registration here rather than relying on field
        // declaration order.
        self.registered_gpu_synapse.take();
    }
}

pub(in crate::moe) fn extract_named_token_embedding_from_checkpoint(
    checkpoint: &mut MappedGgufCheckpoint,
    tensor_name: &str,
    path: &str,
    token_id: usize,
) -> Result<Vec<f32>> {
    checkpoint.extract_token_embedding(tensor_name, path, token_id)
}

impl MappedGgufCheckpoint {
    fn extract_token_embedding(
        &mut self,
        tensor_name: &str,
        path: &str,
        token_id: usize,
    ) -> Result<Vec<f32>> {
        let info = self.tensor_info(tensor_name, path)?.clone();
        if info.dims.is_empty() {
            return Err(HybridError::UnsupportedFormat(format!(
                "tensor '{tensor_name}' has no dimensions"
            )));
        }
        let d0 = info.dims[0];
        let d1 = info.dims.get(1).copied().unwrap_or(0);

        match info.ggml_type {
            GGML_TYPE_F32 => {
                let weights = self.f32_tensor(tensor_name, path)?;
                if token_id >= d1 {
                    return Err(HybridError::InputLengthMismatch {
                        expected: d1,
                        got: token_id,
                    });
                }
                Ok(weights[token_id * d0..token_id * d0 + d0].to_vec())
            }
            GGML_TYPE_F16 => {
                let values = self.u16_tensor_values(&info, path, tensor_name)?;
                if token_id >= d1 {
                    return Err(HybridError::InputLengthMismatch {
                        expected: d1,
                        got: token_id,
                    });
                }
                Ok(values[token_id * d0..token_id * d0 + d0]
                    .iter()
                    .map(|&b| f16_to_f32(b))
                    .collect())
            }
            GGML_TYPE_Q8_0 => {
                dequantize_row_q8_0(self.row_bytes(&info, token_id, path, tensor_name)?, d0)
            }
            GGML_TYPE_Q5_K => {
                dequantize_row_q5_k(self.row_bytes(&info, token_id, path, tensor_name)?, d0)
            }
            GGML_TYPE_Q6_K => {
                dequantize_row_q6_k(self.row_bytes(&info, token_id, path, tensor_name)?, d0)
            }
            GGML_TYPE_IQ3_S => Err(HybridError::UnsupportedFormat(format!(
                "tensor '{tensor_name}' uses IQ3_S token embeddings; checkpoint-backed token embedding extraction is unsupported for this quantization"
            ))),
            other => Err(HybridError::UnsupportedFormat(format!(
                "tensor '{tensor_name}' has unsupported ggml_type={other}"
            ))),
        }
    }
}

pub(in crate::moe) fn probe_and_map_checkpoint(
    path: &str,
) -> Result<(GgufMetadata, MappedGgufCheckpoint)> {
    let file = OpenOptions::new()
        .read(true)
        .open(path)
        .map_err(|e| HybridError::ModelLoad {
            path: path.to_owned(),
            reason: e.to_string(),
        })?;
    // SAFETY: The file is a valid, readable file descriptor opened above.
    // `map_copy` creates a private copy-on-write mapping that does not
    // write back to the underlying file.  The writable mapping is required
    // by `cuMemHostRegister_v2`, which expects a non-const pointer even
    // though it does not modify the memory contents.
    let mmap =
        unsafe { MmapOptions::new().map_copy(&file) }.map_err(|e| HybridError::ModelLoad {
            path: path.to_owned(),
            reason: format!("copy-on-write mmap failed: {e}"),
        })?;

    let parsed = parse_checkpoint_layout(&mmap, path)?;

    Ok((
        parsed.metadata.clone(),
        MappedGgufCheckpoint {
            mmap,
            tensors: parsed.tensors,
            #[cfg(feature = "cuda")]
            registered_gpu_synapse: None,
            metadata: parsed.metadata,
        },
    ))
}

impl MappedGgufCheckpoint {
    pub(in crate::moe) fn metadata(&self) -> &GgufMetadata {
        &self.metadata
    }

    #[cfg(test)]
    pub(in crate::moe) fn set_quantization_for_test(&mut self, quantization: String) {
        self.metadata.set_quantization_for_test(quantization);
    }

    #[cfg(test)]
    pub(in crate::moe) fn set_numeric_for_test(&mut self, key: &str, value: Option<u64>) {
        self.metadata.set_numeric_for_test(key, value);
    }

    pub(in crate::moe) fn has_tensor(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }

    pub(in crate::moe) fn find_first_tensor_with_suffix(&self, suffix: &str) -> Option<&str> {
        let mut matches: Vec<&str> = self
            .tensors
            .keys()
            .map(String::as_str)
            .filter(|name| name.ends_with(suffix))
            .collect();
        matches.sort_unstable_by_key(|name| tensor_block_sort_key(name));
        matches.into_iter().next()
    }

    pub(in crate::moe) fn tensor_info<'a>(
        &'a self,
        name: &str,
        path: &str,
    ) -> Result<&'a GgufTensorInfo> {
        self.tensors
            .get(name)
            .ok_or_else(|| HybridError::MissingTensor {
                name: name.to_owned(),
                path: path.to_owned(),
            })
    }

    pub(in crate::moe) fn f32_tensor<'a>(&'a self, name: &str, path: &str) -> Result<&'a [f32]> {
        let info = self.tensor_info(name, path)?;
        if info.ggml_type != GGML_TYPE_F32 {
            return Err(HybridError::UnsupportedFormat(format!(
                "tensor '{name}' must be F32, got ggml_type={}",
                info.ggml_type
            )));
        }

        let start = info.absolute_offset;
        let byte_len = info
            .n_elements
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| HybridError::ModelLoad {
                path: path.to_owned(),
                reason: format!("tensor '{name}' byte length overflow"),
            })?;
        let end = start
            .checked_add(byte_len)
            .ok_or_else(|| HybridError::ModelLoad {
                path: path.to_owned(),
                reason: format!("tensor '{name}' end offset overflow"),
            })?;
        if end > self.mmap.len() {
            return Err(HybridError::ModelLoad {
                path: path.to_owned(),
                reason: format!("tensor '{name}' extends beyond mapped file"),
            });
        }
        if !start.is_multiple_of(std::mem::align_of::<f32>()) {
            return Err(HybridError::ModelLoad {
                path: path.to_owned(),
                reason: format!("tensor '{name}' has misaligned F32 data offset {start}"),
            });
        }

        // SAFETY: `start` is a valid byte offset into the mmap and `end` is
        // checked against `mmap.len()` above.  F32 alignment is guaranteed
        // because GGUF aligns all tensor data to at least 32 bytes (enforced
        // by the `alignment` field parsed from the file header).  The returned
        // slice borrows `self` for lifetime `'a`, keeping the mmap alive.
        let ptr = unsafe { self.mmap.as_ptr().add(start) as *const f32 };
        Ok(unsafe { slice::from_raw_parts(ptr, info.n_elements) })
    }

    fn u16_tensor_values(
        &self,
        info: &GgufTensorInfo,
        path: &str,
        tensor_name: &str,
    ) -> Result<Vec<u16>> {
        let byte_start = info.absolute_offset;
        let byte_len = info
            .n_elements
            .checked_mul(std::mem::size_of::<u16>())
            .ok_or_else(|| HybridError::ModelLoad {
                path: path.to_owned(),
                reason: format!("tensor '{tensor_name}' byte length overflow"),
            })?;
        let byte_end = byte_start
            .checked_add(byte_len)
            .ok_or_else(|| HybridError::ModelLoad {
                path: path.to_owned(),
                reason: format!("tensor '{tensor_name}' end offset overflow"),
            })?;
        if byte_end > self.mmap.len() {
            return Err(HybridError::ModelLoad {
                path: path.to_owned(),
                reason: format!("tensor '{tensor_name}' extends beyond mapped file"),
            });
        }
        Ok(self.mmap[byte_start..byte_end]
            .chunks_exact(2)
            .map(|b| u16::from_le_bytes([b[0], b[1]]))
            .collect())
    }

    fn row_bytes<'a>(
        &'a self,
        info: &GgufTensorInfo,
        row_idx: usize,
        path: &str,
        tensor_name: &str,
    ) -> Result<&'a [u8]> {
        let n_rows = info.dims.get(1).copied().unwrap_or(0);
        if row_idx >= n_rows {
            return Err(HybridError::InputLengthMismatch {
                expected: n_rows,
                got: row_idx,
            });
        }

        let row_size = tensor_row_size(info.ggml_type, info.dims[0])?;
        let start =
            info.absolute_offset
                .checked_add(row_idx.checked_mul(row_size).ok_or_else(|| {
                    HybridError::ModelLoad {
                        path: path.to_owned(),
                        reason: format!("tensor '{tensor_name}' row offset overflow"),
                    }
                })?)
                .ok_or_else(|| HybridError::ModelLoad {
                    path: path.to_owned(),
                    reason: format!("tensor '{tensor_name}' row offset overflow"),
                })?;
        let end = start
            .checked_add(row_size)
            .ok_or_else(|| HybridError::ModelLoad {
                path: path.to_owned(),
                reason: format!("tensor '{tensor_name}' row end offset overflow"),
            })?;
        if end > self.mmap.len() {
            return Err(HybridError::ModelLoad {
                path: path.to_owned(),
                reason: format!("tensor '{tensor_name}' row extends beyond mapped file"),
            });
        }
        Ok(&self.mmap[start..end])
    }

    #[cfg(feature = "cuda")]
    pub(in crate::moe) fn registered_f16_tensor<'a>(
        &'a mut self,
        name: &str,
        path: &str,
    ) -> Result<&'a [u16]> {
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

    /// Dequantize a full Q8_0 tensor to a flat `Vec<f32>`.
    ///
    /// Iterates over every row of the tensor and applies the Q8_0
    /// block-scale dequantization, producing `dims[0] * dims[1]` output
    /// elements laid out row-major. `dims[0]` must be divisible by 32.
    #[allow(dead_code)]
    pub(in crate::moe) fn dequantize_q8_0_tensor(
        &self,
        name: &str,
        path: &str,
    ) -> Result<Vec<f32>> {
        let info = self.tensor_info(name, path)?.clone();
        if info.ggml_type != GGML_TYPE_Q8_0 {
            return Err(HybridError::UnsupportedFormat(format!(
                "tensor '{name}' must be Q8_0, got ggml_type={}",
                info.ggml_type
            )));
        }
        if info.dims.is_empty() {
            return Err(HybridError::UnsupportedFormat(format!(
                "tensor '{name}' has no dimensions"
            )));
        }
        let width = info.dims[0];
        let n_rows = info.dims.get(1).copied().unwrap_or(1);
        let capacity = width
            .checked_mul(n_rows)
            .ok_or_else(|| HybridError::ModelLoad {
                path: path.to_owned(),
                reason: format!("tensor '{name}' element count overflow ({width}×{n_rows})"),
            })?;
        let mut out = Vec::with_capacity(capacity);
        for row in 0..n_rows {
            let row_bytes = self.row_bytes(&info, row, path, name)?;
            let dequantized = dequantize_row_q8_0(row_bytes, width)?;
            out.extend_from_slice(&dequantized);
        }
        Ok(out)
    }

    /// Dequantize a full Q5_K tensor to a flat `Vec<f32>`.
    ///
    /// Iterates over every row of the tensor and applies the Q5_K
    /// block-scale dequantization, producing `dims[0] * dims[1]` output
    /// elements laid out row-major. `dims[0]` must be divisible by 256.
    #[allow(dead_code)]
    pub(in crate::moe) fn dequantize_q5_k_tensor(
        &self,
        name: &str,
        path: &str,
    ) -> Result<Vec<f32>> {
        let info = self.tensor_info(name, path)?.clone();
        if info.ggml_type != GGML_TYPE_Q5_K {
            return Err(HybridError::UnsupportedFormat(format!(
                "tensor '{name}' must be Q5_K, got ggml_type={}",
                info.ggml_type
            )));
        }
        if info.dims.is_empty() {
            return Err(HybridError::UnsupportedFormat(format!(
                "tensor '{name}' has no dimensions"
            )));
        }
        let width = info.dims[0];
        let n_rows = info.dims.get(1).copied().unwrap_or(1);
        let capacity = width
            .checked_mul(n_rows)
            .ok_or_else(|| HybridError::ModelLoad {
                path: path.to_owned(),
                reason: format!("tensor '{name}' element count overflow ({width}×{n_rows})"),
            })?;
        let mut out = Vec::with_capacity(capacity);
        for row in 0..n_rows {
            let row_bytes = self.row_bytes(&info, row, path, name)?;
            let dequantized = dequantize_row_q5_k(row_bytes, width)?;
            out.extend_from_slice(&dequantized);
        }
        Ok(out)
    }

    #[allow(dead_code)]
    pub(in crate::moe) fn dequantize_q6_k_tensor(
        &self,
        name: &str,
        path: &str,
    ) -> Result<Vec<f32>> {
        let info = self.tensor_info(name, path)?.clone();
        if info.ggml_type != GGML_TYPE_Q6_K {
            return Err(HybridError::UnsupportedFormat(format!(
                "tensor '{name}' must be Q6_K, got ggml_type={}",
                info.ggml_type
            )));
        }
        if info.dims.is_empty() {
            return Err(HybridError::UnsupportedFormat(format!(
                "tensor '{name}' has no dimensions"
            )));
        }
        let width = info.dims[0];
        let n_rows = info.dims.get(1).copied().unwrap_or(1);
        let capacity = width
            .checked_mul(n_rows)
            .ok_or_else(|| HybridError::ModelLoad {
                path: path.to_owned(),
                reason: format!("tensor '{name}' element count overflow ({width}x{n_rows})"),
            })?;
        let mut out = Vec::with_capacity(capacity);
        for row in 0..n_rows {
            let row_bytes = self.row_bytes(&info, row, path, name)?;
            let dequantized = dequantize_row_q6_k(row_bytes, width)?;
            out.extend_from_slice(&dequantized);
        }
        Ok(out)
    }

    /// Dequantize a full IQ3_M tensor to a flat `Vec<f32>`.
    ///
    /// Iterates over every row of the tensor and applies the IQ3_M
    /// block-scale dequantization, producing `dims[0] * dims[1]` output
    /// elements laid out row-major. `dims[0]` must be divisible by 256.
    #[allow(dead_code)]
    pub(in crate::moe) fn dequantize_iq3_m_tensor(
        &self,
        name: &str,
        path: &str,
    ) -> Result<Vec<f32>> {
        let info = self.tensor_info(name, path)?.clone();
        if info.ggml_type != GGML_TYPE_IQ3_M_BLOCK {
            return Err(HybridError::UnsupportedFormat(format!(
                "tensor '{name}' must be IQ3_M block layout (internal type), got ggml_type={}",
                info.ggml_type
            )));
        }
        if info.dims.is_empty() {
            return Err(HybridError::UnsupportedFormat(format!(
                "tensor '{name}' has no dimensions"
            )));
        }
        let width = info.dims[0];
        let n_rows = info.dims.get(1).copied().unwrap_or(1);
        let capacity = width
            .checked_mul(n_rows)
            .ok_or_else(|| HybridError::ModelLoad {
                path: path.to_owned(),
                reason: format!("tensor '{name}' element count overflow ({width}x{n_rows})"),
            })?;
        let mut out = Vec::with_capacity(capacity);
        for row in 0..n_rows {
            let row_bytes = self.row_bytes(&info, row, path, name)?;
            let dequantized = dequantize_row_iq3_m(row_bytes, width)?;
            out.extend_from_slice(&dequantized);
        }
        Ok(out)
    }
}

fn tensor_block_sort_key(name: &str) -> (usize, &str) {
    let block = name
        .strip_prefix("blk.")
        .and_then(|rest| rest.split_once('.'))
        .and_then(|(idx, _)| idx.parse::<usize>().ok())
        .unwrap_or(usize::MAX);
    (block, name)
}
