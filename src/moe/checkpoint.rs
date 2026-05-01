//! GGUF checkpoint parsing and mapped tensor access for the router bridge.

use super::{
    GGML_TYPE_F16, GGML_TYPE_F32, GGML_TYPE_IQ3_S, GGML_TYPE_Q5_K, GGML_TYPE_Q8_0, GGUF_MAGIC,
    GGUF_VALUE_TYPE_ARRAY, GGUF_VALUE_TYPE_BOOL, GGUF_VALUE_TYPE_FLOAT32, GGUF_VALUE_TYPE_FLOAT64,
    GGUF_VALUE_TYPE_INT8, GGUF_VALUE_TYPE_INT16, GGUF_VALUE_TYPE_INT32, GGUF_VALUE_TYPE_INT64,
    GGUF_VALUE_TYPE_STRING, GGUF_VALUE_TYPE_UINT8, GGUF_VALUE_TYPE_UINT16, GGUF_VALUE_TYPE_UINT32,
    GGUF_VALUE_TYPE_UINT64, GGUF_VERSION,
};
use crate::error::{HybridError, Result};
use memmap2::{MmapMut, MmapOptions};
use std::collections::HashMap;
#[cfg(feature = "cuda")]
use std::ffi::c_void;
use std::fs::OpenOptions;
use std::slice;

#[derive(Debug)]
pub(super) struct MappedGgufCheckpoint {
    mmap: MmapMut,
    tensors: HashMap<String, GgufTensorInfo>,
    #[cfg(feature = "cuda")]
    registered_gpu_synapse: Option<RegisteredTensorSliceU16>,
    metadata: GgufMetadata,
}

#[derive(Debug, Clone)]
pub(super) struct GgufTensorInfo {
    pub(super) dims: Vec<usize>,
    pub(super) ggml_type: u32,
    pub(super) relative_offset: usize,
    pub(super) absolute_offset: usize,
    pub(super) n_elements: usize,
}

#[cfg(feature = "cuda")]
#[derive(Debug)]
struct RegisteredTensorSliceU16 {
    tensor_name: String,
    _region: RegisteredCudaRegion,
    ptr: *const u16,
    len: usize,
}

#[cfg(feature = "cuda")]
#[derive(Debug)]
struct RegisteredCudaRegion {
    ptr: *mut c_void,
}

pub(super) struct ParsedCheckpointLayout {
    pub(super) metadata: GgufMetadata,
    pub(super) tensors: HashMap<String, GgufTensorInfo>,
}

#[derive(Debug, Clone, Default)]
pub(super) struct GgufMetadata {
    architecture: String,
    quantization: String,
    numerics: HashMap<String, u64>,
}

struct GgufCursor<'a> {
    bytes: &'a [u8],
    offset: usize,
}

pub(super) fn extract_named_token_embedding_from_checkpoint(
    checkpoint: &mut MappedGgufCheckpoint,
    tensor_name: &str,
    path: &str,
    token_id: usize,
) -> Result<Vec<f32>> {
    checkpoint.extract_token_embedding(tensor_name, path, token_id)
}

impl GgufMetadata {
    pub(super) fn architecture(&self) -> &str {
        &self.architecture
    }

    pub(super) fn quantization(&self) -> &str {
        &self.quantization
    }

    pub(super) fn numeric(&self, key: &str) -> Option<usize> {
        self.numerics.get(key).copied().map(|v| v as usize)
    }
}

impl MappedGgufCheckpoint {
    fn extract_token_embedding(
        &mut self,
        tensor_name: &str,
        path: &str,
        token_id: usize,
    ) -> Result<Vec<f32>> {
        let info = self.tensor_info(tensor_name, path)?.clone();
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
            GGML_TYPE_IQ3_S => Err(HybridError::UnsupportedFormat(format!(
                "tensor '{tensor_name}' uses IQ3_S token embeddings; use llama.cpp prompt embeddings for this checkpoint"
            ))),
            other => Err(HybridError::UnsupportedFormat(format!(
                "tensor '{tensor_name}' has unsupported ggml_type={other}"
            ))),
        }
    }
}

pub(super) fn probe_and_map_checkpoint(path: &str) -> Result<(GgufMetadata, MappedGgufCheckpoint)> {
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

pub(super) fn parse_checkpoint_layout(bytes: &[u8], path: &str) -> Result<ParsedCheckpointLayout> {
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

    // Sanity-bound the header counts to prevent OOM allocation from malformed files.
    const MAX_TENSOR_COUNT: usize = 100_000;
    const MAX_KV_COUNT: usize = 100_000;
    const MAX_TENSOR_DIMS: usize = 8;

    let tensor_count_raw = cursor.read_u64(path)?;
    if tensor_count_raw > MAX_TENSOR_COUNT as u64 {
        return Err(HybridError::UnsupportedFormat(format!(
            "tensor_count {tensor_count_raw} exceeds maximum allowed {MAX_TENSOR_COUNT}"
        )));
    }
    let tensor_count = tensor_count_raw as usize;

    let kv_count_raw = cursor.read_u64(path)?;
    if kv_count_raw > MAX_KV_COUNT as u64 {
        return Err(HybridError::UnsupportedFormat(format!(
            "kv_count {kv_count_raw} exceeds maximum allowed {MAX_KV_COUNT}"
        )));
    }
    let kv_count = kv_count_raw as usize;

    let mut alignment = 32usize;
    let mut file_type = None;
    let mut architecture = None;
    let mut numerics = HashMap::new();

    for _ in 0..kv_count {
        let key = cursor.read_string(path)?;
        let value_type = cursor.read_u32(path)?;
        match key.as_str() {
            "general.alignment" => alignment = cursor.read_numeric_as_usize(value_type, path)?,
            "general.file_type" => {
                let value = cursor.read_numeric_as_u32(value_type, path)?;
                file_type = Some(value);
                numerics.insert(key, value as u64);
            }
            "general.architecture" => {
                let value = cursor.read_string(path)?;
                architecture = Some(value);
            }
            _ => {
                if let Some(value) = cursor.read_numeric_value(value_type, path)? {
                    numerics.insert(key, value);
                } else if value_type == GGUF_VALUE_TYPE_STRING {
                    let _ = cursor.read_string(path)?;
                } else {
                    cursor.skip_value(value_type, path)?;
                }
            }
        }
    }

    let mut tensors = HashMap::with_capacity(tensor_count);
    for _ in 0..tensor_count {
        let name = cursor.read_string(path)?;
        let n_dims_raw = cursor.read_u32(path)? as usize;
        if n_dims_raw > MAX_TENSOR_DIMS {
            return Err(HybridError::UnsupportedFormat(format!(
                "tensor '{name}' has {n_dims_raw} dims, which exceeds maximum allowed {MAX_TENSOR_DIMS}"
            )));
        }
        let n_dims = n_dims_raw;
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

    Ok(ParsedCheckpointLayout {
        metadata: GgufMetadata {
            architecture: architecture.unwrap_or_else(|| "unknown".into()),
            quantization: quantization_label(file_type),
            numerics,
        },
        tensors,
    })
}

impl MappedGgufCheckpoint {
    pub(super) fn metadata(&self) -> &GgufMetadata {
        &self.metadata
    }

    pub(super) fn has_tensor(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }

    pub(super) fn find_first_tensor_with_suffix(&self, suffix: &str) -> Option<&str> {
        let mut matches: Vec<&str> = self
            .tensors
            .keys()
            .map(String::as_str)
            .filter(|name| name.ends_with(suffix))
            .collect();
        matches.sort_unstable_by_key(|name| tensor_block_sort_key(name));
        matches.into_iter().next()
    }

    pub(super) fn tensor_info<'a>(&'a self, name: &str, path: &str) -> Result<&'a GgufTensorInfo> {
        self.tensors
            .get(name)
            .ok_or_else(|| HybridError::MissingTensor {
                name: name.to_owned(),
                path: path.to_owned(),
            })
    }

    pub(super) fn f32_tensor<'a>(&'a self, name: &str, path: &str) -> Result<&'a [f32]> {
        let info = self.tensor_info(name, path)?;
        if info.ggml_type != GGML_TYPE_F32 {
            return Err(HybridError::UnsupportedFormat(format!(
                "tensor '{name}' must be F32, got ggml_type={}",
                info.ggml_type
            )));
        }

        let start = info.absolute_offset;
        let end = start + info.n_elements * size_of::<f32>();
        if end > self.mmap.len() {
            return Err(HybridError::ModelLoad {
                path: path.to_owned(),
                reason: format!("tensor '{name}' extends beyond mapped file"),
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
        let byte_end = byte_start + info.n_elements * 2;
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
        let end = start + row_size;
        if end > self.mmap.len() {
            return Err(HybridError::ModelLoad {
                path: path.to_owned(),
                reason: format!("tensor '{tensor_name}' row extends beyond mapped file"),
            });
        }
        Ok(&self.mmap[start..end])
    }

    #[cfg(feature = "cuda")]
    pub(super) fn registered_f16_tensor<'a>(
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
    pub(super) fn dequantize_q8_0_tensor(&self, name: &str, path: &str) -> Result<Vec<f32>> {
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
    pub(super) fn dequantize_q5_k_tensor(&self, name: &str, path: &str) -> Result<Vec<f32>> {
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
}

#[cfg(feature = "cuda")]
impl RegisteredTensorSliceU16 {
    fn register(
        tensor_name: &str,
        mmap: &MmapMut,
        absolute_offset: usize,
        n_elements: usize,
        path: &str,
    ) -> Result<Self> {
        let byte_len =
            n_elements
                .checked_mul(size_of::<u16>())
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

        // SAFETY: `aligned_start` is a page-aligned byte offset within the
        // mmap (verified by the `aligned_end > mmap.len()` guard above).
        // The `MmapMut` backing is a private copy-on-write mapping, so the
        // underlying pages are writable — required by `cuMemHostRegister_v2`
        // even though it does not modify the memory contents.
        let register_ptr = unsafe { mmap.as_ptr().add(aligned_start) as *mut c_void };
        cuda_host_register(register_ptr, register_len, path, tensor_name)?;

        // SAFETY: `absolute_offset` is within the mmap (covered by the
        // registered region above) and F16 data is at least 2-byte aligned
        // due to the GGUF alignment guarantee (min 32 bytes).  `n_elements`
        // is the exact count of u16 values stored there.
        let tensor_ptr = unsafe { mmap.as_ptr().add(absolute_offset) as *const u16 };
        Ok(Self {
            tensor_name: tensor_name.to_owned(),
            _region: RegisteredCudaRegion { ptr: register_ptr },
            ptr: tensor_ptr,
            len: n_elements,
        })
    }

    fn as_slice(&self) -> &[u16] {
        // SAFETY: `ptr` and `len` are set in `register` and satisfy the
        // invariants of `slice::from_raw_parts`: the pointer is non-null,
        // correctly aligned (u16 = 2 bytes, GGUF alignment ≥ 32), and the
        // slice is live as long as the owning `MappedOlmoeCheckpoint` (and
        // hence the mmap) is alive.
        unsafe { slice::from_raw_parts(self.ptr, self.len) }
    }
}

#[cfg(feature = "cuda")]
impl Drop for RegisteredCudaRegion {
    fn drop(&mut self) {
        // SAFETY: `ptr` was successfully registered by `cuMemHostRegister_v2`
        // in `RegisteredTensorSliceU16::register`, so it is valid to unregister.
        let result = unsafe { cust::sys::cuMemHostUnregister(self.ptr) };
        if result != cust::sys::CUresult::CUDA_SUCCESS {
            // Silently ignore: panicking inside `drop` is unsound, and the
            // model remains usable even if CUDA pin-registration is leaked.
        }
    }
}

fn tensor_row_size(ggml_type: u32, width: usize) -> Result<usize> {
    match ggml_type {
        GGML_TYPE_Q8_0 => {
            if !width.is_multiple_of(32) {
                return Err(HybridError::UnsupportedFormat(format!(
                    "Q8_0 tensor width {width} is not divisible by 32"
                )));
            }
            Ok((width / 32) * (2 + 32))
        }
        GGML_TYPE_Q5_K => {
            if !width.is_multiple_of(256) {
                return Err(HybridError::UnsupportedFormat(format!(
                    "Q5_K tensor width {width} is not divisible by 256"
                )));
            }
            Ok((width / 256) * (2 + 2 + 12 + 32 + 128))
        }
        other => Err(HybridError::UnsupportedFormat(format!(
            "row-size lookup is not implemented for ggml_type={other}"
        ))),
    }
}

fn dequantize_row_q8_0(row: &[u8], width: usize) -> Result<Vec<f32>> {
    if !width.is_multiple_of(32) {
        return Err(HybridError::UnsupportedFormat(format!(
            "Q8_0 width {width} is not divisible by 32"
        )));
    }

    let mut out = Vec::with_capacity(width);
    for block in row.chunks_exact(34) {
        let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        for &quant in &block[2..34] {
            out.push((quant as i8) as f32 * d);
        }
    }
    Ok(out)
}

fn dequantize_row_q5_k(row: &[u8], width: usize) -> Result<Vec<f32>> {
    if !width.is_multiple_of(256) {
        return Err(HybridError::UnsupportedFormat(format!(
            "Q5_K width {width} is not divisible by 256"
        )));
    }

    let mut out = Vec::with_capacity(width);
    for block in row.chunks_exact(176) {
        let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let dmin = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
        let scales = &block[4..16];
        let qh = &block[16..48];
        let ql = &block[48..176];

        let mut is = 0usize;
        let mut u1 = 1u8;
        let mut u2 = 2u8;

        for ql_chunk in ql.chunks_exact(32) {
            let (sc1, m1) = scale_min_k4(is, scales);
            let (sc2, m2) = scale_min_k4(is + 1, scales);
            let d1 = d * sc1 as f32;
            let mn1 = dmin * m1 as f32;
            let d2 = d * sc2 as f32;
            let mn2 = dmin * m2 as f32;

            for (lane, &q) in ql_chunk.iter().enumerate() {
                let qh_byte = qh[lane];
                let hi1 = if qh_byte & u1 != 0 { 16 } else { 0 };
                let hi2 = if qh_byte & u2 != 0 { 16 } else { 0 };
                out.push(d1 * ((q & 0x0F) + hi1) as f32 - mn1);
                out.push(d2 * ((q >> 4) + hi2) as f32 - mn2);
            }

            is += 2;
            u1 <<= 2;
            u2 <<= 2;
        }
    }
    Ok(out)
}

fn scale_min_k4(index: usize, scales: &[u8]) -> (u8, u8) {
    if index < 4 {
        (scales[index] & 63, scales[index + 4] & 63)
    } else {
        (
            (scales[index + 4] & 0x0F) | ((scales[index - 4] >> 6) << 4),
            (scales[index + 4] >> 4) | ((scales[index] >> 6) << 4),
        )
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

fn quantization_label(file_type: Option<u32>) -> String {
    match file_type {
        Some(0) => "F32".into(),
        Some(1) => "F16".into(),
        Some(other) => format!("GGUF({other})"),
        None => "GGUF".into(),
    }
}

#[allow(dead_code)]
fn page_size_bytes(path: &str) -> Result<usize> {
    // SAFETY: `sysconf` is a pure query with no preconditions; valid to call at any time.
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

#[cfg(feature = "cuda")]
fn cuda_host_register(ptr: *mut c_void, len: usize, path: &str, tensor_name: &str) -> Result<()> {
    // SAFETY: `ptr` points to a page-aligned region within a live `MmapMut`
    // (validated by the caller) and `len` covers only that region.  Flags = 0
    // requests default portable pinned-host registration without any write
    // semantics imposed on the pages.
    let result = unsafe { cust::sys::cuMemHostRegister_v2(ptr, len, 0) };
    if result == cust::sys::CUresult::CUDA_SUCCESS {
        return Ok(());
    }

    Err(HybridError::ModelLoad {
        path: path.to_owned(),
        reason: format!("cuMemHostRegister_v2 failed for '{tensor_name}': {result:?}"),
    })
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

    fn read_numeric_value(&mut self, value_type: u32, path: &str) -> Result<Option<u64>> {
        let value = match value_type {
            GGUF_VALUE_TYPE_UINT8 => Some(self.read_u8(path)? as u64),
            GGUF_VALUE_TYPE_INT8 => Some(self.read_u8(path)? as i8 as i64 as u64),
            GGUF_VALUE_TYPE_UINT16 => Some(self.read_u16(path)? as u64),
            GGUF_VALUE_TYPE_INT16 => Some(self.read_i16(path)? as i64 as u64),
            GGUF_VALUE_TYPE_UINT32 => Some(self.read_u32(path)? as u64),
            GGUF_VALUE_TYPE_INT32 => Some(self.read_i32(path)? as i64 as u64),
            GGUF_VALUE_TYPE_UINT64 => Some(self.read_u64(path)?),
            GGUF_VALUE_TYPE_INT64 => Some(self.read_i64(path)? as u64),
            GGUF_VALUE_TYPE_BOOL => Some(self.read_u8(path)? as u64),
            GGUF_VALUE_TYPE_FLOAT32 => Some(self.read_u32(path)? as u64),
            GGUF_VALUE_TYPE_FLOAT64 => Some(self.read_u64(path)?),
            GGUF_VALUE_TYPE_STRING | GGUF_VALUE_TYPE_ARRAY => None,
            other => {
                return Err(HybridError::UnsupportedFormat(format!(
                    "unsupported GGUF value type {other}"
                )));
            }
        };
        Ok(value)
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
