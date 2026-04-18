//! GGUF checkpoint parsing and mapped tensor access for the OlmoeRouter bridge.

use super::{
    DEFAULT_GPU_SYNAPSE_TENSOR_NAME, GGML_TYPE_F16, GGML_TYPE_F32, GGUF_MAGIC,
    GGUF_VALUE_TYPE_ARRAY, GGUF_VALUE_TYPE_BOOL, GGUF_VALUE_TYPE_FLOAT32, GGUF_VALUE_TYPE_FLOAT64,
    GGUF_VALUE_TYPE_INT8, GGUF_VALUE_TYPE_INT16, GGUF_VALUE_TYPE_INT32, GGUF_VALUE_TYPE_INT64,
    GGUF_VALUE_TYPE_STRING, GGUF_VALUE_TYPE_UINT8, GGUF_VALUE_TYPE_UINT16, GGUF_VALUE_TYPE_UINT32,
    GGUF_VALUE_TYPE_UINT64, GGUF_VERSION, OLMOE_HIDDEN, OLMOE_NUM_EXPERTS, OLMOE_NUM_LAYERS,
    OlmoeMetadata, ROUTING_TENSOR_NAME,
};
use crate::error::{HybridError, Result};
use crate::types::EMBEDDING_DIM;
use memmap2::{MmapMut, MmapOptions};
use std::collections::HashMap;
use std::ffi::c_void;
use std::fs::OpenOptions;
use std::slice;

#[derive(Debug)]
pub(super) struct MappedOlmoeCheckpoint {
    mmap: MmapMut,
    tensors: HashMap<String, GgufTensorInfo>,
    registered_gpu_synapse: Option<RegisteredTensorSliceU16>,
}

#[derive(Debug, Clone)]
pub(super) struct GgufTensorInfo {
    pub(super) dims: Vec<usize>,
    ggml_type: u32,
    pub(super) relative_offset: usize,
    pub(super) absolute_offset: usize,
    pub(super) n_elements: usize,
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

pub(super) struct ParsedCheckpointLayout {
    pub(super) metadata: OlmoeMetadata,
    pub(super) tensors: HashMap<String, GgufTensorInfo>,
}

struct GgufCursor<'a> {
    bytes: &'a [u8],
    offset: usize,
}

pub(super) fn extract_token_embedding_from_checkpoint(
    checkpoint: &mut MappedOlmoeCheckpoint,
    path: &str,
    token_id: usize,
) -> Result<Vec<f32>> {
    let info = checkpoint.tensor_info("token_embd.weight", path)?.clone();
    let d0 = info.dims[0];
    let d1 = info.dims.get(1).copied().unwrap_or(0);

    match info.ggml_type {
        GGML_TYPE_F32 => {
            let weights = checkpoint.f32_tensor("token_embd.weight", path)?;
            if d0 == EMBEDDING_DIM {
                if token_id >= d1 {
                    return Err(HybridError::InputLengthMismatch {
                        expected: d1,
                        got: token_id,
                    });
                }
                let start = token_id * EMBEDDING_DIM;
                Ok(weights[start..start + EMBEDDING_DIM].to_vec())
            } else if d1 == EMBEDDING_DIM {
                if token_id >= d0 {
                    return Err(HybridError::InputLengthMismatch {
                        expected: d0,
                        got: token_id,
                    });
                }
                Ok((0..EMBEDDING_DIM)
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
                    path: path.to_owned(),
                    reason: "token_embd.weight F16 tensor extends beyond mapped file".into(),
                });
            }
            let raw = &checkpoint.mmap[byte_start..byte_end];
            let u16s: Vec<u16> = raw
                .chunks_exact(2)
                .map(|b| u16::from_le_bytes([b[0], b[1]]))
                .collect();
            if d0 == EMBEDDING_DIM {
                if token_id >= d1 {
                    return Err(HybridError::InputLengthMismatch {
                        expected: d1,
                        got: token_id,
                    });
                }
                let start = token_id * EMBEDDING_DIM;
                Ok(u16s[start..start + EMBEDDING_DIM]
                    .iter()
                    .map(|&b| f16_to_f32(b))
                    .collect())
            } else if d1 == EMBEDDING_DIM {
                if token_id >= d0 {
                    return Err(HybridError::InputLengthMismatch {
                        expected: d0,
                        got: token_id,
                    });
                }
                Ok((0..EMBEDDING_DIM)
                    .map(|dim| f16_to_f32(u16s[dim * d0 + token_id]))
                    .collect())
            } else {
                Err(HybridError::UnsupportedFormat(format!(
                    "tensor 'token_embd.weight' has unexpected dimensions {:?}",
                    info.dims
                )))
            }
        }
        other => Err(HybridError::UnsupportedFormat(format!(
            "tensor 'token_embd.weight' has unsupported ggml_type={other}"
        ))),
    }
}

pub(super) fn probe_and_map_checkpoint(
    path: &str,
) -> Result<(OlmoeMetadata, MappedOlmoeCheckpoint)> {
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
    let routing =
        parsed
            .tensors
            .get(ROUTING_TENSOR_NAME)
            .ok_or_else(|| HybridError::MissingTensor {
                name: ROUTING_TENSOR_NAME.into(),
                path: path.to_owned(),
            })?;
    validate_routing_tensor(path, routing)?;

    let synapse = parsed
        .tensors
        .get(DEFAULT_GPU_SYNAPSE_TENSOR_NAME)
        .ok_or_else(|| HybridError::MissingTensor {
            name: DEFAULT_GPU_SYNAPSE_TENSOR_NAME.into(),
            path: path.to_owned(),
        })?;
    validate_gpu_synapse_tensor(path, DEFAULT_GPU_SYNAPSE_TENSOR_NAME, synapse)?;

    Ok((
        parsed.metadata,
        MappedOlmoeCheckpoint {
            mmap,
            tensors: parsed.tensors,
            registered_gpu_synapse: None,
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
    let mut expert_count = OLMOE_NUM_EXPERTS;

    for _ in 0..kv_count {
        let key = cursor.read_string(path)?;
        let value_type = cursor.read_u32(path)?;
        match key.as_str() {
            "general.alignment" => alignment = cursor.read_numeric_as_usize(value_type, path)?,
            "general.file_type" => file_type = Some(cursor.read_numeric_as_u32(value_type, path)?),
            "olmoe.expert_count" => {
                expert_count = cursor.read_numeric_as_usize(value_type, path)?
            }
            _ => cursor.skip_value(value_type, path)?,
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
        metadata: OlmoeMetadata {
            hidden_size: OLMOE_HIDDEN,
            num_layers: OLMOE_NUM_LAYERS,
            num_experts: expert_count,
            quantization: quantization_label(file_type),
        },
        tensors,
    })
}

impl MappedOlmoeCheckpoint {
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
        let end = start + info.n_elements * std::mem::size_of::<f32>();
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

impl Drop for RegisteredCudaRegion {
    fn drop(&mut self) {
        // SAFETY: `ptr` was successfully registered by `cuMemHostRegister_v2`
        // in `RegisteredTensorSliceU16::register`, so it is valid to unregister.
        let result = unsafe { cust::sys::cuMemHostUnregister(self.ptr) };
        if result != cust::sys::CUresult::CUDA_SUCCESS {
            // Silently ignore: panicking inside `drop` is unsound and the
            // model remains usable even if CUDA pin-registration is leaked.
        }
    }
}

fn validate_routing_tensor(path: &str, tensor: &GgufTensorInfo) -> Result<()> {
    if tensor.ggml_type != GGML_TYPE_F32 {
        return Err(HybridError::UnsupportedFormat(format!(
            "tensor '{ROUTING_TENSOR_NAME}' must be F32 in '{path}'"
        )));
    }
    if tensor.dims.len() != 2 {
        return Err(HybridError::UnsupportedFormat(format!(
            "tensor '{ROUTING_TENSOR_NAME}' must be rank-2, got {:?}",
            tensor.dims
        )));
    }
    if tensor.n_elements != OLMOE_HIDDEN * OLMOE_NUM_EXPERTS {
        return Err(HybridError::UnsupportedFormat(format!(
            "tensor '{ROUTING_TENSOR_NAME}' has unexpected size {:?}",
            tensor.dims
        )));
    }
    Ok(())
}

fn validate_gpu_synapse_tensor(
    path: &str,
    tensor_name: &str,
    tensor: &GgufTensorInfo,
) -> Result<()> {
    if tensor.ggml_type != GGML_TYPE_F16 {
        return Err(HybridError::UnsupportedFormat(format!(
            "tensor '{tensor_name}' must be F16 in '{path}'"
        )));
    }
    if tensor.dims != [OLMOE_HIDDEN, OLMOE_HIDDEN] {
        return Err(HybridError::UnsupportedFormat(format!(
            "tensor '{tensor_name}' must be [2048, 2048], got {:?}",
            tensor.dims
        )));
    }
    Ok(())
}

fn quantization_label(file_type: Option<u32>) -> String {
    match file_type {
        Some(0) => "F32".into(),
        Some(1) => "F16".into(),
        Some(other) => format!("GGUF({other})"),
        None => "GGUF".into(),
    }
}

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
