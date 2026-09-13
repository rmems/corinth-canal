// SPDX-License-Identifier: Apache-2.0 OR MIT
//! CUDA host-registration helpers for mapped GGUF tensor regions.
//!
//! All items are gated on `feature = "cuda"`. The module itself remains
//! present in CPU-only builds so `mod cuda_register;` is unconditional.

#[cfg(feature = "cuda")]
use crate::error::{HybridError, Result};
#[cfg(feature = "cuda")]
use memmap2::MmapMut;
#[cfg(feature = "cuda")]
use std::ffi::c_void;
#[cfg(feature = "cuda")]
use std::slice;

#[cfg(feature = "cuda")]
#[derive(Debug)]
pub(in crate::moe) struct RegisteredTensorSliceU16 {
    pub(in crate::moe) tensor_name: String,
    _region: RegisteredCudaRegion,
    ptr: *const u16,
    len: usize,
}

#[cfg(feature = "cuda")]
#[derive(Debug)]
struct RegisteredCudaRegion {
    ptr: *mut c_void,
}

#[cfg(feature = "cuda")]
impl RegisteredTensorSliceU16 {
    pub(in crate::moe) fn register(
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
        let tensor_end =
            absolute_offset
                .checked_add(byte_len)
                .ok_or_else(|| HybridError::ModelLoad {
                    path: path.to_owned(),
                    reason: format!("tensor '{tensor_name}' registration end overflow"),
                })?;
        let aligned_end = checked_align_up(tensor_end, page_size, path, tensor_name)?;
        let register_len =
            aligned_end
                .checked_sub(aligned_start)
                .ok_or_else(|| HybridError::ModelLoad {
                    path: path.to_owned(),
                    reason: format!("tensor '{tensor_name}' registration range underflow"),
                })?;

        if aligned_end > mmap.len() {
            return Err(HybridError::ModelLoad {
                path: path.to_owned(),
                reason: format!("tensor '{tensor_name}' registration range exceeds mmap"),
            });
        }

        if !absolute_offset.is_multiple_of(std::mem::align_of::<u16>()) {
            return Err(HybridError::ModelLoad {
                path: path.to_owned(),
                reason: format!(
                    "tensor '{tensor_name}' has misaligned F16 data offset {absolute_offset}"
                ),
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

    pub(in crate::moe) fn as_slice(&self) -> &[u16] {
        // SAFETY: `ptr` and `len` are set in `register` and satisfy the
        // invariants of `slice::from_raw_parts`: the pointer is non-null,
        // correctly aligned (u16 = 2 bytes, GGUF alignment ≥ 32), and the
        // slice is live as long as the owning `MappedGgufCheckpoint` (and
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
            // Never panic in `drop`, but do not swallow it either: a failed
            // unregister leaks a pinned mapping, and the empty arm made that
            // invisible.
            tracing::warn!(
                ?result,
                "cuMemHostUnregister failed; CUDA pinned registration leaked"
            );
        }
    }
}

#[cfg(all(feature = "cuda", unix))]
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

#[cfg(all(feature = "cuda", not(unix)))]
#[allow(dead_code)]
fn page_size_bytes(_path: &str) -> Result<usize> {
    // Fallback for Windows and other platforms (common page size 4KiB is sufficient
    // for the mmap alignment use case here).
    Ok(4096)
}

#[cfg(feature = "cuda")]
fn checked_align_up(
    value: usize,
    alignment: usize,
    path: &str,
    tensor_name: &str,
) -> Result<usize> {
    if alignment == 0 {
        return Ok(value);
    }

    value
        .div_ceil(alignment)
        .checked_mul(alignment)
        .ok_or_else(|| HybridError::ModelLoad {
            path: path.to_owned(),
            reason: format!("tensor '{tensor_name}' aligned offset overflow"),
        })
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
