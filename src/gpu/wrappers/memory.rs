// ════════════════════════════════════════════════════════════════════
//  gpu/memory.rs — GPU device buffer wrapper
// ════════════════════════════════════════════════════════════════════

use super::error::{GpuError, GpuResult};
use cust::memory::{CopyDestination, DeviceBuffer};

/// Owned device buffer of type `T`.
pub struct GpuBuffer<T: cust::memory::DeviceCopy> {
    inner: DeviceBuffer<T>,
    len: usize,
}

impl<T: cust::memory::DeviceCopy + Default + Clone> GpuBuffer<T> {
    /// Allocate a device buffer of `len` elements (uninitialised).
    pub fn alloc(len: usize) -> GpuResult<Self> {
        let inner = unsafe {
            DeviceBuffer::uninitialized(len)
                .map_err(|e| GpuError::MemoryError(format!("alloc({len}): {e:?}")))?
        };
        Ok(Self { inner, len })
    }

    /// Allocate and upload a host slice to device memory.
    pub fn from_slice(data: &[T]) -> GpuResult<Self> {
        let inner = DeviceBuffer::from_slice(data)
            .map_err(|e| GpuError::MemoryError(format!("from_slice: {e:?}")))?;
        Ok(Self {
            inner,
            len: data.len(),
        })
    }

    /// Download device data into a freshly-allocated `Vec<T>`.
    pub fn to_vec(&self) -> GpuResult<Vec<T>> {
        let mut host = vec![T::default(); self.len];
        self.inner
            .copy_to(&mut host)
            .map_err(|e| GpuError::MemoryError(format!("copy_to: {e:?}")))?;
        Ok(host)
    }

    /// Upload from a host slice (must be same length).
    pub fn upload(&mut self, data: &[T]) -> GpuResult<()> {
        self.inner
            .copy_from(data)
            .map_err(|e| GpuError::MemoryError(format!("upload: {e:?}")))
    }

    /// Number of elements.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Raw device pointer (for kernel launches via `cust`).
    pub fn as_device_ptr(&self) -> cust::memory::DevicePointer<T> {
        self.inner.as_device_ptr()
    }
}
