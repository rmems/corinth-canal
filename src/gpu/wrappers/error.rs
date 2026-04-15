// ════════════════════════════════════════════════════════════════════
//  gpu/error.rs — GPU error types
// ════════════════════════════════════════════════════════════════════

use std::fmt;

pub type GpuResult<T> = Result<T, GpuError>;

#[derive(Debug)]
pub enum GpuError {
    /// CUDA context / device initialisation failed.
    InitFailed(String),
    /// PTX module failed to load or JIT-compile.
    ModuleLoadFailed(String),
    /// A kernel function was not found in the loaded module.
    KernelNotFound(String),
    /// Memory allocation or copy failed.
    MemoryError(String),
    /// Kernel launch failed.
    LaunchFailed(String),
    /// Generic CUDA error forwarded from `cust`.
    CudaError(String),
    /// GPU is not available on this system.
    NoGpu,
}

impl fmt::Display for GpuError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GpuError::InitFailed(s) => write!(f, "GPU init failed: {s}"),
            GpuError::ModuleLoadFailed(s) => write!(f, "PTX module load failed: {s}"),
            GpuError::KernelNotFound(s) => write!(f, "Kernel not found: {s}"),
            GpuError::MemoryError(s) => write!(f, "GPU memory error: {s}"),
            GpuError::LaunchFailed(s) => write!(f, "Kernel launch failed: {s}"),
            GpuError::CudaError(s) => write!(f, "CUDA error: {s}"),
            GpuError::NoGpu => write!(f, "No GPU available"),
        }
    }
}

impl std::error::Error for GpuError {}

impl From<cust::error::CudaError> for GpuError {
    fn from(e: cust::error::CudaError) -> Self {
        GpuError::CudaError(format!("{e:?}"))
    }
}
