// ════════════════════════════════════════════════════════════════════
//  gpu/mod.rs — GPU sub-module declarations and public re-exports
// ════════════════════════════════════════════════════════════════════

pub mod accelerator;
pub mod context;
pub mod error;
pub mod kernel;
pub mod memory;

pub use accelerator::GpuAccelerator;
pub use context::GpuContext;
pub use error::{GpuError, GpuResult};
pub use kernel::KernelModule;
pub use memory::GpuBuffer;
