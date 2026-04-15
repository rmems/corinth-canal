// ════════════════════════════════════════════════════════════════════
//  gpu/context.rs — CUDA device context initialisation
// ════════════════════════════════════════════════════════════════════

use super::error::{GpuError, GpuResult};
use cust::context::Context;
use cust::device::Device;

/// Owns a CUDA primary context for device 0.
pub struct GpuContext {
    pub(crate) _ctx: Context,
}

impl GpuContext {
    /// Initialise CUDA and create a context on the first available device.
    pub fn init() -> GpuResult<Self> {
        cust::init(cust::CudaFlags::empty())
            .map_err(|e| GpuError::InitFailed(format!("cust::init: {e:?}")))?;

        let device = Device::get_device(0)
            .map_err(|e| GpuError::InitFailed(format!("get_device(0): {e:?}")))?;

        let ctx = Context::new(device)
            .map_err(|e| GpuError::InitFailed(format!("Context::new: {e:?}")))?;

        Ok(Self { _ctx: ctx })
    }

    /// Returns `true` when a CUDA device is accessible.
    pub fn is_available() -> bool {
        cust::init(cust::CudaFlags::empty()).is_ok() && Device::get_device(0).is_ok()
    }
}
