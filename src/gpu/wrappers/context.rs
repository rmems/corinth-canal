// SPDX-License-Identifier: Apache-2.0 OR MIT
// ════════════════════════════════════════════════════════════════════
//  gpu/context.rs — CUDA device context initialisation
// ════════════════════════════════════════════════════════════════════

use super::error::{GpuError, GpuResult};
use cust::context::legacy::{Context, ContextFlags};
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

        let ctx =
            Context::create_and_push(ContextFlags::MAP_HOST | ContextFlags::SCHED_AUTO, device)
                .map_err(|e| GpuError::InitFailed(format!("Context::create_and_push: {e:?}")))?;

        Ok(Self { _ctx: ctx })
    }

    /// Returns `true` when a CUDA device is accessible.
    ///
    /// Stub builds report unavailable so callers do not select GPU execution
    /// even when the host driver would otherwise initialise.
    pub fn is_available() -> bool {
        if cfg!(gpu_stub) {
            return false;
        }
        cust::init(cust::CudaFlags::empty()).is_ok() && Device::get_device(0).is_ok()
    }
}
