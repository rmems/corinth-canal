// SPDX-License-Identifier: Apache-2.0 OR MIT
// ════════════════════════════════════════════════════════════════════
//  gpu/ffi.rs — C ABI shim wrappers for Blackwell-critical kernels
//
//  Most kernels launch through the fatbin modules loaded in kernel.rs. The two
//  latency-critical Blackwell paths below launch through a linked CUDA
//  shim so Rust can pass raw driver handles into a runtime `<<<>>>` call.
// ════════════════════════════════════════════════════════════════════

use super::error::{GpuError, GpuResult};
use super::sentry_capture::{LaunchContext, LaunchType, capture_launch_failure};
use cust::memory::{DeviceCopy, DevicePointer};
use cust::stream::Stream;
use std::ffi::c_void;

unsafe extern "C" {
    fn myelin_launch_gif_step_weighted_f16(
        stream: *mut c_void,
        grid_x: u32,
        block_x: u32,
        shared_bytes: u32,
        membrane: *mut c_void,
        adaptation: *mut c_void,
        weights: *mut c_void,
        input_spikes: *mut c_void,
        refractory: *mut c_void,
        spikes_out: *mut c_void,
        n_neurons: i32,
        n_inputs: i32,
    ) -> i32;

    fn myelin_launch_saaq_find_best_walker(
        stream: *mut c_void,
        grid_x: u32,
        block_x: u32,
        shared_bytes: u32,
        membrane: *mut c_void,
        adaptation: *mut c_void,
        partial_scores: *mut c_void,
        partial_walkers: *mut c_void,
        best_walker_out: *mut c_void,
        n_neurons: i32,
        adaptation_scale: f32,
    ) -> i32;
}

#[allow(clippy::too_many_arguments)]
pub fn launch_gif_step_weighted_f16(
    stream: &Stream,
    grid_x: u32,
    block_x: u32,
    shared_bytes: u32,
    membrane: DevicePointer<f32>,
    adaptation: DevicePointer<f32>,
    weights: DevicePointer<u16>,
    input_spikes: DevicePointer<f32>,
    refractory: DevicePointer<u32>,
    spikes_out: DevicePointer<u32>,
    n_neurons: i32,
    n_inputs: i32,
) -> GpuResult<()> {
    let code = unsafe {
        myelin_launch_gif_step_weighted_f16(
            stream.as_inner().cast::<c_void>(),
            grid_x,
            block_x,
            shared_bytes,
            device_ptr_to_void(membrane),
            device_ptr_to_void(adaptation),
            device_ptr_to_void(weights),
            device_ptr_to_void(input_spikes),
            device_ptr_to_void(refractory),
            device_ptr_to_void(spikes_out),
            n_neurons,
            n_inputs,
        )
    };

    if code == 0 {
        Ok(())
    } else {
        let context = LaunchContext {
            kernel_name: "gif_step_weighted_f16".to_string(),
            launch_type: LaunchType::CAbiShim,
            grid: (grid_x, 1, 1),
            block: (block_x, 1, 1),
            shared_mem: shared_bytes,
            neuron_count: Some(n_neurons as usize),
        };
        let gpu_error = GpuError::LaunchFailed(format!(
            "myelin_launch_gif_step_weighted_f16 failed with CUDA runtime error code {code}"
        ));
        capture_launch_failure(context, &gpu_error, None);
        Err(gpu_error)
    }
}

#[allow(clippy::too_many_arguments)]
pub fn launch_saaq_find_best_walker(
    stream: &Stream,
    grid_x: u32,
    block_x: u32,
    shared_bytes: u32,
    membrane: DevicePointer<f32>,
    adaptation: DevicePointer<f32>,
    partial_scores: DevicePointer<f32>,
    partial_walkers: DevicePointer<u32>,
    best_walker_out: DevicePointer<u32>,
    n_neurons: i32,
    adaptation_scale: f32,
) -> GpuResult<()> {
    let code = unsafe {
        myelin_launch_saaq_find_best_walker(
            stream.as_inner().cast::<c_void>(),
            grid_x,
            block_x,
            shared_bytes,
            device_ptr_to_void(membrane),
            device_ptr_to_void(adaptation),
            device_ptr_to_void(partial_scores),
            device_ptr_to_void(partial_walkers),
            device_ptr_to_void(best_walker_out),
            n_neurons,
            adaptation_scale,
        )
    };

    if code == 0 {
        Ok(())
    } else {
        let context = LaunchContext {
            kernel_name: "saaq_find_best_walker".to_string(),
            launch_type: LaunchType::CAbiShim,
            grid: (grid_x, 1, 1),
            block: (block_x, 1, 1),
            shared_mem: shared_bytes,
            neuron_count: Some(n_neurons as usize),
        };
        let gpu_error = GpuError::LaunchFailed(format!(
            "myelin_launch_saaq_find_best_walker failed with CUDA runtime error code {code}"
        ));
        capture_launch_failure(context, &gpu_error, None);
        Err(gpu_error)
    }
}

fn device_ptr_to_void<T: DeviceCopy>(ptr: DevicePointer<T>) -> *mut c_void {
    ptr.as_raw() as usize as *mut c_void
}
