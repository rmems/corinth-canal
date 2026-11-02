// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Integration test for the no-client path of GPU Sentry launch-failure capture.
//!
//! Without a Sentry client `capture_launch_failure` returns early, and this file
//! never binds one. It therefore pins only that no-op path and that
//! `LaunchContext`, `LaunchType` and `GpuError` can be constructed from outside
//! the crate. It does not assert event contents.

#[cfg(feature = "cuda")]
mod gpu_telemetry_tests {
    use corinth_canal::gpu::wrappers::error::GpuError;
    use corinth_canal::gpu::wrappers::sentry_capture::{
        LaunchContext, LaunchType, capture_launch_failure,
    };

    fn context(
        kernel: &str,
        launch_type: LaunchType,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        shared_mem: u32,
        neuron_count: Option<usize>,
    ) -> LaunchContext {
        LaunchContext {
            kernel_name: kernel.to_string(),
            launch_type,
            grid,
            block,
            shared_mem,
            neuron_count,
        }
    }

    #[test]
    fn capture_launch_failure_is_noop_without_client() {
        let rows = vec![
            (
                context(
                    "gif_step_weighted",
                    LaunchType::PtxFatbin,
                    (8, 1, 1),
                    (256, 1, 1),
                    4096,
                    Some(2048),
                ),
                GpuError::LaunchFailed("CUDA error: invalid configuration".to_string()),
                None,
            ),
            (
                context(
                    "spiking_network",
                    LaunchType::PtxFatbin,
                    (0, 0, 0),
                    (0, 0, 0),
                    0,
                    None,
                ),
                GpuError::ModuleLoadFailed(
                    "JIT compilation failed: unsupported PTX version".to_string(),
                ),
                Some((
                    "error: ptxas fatal: Unsupported .version 8.5".to_string(),
                    "info: 0 bytes gmem".to_string(),
                )),
            ),
            (
                context(
                    "gif_step_weighted_f16",
                    LaunchType::CAbiShim,
                    (8, 1, 1),
                    (256, 1, 1),
                    8192,
                    Some(2048),
                ),
                GpuError::LaunchFailed("CUDA runtime error code 1".to_string()),
                None,
            ),
            (
                context(
                    "satsolver_aux_update",
                    LaunchType::PtxFatbin,
                    (65535, 65535, 1),
                    (1024, 1, 1),
                    49152,
                    None,
                ),
                GpuError::LaunchFailed("too many resources requested".to_string()),
                None,
            ),
            (
                context(
                    "test_memory_error",
                    LaunchType::PtxFatbin,
                    (1, 1, 1),
                    (256, 1, 1),
                    0,
                    None,
                ),
                GpuError::MemoryError("out of memory".to_string()),
                None,
            ),
            (
                context(
                    "test_kernel_not_found",
                    LaunchType::PtxFatbin,
                    (1, 1, 1),
                    (256, 1, 1),
                    0,
                    None,
                ),
                GpuError::KernelNotFound("unknown_kernel".to_string()),
                None,
            ),
            (
                context(
                    "test_no_gpu",
                    LaunchType::PtxFatbin,
                    (1, 1, 1),
                    (256, 1, 1),
                    0,
                    None,
                ),
                GpuError::NoGpu,
                None,
            ),
        ];

        for (launch_context, error, jit_logs) in rows {
            capture_launch_failure(launch_context, &error, jit_logs);
        }
    }
}
