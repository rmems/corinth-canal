// ════════════════════════════════════════════════════════════════════
//  gpu/kernel.rs — Fatbin Module Loading and Kernel Management
//
//  Kernel binaries are produced by build.rs as nvcc fatbin files
//  (sm_120 SASS + compute_120 PTX as a JIT fallback) written to
//  OUT_DIR and embedded at compile time with include_bytes!. There is
//  no runtime file-system lookup — the bytes travel with the binary.
//
//  Most kernels load through the fatbin/SASS path; the Blackwell-
//  critical F16 GIF and SAAQ paths launch through a linked C ABI shim
//  in ffi.rs instead.
//
//  When module load fails (typically only via the PTX-JIT fallback
//  path) we re-run cuModuleLoadDataEx through the raw driver API to
//  capture CU_JIT_ERROR_LOG_BUFFER / CU_JIT_INFO_LOG_BUFFER and
//  surface them in the returned GpuError. That makes InvalidPtx and
//  similar driver rejections actionable instead of opaque.
// ════════════════════════════════════════════════════════════════════

use super::error::{GpuError, GpuResult};
use cust::function::Function;
use cust::module::Module;
use cust::sys as cuda;
use std::collections::HashMap;
use std::ffi::c_void;
use std::os::raw::c_uint;
use std::ptr;

// ── Compile-time fatbin embedding ────────────────────────────────────────────
//
// OUT_DIR is set by Cargo to the directory where build.rs wrote its outputs.
// include_bytes! expands at compile time, so no file-system access at runtime.
static SPIKING_NETWORK_FATBIN: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/spiking_network_sm_120.fatbin"));

static VECTOR_SIMILARITY_FATBIN: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/vector_similarity_sm_120.fatbin"));

static SATSOLVER_FATBIN: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/satsolver_sm_120.fatbin"));

// ── KernelModule ─────────────────────────────────────────────────────────────

/// Manages compiled fatbin modules and kernel function handles.
pub struct KernelModule {
    modules: HashMap<String, Module>,
    func_map: HashMap<String, String>,
}

impl KernelModule {
    /// Load all fatbin modules from their compile-time-embedded byte slices.
    ///
    /// On sm_120 hardware the driver picks precompiled SASS directly. The
    /// embedded PTX (compute_120) is only consulted when SASS is missing
    /// or incompatible — for example on a future arch.
    pub fn load() -> GpuResult<Self> {
        let mut modules = HashMap::new();
        let mut func_map = HashMap::new();

        let mut load_and_map = |bytes: &[u8], mod_name: &str, funcs: &[&str]| -> GpuResult<()> {
            let module = Self::load_module_from_fatbin(bytes, mod_name)?;

            for &func_name in funcs {
                if module.get_function(func_name).is_err() {
                    return Err(GpuError::KernelNotFound(format!(
                        "{func_name} in {mod_name}"
                    )));
                }
                func_map.insert(func_name.to_string(), mod_name.to_string());
            }
            modules.insert(mod_name.to_string(), module);
            Ok(())
        };

        load_and_map(
            SPIKING_NETWORK_FATBIN,
            "spiking_network",
            &[
                "poisson_encode",
                "project_snapshot_current",
                "lif_step",
                "lif_step_weighted",
                "gif_step_weighted",
                "spike_rate",
                "reset_membrane",
                "stdp_update",
                "neuro_bias_logits",
                "membrane_dv_dt_reduce_pass1",
                "routing_entropy_reduce_pass1",
                "latent_reduce_pass2",
            ],
        )?;

        load_and_map(
            VECTOR_SIMILARITY_FATBIN,
            "vector_similarity",
            &["cosine_similarity_batched", "cosine_similarity_top_k"],
        )?;

        load_and_map(
            SATSOLVER_FATBIN,
            "satsolver",
            &[
                "satsolver_init",
                "satsolver_step",
                "satsolver_aux_update",
                "satsolver_check_solution",
                "satsolver_extract",
                "satsolver_best_reduce_pass1",
                "satsolver_best_reduce_pass2",
            ],
        )?;

        Ok(Self { modules, func_map })
    }

    /// Alias kept for satsolver call-sites.
    pub fn load_satsolver() -> GpuResult<Self> {
        Self::load()
    }

    /// Retrieve a kernel [`Function`] handle by name.
    pub fn get_function<'a>(&'a self, name: &str) -> GpuResult<Function<'a>> {
        let mod_name = self
            .func_map
            .get(name)
            .ok_or_else(|| GpuError::KernelNotFound(name.to_string()))?;

        let module = self
            .modules
            .get(mod_name)
            .ok_or_else(|| GpuError::KernelNotFound(format!("module {mod_name} missing")))?;

        module
            .get_function(name)
            .map_err(|e| GpuError::KernelNotFound(format!("{name}: {e}")))
    }

    // ── private helpers ───────────────────────────────────────────────────────

    /// Load a fatbin module. On failure, re-run the load through the raw
    /// driver API to capture the CUDA JIT logs and embed them in the error.
    fn load_module_from_fatbin(bytes: &[u8], name: &str) -> GpuResult<Module> {
        if bytes.is_empty() {
            return Err(GpuError::ModuleLoadFailed(format!(
                "fatbin for '{name}' is empty — built with `gpu-stub` feature \
                 and no nvcc available; rebuild with CUDA Toolkit \u{2265} 12.8"
            )));
        }

        match Module::from_fatbin(bytes, &[]) {
            Ok(module) => Ok(module),
            Err(e) => {
                let log = capture_jit_log(bytes);
                eprintln!(
                    "[CUDA JIT] Failed to load module '{name}': {e:?}\n\
                     --- CUDA JIT log ---\n{log}\n--------------------"
                );
                Err(GpuError::ModuleLoadFailed(format!(
                    "JIT/load failed for '{name}': {e:?} \
                     (target: sm_120 — check driver \u{2265} 570 and CUDA toolkit \u{2265} 12.8)\n\
                     --- CUDA JIT log ---\n{log}\n--------------------"
                )))
            }
        }
    }
}

/// Re-run `cuModuleLoadDataEx` through the raw driver API with error/info
/// log buffers attached, and return the captured diagnostics as a single
/// human-readable string. The loaded module (if any) is immediately
/// unloaded — this is purely a diagnostic pass.
fn capture_jit_log(bytes: &[u8]) -> String {
    // 16 KiB is plenty for any real-world JIT log, and we cap below to be safe.
    const LOG_CAP: usize = 16 * 1024;

    // The driver requires the input image to be valid for the lifetime of the
    // call. For PTX it must be NUL-terminated; cubin/fatbin are length-encoded
    // ELF/fatbin images, but appending a trailing 0 is harmless and matches
    // what `cust::module::Module::from_cubin` does.
    let mut image = bytes.to_vec();
    image.push(0);

    let mut error_buf = vec![0u8; LOG_CAP];
    let mut info_buf = vec![0u8; LOG_CAP];

    let mut options: [cuda::CUjit_option; 5] = [
        cuda::CUjit_option::CU_JIT_ERROR_LOG_BUFFER,
        cuda::CUjit_option::CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES,
        cuda::CUjit_option::CU_JIT_INFO_LOG_BUFFER,
        cuda::CUjit_option::CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES,
        cuda::CUjit_option::CU_JIT_LOG_VERBOSE,
    ];

    // cuModuleLoadDataEx is the documented exception in the driver API:
    // small option values are passed by value, cast through usize so they
    // fit in a void*. Buffer pointers are passed as their actual pointers.
    let mut option_values: [*mut c_void; 5] = [
        error_buf.as_mut_ptr() as *mut c_void,
        LOG_CAP as *mut c_void,
        info_buf.as_mut_ptr() as *mut c_void,
        LOG_CAP as *mut c_void,
        0usize as *mut c_void,
    ];

    let mut module: cuda::CUmodule = ptr::null_mut();
    let result = unsafe {
        cuda::cuModuleLoadDataEx(
            &mut module as *mut cuda::CUmodule,
            image.as_ptr() as *const c_void,
            options.len() as c_uint,
            options.as_mut_ptr(),
            option_values.as_mut_ptr(),
        )
    };

    // If the diagnostic load actually succeeded, immediately unload it so we
    // don't leak the module. We don't return it because the caller already
    // has a more idiomatic load attempt of its own.
    if result == cuda::cudaError_enum::CUDA_SUCCESS && !module.is_null() {
        unsafe {
            let _ = cuda::cuModuleUnload(module);
        }
    }

    let error_log = cstr_from_buf(&error_buf);
    let info_log = cstr_from_buf(&info_buf);

    let mut combined = String::new();
    if !error_log.is_empty() {
        combined.push_str("error: ");
        combined.push_str(&error_log);
    }
    if !info_log.is_empty() {
        if !combined.is_empty() {
            combined.push('\n');
        }
        combined.push_str("info: ");
        combined.push_str(&info_log);
    }
    if combined.is_empty() {
        combined.push_str("<driver returned no JIT diagnostics>");
    }
    combined
}

fn cstr_from_buf(buf: &[u8]) -> String {
    let end = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
    String::from_utf8_lossy(&buf[..end]).trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::wrappers::context::GpuContext;

    #[test]
    #[cfg(feature = "cuda")] // requires GPU + driver ≥ 570
    fn test_load_kernels() {
        let _ctx = GpuContext::init().expect("Failed to initialize GPU context");
        let kernels = KernelModule::load().expect("Failed to load kernels");

        assert!(kernels.get_function("cosine_similarity_batched").is_ok());
        assert!(kernels.get_function("lif_step").is_ok());
        assert!(kernels.get_function("satsolver_step").is_ok());
    }
}
