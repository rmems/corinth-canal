// ════════════════════════════════════════════════════════════════════
//  gpu/kernel.rs — PTX Module Loading and Kernel Management
//
//  PTX files are compiled by myelin-accelerator/build.rs into OUT_DIR
//  and embedded at compile time with include_str!.  There is no runtime
//  file-system lookup — the bytes travel with the binary.
// ════════════════════════════════════════════════════════════════════

use super::error::{GpuError, GpuResult};
use cust::function::Function;
use cust::module::Module;
use std::collections::HashMap;

// ── Compile-time PTX embedding ───────────────────────────────────────────────
//
// OUT_DIR is set by Cargo to the directory where build.rs wrote its outputs.
// include_str! expands at compile time, so no file-system access at runtime.
static SPIKING_NETWORK_PTX: &str =
    include_str!(concat!(env!("OUT_DIR"), "/spiking_network_sm_120.ptx"));

static VECTOR_SIMILARITY_PTX: &str =
    include_str!(concat!(env!("OUT_DIR"), "/vector_similarity_sm_120.ptx"));

static SATSOLVER_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/satsolver_sm_120.ptx"));

// ── KernelModule ─────────────────────────────────────────────────────────────

/// Manages compiled PTX modules and kernel function handles.
pub struct KernelModule {
    modules: HashMap<String, Module>,
    func_map: HashMap<String, String>,
}

impl KernelModule {
    /// Load all PTX modules from their compile-time-embedded byte strings.
    ///
    /// The PTX is JIT-compiled by the CUDA driver on first call.
    /// On sm_120 hardware with an up-to-date driver this takes < 1 s.
    pub fn load() -> GpuResult<Self> {
        let mut modules = HashMap::new();
        let mut func_map = HashMap::new();

        // Helper closure: load one PTX string, verify required functions exist,
        // then register everything in the maps.
        let mut load_and_map = |ptx: &str, mod_name: &str, funcs: &[&str]| -> GpuResult<()> {
            let module = Self::load_module_from_ptx(ptx, mod_name)?;

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
            SPIKING_NETWORK_PTX,
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
                "saaq_find_best_walker",
            ],
        )?;

        load_and_map(
            VECTOR_SIMILARITY_PTX,
            "vector_similarity",
            &["cosine_similarity_batched", "cosine_similarity_top_k"],
        )?;

        load_and_map(
            SATSOLVER_PTX,
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

    /// JIT-compile a PTX string into a loaded CUDA module.
    fn load_module_from_ptx(ptx: &str, name: &str) -> GpuResult<Module> {
        Module::from_ptx(ptx, &[]).map_err(|e| {
            eprintln!("[CUDA JIT] Failed to load module '{name}': {e:?}");
            GpuError::ModuleLoadFailed(format!(
                "JIT compilation failed for '{name}': {e:?} \
                 (target: sm_120 — check driver ≥ 570 and CUDA toolkit ≥ 12.8)"
            ))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::wrappers::context::GpuContext;

    #[test]
    #[ignore] // requires GPU + driver ≥ 570
    fn test_load_kernels() {
        let _ctx = GpuContext::init().expect("Failed to initialize GPU context");
        let kernels = KernelModule::load().expect("Failed to load kernels");

        assert!(kernels.get_function("cosine_similarity_batched").is_ok());
        assert!(kernels.get_function("lif_step").is_ok());
        assert!(kernels.get_function("satsolver_step").is_ok());
    }
}
