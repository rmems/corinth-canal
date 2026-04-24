// ════════════════════════════════════════════════════════════════════
//  gpu/accelerator.rs — High-level GPU accelerator facade
//
//  GpuAccelerator bundles the CUDA context, loaded PTX modules, and
//  convenience launchers for the neuro-spike kernel set.
//
//  When no GPU is present the struct can still be constructed in
//  "stub" mode — all kernel calls return GpuError::NoGpu so the
//  caller can fall back to the CPU engine gracefully.
// ════════════════════════════════════════════════════════════════════

use super::context::GpuContext;
use super::error::{GpuError, GpuResult};
use super::ffi;
use super::kernel::KernelModule;
use super::memory::GpuBuffer;
use crate::types::TelemetrySnapshot;
use cust::launch;
use cust::stream::{Stream, StreamFlags};
use tracing::warn;

const SATSOLVER_BLOCK_SIZE: u32 = 256;
const SATSOLVER_SHARED_MEM_BYTES: u32 = 0;
const TEMPORAL_BLOCK_SIZE: u32 = 256;
const TEMPORAL_GRID_SIZE: u32 = 8; // 8 * 256 = 2048 exact for Blackwell alignment (N_NEURONS)
const TEMPORAL_SHARED_MEM_BYTES: u32 = 0;
const SNAPSHOT_CHANNELS: usize = 4;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SynapsePrecision {
    None,
    F32,
    F16,
}

struct TemporalState {
    neuron_count: usize,
    n_inputs: usize,
    membrane: GpuBuffer<f32>,
    refractory: GpuBuffer<u32>,
    spikes_out: GpuBuffer<u32>,
    input_current: GpuBuffer<f32>,
    input_spikes: GpuBuffer<f32>,
    adaptation: GpuBuffer<f32>,
    weights_f32: GpuBuffer<f32>,
    weights_f16: Option<GpuBuffer<u16>>,
    synapse_precision: SynapsePrecision,
    synapse_signature: Option<String>,
    snapshot: GpuBuffer<f32>,
    best_walker: GpuBuffer<u32>, // persistent 1-element for SAAQ on-device reduction
    saaq_partial_scores: GpuBuffer<f32>,
    saaq_partial_walkers: GpuBuffer<u32>,
}

/// Facade that owns a CUDA context and the compiled PTX modules.
pub struct GpuAccelerator {
    /// `None` when running in CPU-only / stub mode.
    _ctx: Option<GpuContext>,
    /// `None` when PTX modules failed to load.
    modules: Option<KernelModule>,
    temporal_state: Option<TemporalState>,
}

impl GpuAccelerator {
    /// Attempt to initialise GPU.  Returns a stub if no device is available.
    pub fn new() -> Self {
        match GpuContext::init() {
            Ok(ctx) => match KernelModule::load() {
                Ok(modules) => Self {
                    _ctx: Some(ctx),
                    modules: Some(modules),
                    temporal_state: None,
                },
                Err(e) => {
                    warn!("[GPU] PTX load failed (CPU fallback): {e}");
                    Self {
                        _ctx: Some(ctx),
                        modules: None,
                        temporal_state: None,
                    }
                }
            },
            Err(e) => {
                warn!("[GPU] No CUDA device (CPU fallback): {e}");
                Self {
                    _ctx: None,
                    modules: None,
                    temporal_state: None,
                }
            }
        }
    }

    fn has_context(&self) -> bool {
        self._ctx.is_some()
    }

    /// `true` if a CUDA context is available. PTX-backed helpers may still be
    /// unavailable if module loading failed, but the shim-backed temporal F16
    /// path can still run.
    pub fn is_ready(&self) -> bool {
        self.has_context()
    }

    /// Borrow the loaded kernel module, or return an error.
    pub fn kernels(&self) -> GpuResult<&KernelModule> {
        self.modules.as_ref().ok_or(GpuError::NoGpu)
    }

    pub fn ensure_temporal_state(&mut self, neuron_count: usize) -> GpuResult<()> {
        if !self.has_context() {
            return Err(GpuError::NoGpu);
        }
        if neuron_count == 0 {
            return Err(GpuError::LaunchFailed(
                "temporal state requires neuron_count > 0".into(),
            ));
        }

        let needs_realloc = self
            .temporal_state
            .as_ref()
            .map(|state| state.neuron_count != neuron_count)
            .unwrap_or(true);

        if needs_realloc {
            self.temporal_state = Some(Self::build_temporal_state(neuron_count)?);
        }

        Ok(())
    }

    pub fn project_snapshot_current(
        &mut self,
        snapshot: &TelemetrySnapshot,
        neuron_count: usize,
    ) -> GpuResult<()> {
        self.ensure_temporal_state(neuron_count)?;

        let modules = self.modules.as_ref().ok_or(GpuError::NoGpu)?;
        let project_snapshot_current = modules.get_function("project_snapshot_current")?;
        let state = self
            .temporal_state
            .as_mut()
            .ok_or_else(|| GpuError::MemoryError("temporal state not initialised".into()))?;

        state.snapshot.upload(&[
            snapshot.gpu_temp_c,
            snapshot.gpu_power_w,
            snapshot.cpu_tctl_c,
            snapshot.cpu_package_power_w,
        ])?;

        let stream = Self::new_stream()?;
        let grid = Self::ceil_div_u32(neuron_count as u32, TEMPORAL_BLOCK_SIZE);
        unsafe {
            launch!(project_snapshot_current<<<grid, TEMPORAL_BLOCK_SIZE, TEMPORAL_SHARED_MEM_BYTES, stream>>>(
                state.snapshot.as_device_ptr(),
                state.input_current.as_device_ptr(),
                neuron_count as i32
            ))
            .map_err(|e| {
                GpuError::LaunchFailed(format!("project_snapshot_current launch: {e:?}"))
            })?;
        }

        stream
            .synchronize()
            .map_err(|e| GpuError::LaunchFailed(format!("project_snapshot_current sync: {e:?}")))
    }

    pub fn lif_step_tick(&mut self, neuron_count: usize) -> GpuResult<()> {
        self.ensure_temporal_state(neuron_count)?;

        let modules = self.modules.as_ref().ok_or(GpuError::NoGpu)?;
        let lif_step = modules.get_function("lif_step")?;
        let state = self
            .temporal_state
            .as_mut()
            .ok_or_else(|| GpuError::MemoryError("temporal state not initialised".into()))?;
        let stream = Self::new_stream()?;
        let grid = Self::ceil_div_u32(neuron_count as u32, TEMPORAL_BLOCK_SIZE);

        unsafe {
            launch!(lif_step<<<grid, TEMPORAL_BLOCK_SIZE, TEMPORAL_SHARED_MEM_BYTES, stream>>>(
                state.membrane.as_device_ptr(),
                state.input_current.as_device_ptr(),
                state.refractory.as_device_ptr(),
                state.spikes_out.as_device_ptr(),
                neuron_count as i32
            ))
            .map_err(|e| GpuError::LaunchFailed(format!("lif_step launch: {e:?}")))?;
        }

        stream
            .synchronize()
            .map_err(|e| GpuError::LaunchFailed(format!("lif_step sync: {e:?}")))
    }

    pub fn temporal_spikes_to_vec(&self, neuron_count: usize) -> GpuResult<Vec<u32>> {
        if !self.has_context() {
            return Err(GpuError::NoGpu);
        }
        let state = self
            .temporal_state
            .as_ref()
            .ok_or_else(|| GpuError::MemoryError("temporal state not initialised".into()))?;
        Self::expect_len("temporal spikes", state.spikes_out.len(), neuron_count)?;
        state.spikes_out.to_vec()
    }

    pub fn temporal_membrane_to_vec(&self, neuron_count: usize) -> GpuResult<Vec<f32>> {
        if !self.has_context() {
            return Err(GpuError::NoGpu);
        }
        let state = self
            .temporal_state
            .as_ref()
            .ok_or_else(|| GpuError::MemoryError("temporal state not initialised".into()))?;
        Self::expect_len("temporal membrane", state.membrane.len(), neuron_count)?;
        state.membrane.to_vec()
    }

    /// Download current adaptation values (GIF state) from device.
    pub fn temporal_adaptation_to_vec(&self, neuron_count: usize) -> GpuResult<Vec<f32>> {
        if !self.has_context() {
            return Err(GpuError::NoGpu);
        }
        let state = self
            .temporal_state
            .as_ref()
            .ok_or_else(|| GpuError::MemoryError("temporal state not initialised".into()))?;
        Self::expect_len("temporal adaptation", state.adaptation.len(), neuron_count)?;
        state.adaptation.to_vec()
    }

    /// Upload a per-neuron temporal input vector into the resident `input_spikes` buffer.
    pub(crate) fn upload_temporal_input_spikes(&mut self, input_spikes: &[f32]) -> GpuResult<()> {
        if !self.has_context() {
            return Err(GpuError::NoGpu);
        }
        let state = self
            .temporal_state
            .as_mut()
            .ok_or_else(|| GpuError::MemoryError("temporal state not initialised".into()))?;
        if input_spikes.len() != state.n_inputs {
            return Err(GpuError::MemoryError(format!(
                "temporal input_spikes length mismatch: expected {}, got {}",
                state.n_inputs,
                input_spikes.len()
            )));
        }
        state
            .input_spikes
            .upload(input_spikes)
            .map_err(|e| GpuError::MemoryError(format!("input_spikes upload failed: {e}")))
    }

    /// Load synapse weight matrix for GIF weighted kernel. Must match neuron_count * n_inputs.
    /// Call after ensure_temporal_state or it will be overwritten on realloc.
    pub fn load_synapse_weights(&mut self, weights: &[f32]) -> GpuResult<()> {
        self.load_synapse_weights_named("host-f32", weights)
    }

    pub fn load_synapse_weights_named(
        &mut self,
        signature: &str,
        weights: &[f32],
    ) -> GpuResult<()> {
        if !self.has_context() {
            return Err(GpuError::NoGpu);
        }
        let state = self
            .temporal_state
            .as_mut()
            .ok_or_else(|| GpuError::MemoryError("temporal state not initialised".into()))?;
        let expected = state.neuron_count * state.n_inputs;
        if weights.len() != expected {
            return Err(GpuError::MemoryError(format!(
                "weights length mismatch: expected {} ({}x{}), got {}",
                expected,
                state.neuron_count,
                state.n_inputs,
                weights.len()
            )));
        }
        if state.synapse_precision == SynapsePrecision::F32
            && state.synapse_signature.as_deref() == Some(signature)
        {
            return Ok(());
        }
        state
            .weights_f32
            .upload(weights)
            .map_err(|e| GpuError::MemoryError(format!("synapse weights upload failed: {e}")))?;
        state.synapse_precision = SynapsePrecision::F32;
        state.synapse_signature = Some(signature.to_owned());
        Ok(())
    }

    /// Load an FP16 synapse matrix directly from a CUDA-registered host slice.
    /// Reuses the resident device weights if the same source signature is already loaded.
    pub fn load_synapse_weights_f16_registered(
        &mut self,
        signature: &str,
        weights: &[u16],
    ) -> GpuResult<()> {
        if !self.has_context() {
            return Err(GpuError::NoGpu);
        }
        let state = self
            .temporal_state
            .as_mut()
            .ok_or_else(|| GpuError::MemoryError("temporal state not initialised".into()))?;
        let expected = state.neuron_count * state.n_inputs;
        if weights.len() != expected {
            return Err(GpuError::MemoryError(format!(
                "f16 weights length mismatch: expected {} ({}x{}), got {}",
                expected,
                state.neuron_count,
                state.n_inputs,
                weights.len()
            )));
        }

        if state.synapse_precision == SynapsePrecision::F16
            && state.synapse_signature.as_deref() == Some(signature)
        {
            return Ok(());
        }

        let f16_weights = match state.weights_f16.as_mut() {
            Some(weights_f16) => weights_f16,
            None => {
                state.weights_f16 = Some(GpuBuffer::<u16>::from_slice(&vec![0u16; expected])?);
                state
                    .weights_f16
                    .as_mut()
                    .expect("weights_f16 was just inserted")
            }
        };
        f16_weights.upload(weights).map_err(|e| {
            GpuError::MemoryError(format!("registered f16 synapse upload failed: {e}"))
        })?;
        state.synapse_precision = SynapsePrecision::F16;
        state.synapse_signature = Some(signature.to_owned());
        Ok(())
    }

    /// Run one GIF-weighted LIF step using adaptation, dynamic threshold, and synaptic weights.
    /// Uses shared memory sized for n_inputs. Call load_synapse_weights first.
    /// Fills spikes_out and updates membrane/adaptation/refractory.
    /// Returns the SAAQ best-walker index from on-device reduction (single u32 download).
    pub fn gif_step_weighted_tick(&mut self, neuron_count: usize) -> GpuResult<u32> {
        self.ensure_temporal_state(neuron_count)?;

        let state = self
            .temporal_state
            .as_mut()
            .ok_or_else(|| GpuError::MemoryError("temporal state not initialised".into()))?;
        if state.synapse_precision == SynapsePrecision::None {
            return Err(GpuError::MemoryError(
                "synapse weights must be loaded before gif_step_weighted_tick".into(),
            ));
        }
        let stream = Self::new_stream()?;
        // Hardcoded Blackwell alignment: 8 blocks * 256 threads = 2048 neurons exact.
        // This maxes L1/shared memory bandwidth without warp divergence on sm_120.
        let grid = TEMPORAL_GRID_SIZE;
        let shared_bytes = (state.n_inputs * 4) as u32; // f32 * n_inputs

        unsafe {
            match state.synapse_precision {
                SynapsePrecision::F32 => {
                    let modules = self.modules.as_ref().ok_or(GpuError::NoGpu)?;
                    let gif_step = modules.get_function("gif_step_weighted")?;
                    launch!(gif_step<<<grid, TEMPORAL_BLOCK_SIZE, shared_bytes, stream>>>(
                        state.membrane.as_device_ptr(),
                        state.adaptation.as_device_ptr(),
                        state.weights_f32.as_device_ptr(),
                        state.input_spikes.as_device_ptr(),
                        state.refractory.as_device_ptr(),
                        state.spikes_out.as_device_ptr(),
                        neuron_count as i32,
                        state.n_inputs as i32
                    ))
                    .map_err(|e| {
                        GpuError::LaunchFailed(format!("gif_step_weighted launch: {e:?}"))
                    })?;
                }
                SynapsePrecision::F16 => {
                    let weights_f16 = state.weights_f16.as_ref().ok_or_else(|| {
                        GpuError::MemoryError("f16 synapse buffer not initialised".into())
                    })?;
                    ffi::launch_gif_step_weighted_f16(
                        &stream,
                        grid,
                        TEMPORAL_BLOCK_SIZE,
                        shared_bytes,
                        state.membrane.as_device_ptr(),
                        state.adaptation.as_device_ptr(),
                        weights_f16.as_device_ptr(),
                        state.input_spikes.as_device_ptr(),
                        state.refractory.as_device_ptr(),
                        state.spikes_out.as_device_ptr(),
                        neuron_count as i32,
                        state.n_inputs as i32,
                    )?;
                }
                SynapsePrecision::None => unreachable!("validated above"),
            }
        }

        // Immediately launch SAAQ reduction on same stream (no intermediate sync).
        // This kills the Rust for-loop over 2048 values. Only 4-byte result downloaded.
        let walker = self.saaq_find_best_walker(&stream, neuron_count)?;
        Ok(walker)
    }

    pub fn reset_temporal_state(&mut self) -> GpuResult<()> {
        if !self.has_context() {
            return Err(GpuError::NoGpu);
        }

        let Some(state) = self.temporal_state.as_mut() else {
            return Ok(());
        };
        if let Some(modules) = self.modules.as_ref() {
            let reset_membrane = modules.get_function("reset_membrane")?;
            let stream = Self::new_stream()?;
            // Hardcoded Blackwell alignment for 2048 neurons
            let grid = TEMPORAL_GRID_SIZE;

            unsafe {
                launch!(reset_membrane<<<grid, TEMPORAL_BLOCK_SIZE, TEMPORAL_SHARED_MEM_BYTES, stream>>>(
                    state.membrane.as_device_ptr(),
                    state.neuron_count as i32,
                    0.0f32
                ))
                .map_err(|e| GpuError::LaunchFailed(format!("reset_membrane launch: {e:?}")))?;
            }

            stream
                .synchronize()
                .map_err(|e| GpuError::LaunchFailed(format!("reset_membrane sync: {e:?}")))?;
        } else {
            state
                .membrane
                .upload(&vec![0.0f32; state.neuron_count])
                .map_err(|e| GpuError::MemoryError(format!("reset membrane upload failed: {e}")))?;
        }

        state
            .refractory
            .upload(&vec![0u32; state.neuron_count])
            .map_err(|e| GpuError::MemoryError(format!("reset refractory upload failed: {e}")))?;
        state
            .spikes_out
            .upload(&vec![0u32; state.neuron_count])
            .map_err(|e| GpuError::MemoryError(format!("reset spikes upload failed: {e}")))?;
        state
            .input_current
            .upload(&vec![0.0f32; state.neuron_count])
            .map_err(|e| {
                GpuError::MemoryError(format!("reset input_current upload failed: {e}"))
            })?;
        state
            .input_spikes
            .upload(&vec![0.0f32; state.n_inputs])
            .map_err(|e| GpuError::MemoryError(format!("reset input_spikes upload failed: {e}")))?;
        state
            .adaptation
            .upload(&vec![0.0f32; state.neuron_count])
            .map_err(|e| GpuError::MemoryError(format!("reset adaptation upload failed: {e}")))?;

        // Reset SAAQ best walker. Partial buffers are overwritten on every SAAQ launch.
        state
            .best_walker
            .upload(&[0u32; 1])
            .map_err(|e| GpuError::MemoryError(format!("reset best_walker upload failed: {e}")))?;

        Ok(())
    }

    pub fn synapse_signature(&self) -> Option<&str> {
        self.temporal_state
            .as_ref()
            .and_then(|state| state.synapse_signature.as_deref())
    }

    /// Copy the best SAT walker assignment into `output`.
    ///
    /// This wrapper matches the updated CUDA signature and blocks until the
    /// extract kernel has completed.
    #[allow(clippy::too_many_arguments)]
    pub fn satsolver_extract(
        &self,
        assignment: &GpuBuffer<u8>,
        best_walker: &GpuBuffer<i32>,
        output: &mut GpuBuffer<u8>,
        n_vars: i32,
        n_walkers: i32,
    ) -> GpuResult<()> {
        if n_vars < 0 {
            return Err(GpuError::LaunchFailed(format!(
                "satsolver_extract: n_vars must be >= 0, got {n_vars}"
            )));
        }
        if n_walkers <= 0 {
            return Err(GpuError::LaunchFailed(format!(
                "satsolver_extract: n_walkers must be > 0, got {n_walkers}"
            )));
        }
        if n_vars == 0 {
            return Ok(());
        }

        let n_vars = n_vars as usize;
        let n_walkers_usize = n_walkers as usize;
        Self::expect_len(
            "assignment",
            assignment.len(),
            n_walkers_usize.saturating_mul(n_vars),
        )?;
        Self::expect_len("best_walker", best_walker.len(), 1)?;
        Self::expect_len("output", output.len(), n_vars)?;

        let kernels = self.kernels()?;
        let satsolver_extract = kernels.get_function("satsolver_extract")?;
        let stream = Self::new_stream()?;
        let grid = Self::ceil_div_u32(n_vars as u32, SATSOLVER_BLOCK_SIZE);
        let block = SATSOLVER_BLOCK_SIZE;

        unsafe {
            launch!(satsolver_extract<<<grid, block, SATSOLVER_SHARED_MEM_BYTES, stream>>>(
                assignment.as_device_ptr(),
                best_walker.as_device_ptr(),
                output.as_device_ptr(),
                n_vars as i32,
                n_walkers,
            ))
            .map_err(|e| GpuError::LaunchFailed(format!("satsolver_extract launch: {e:?}")))?;
        }

        stream
            .synchronize()
            .map_err(|e| GpuError::LaunchFailed(format!("satsolver_extract sync: {e:?}")))?;
        Ok(())
    }

    /// On-device SAAQ reduction: pass 1 emits one partial winner per block, then a single
    /// warp-sized pass 2 reduces those partials to one final best walker.
    /// Launches on `stream` after `gif_step_weighted`; synchronizes before the minimal device read.
    pub fn saaq_find_best_walker(
        &mut self,
        stream: &Stream,
        neuron_count: usize,
    ) -> GpuResult<u32> {
        self.ensure_temporal_state(neuron_count)?;

        let state = self
            .temporal_state
            .as_mut()
            .ok_or_else(|| GpuError::MemoryError("temporal state not initialised".into()))?;

        let adaptation_scale = 0.22f32; // matches GIF_ADAPTATION_SCALE from spiking_network.cu

        ffi::launch_saaq_find_best_walker(
            stream,
            TEMPORAL_GRID_SIZE,
            TEMPORAL_BLOCK_SIZE,
            0,
            state.membrane.as_device_ptr(),
            state.adaptation.as_device_ptr(),
            state.saaq_partial_scores.as_device_ptr(),
            state.saaq_partial_walkers.as_device_ptr(),
            state.best_walker.as_device_ptr(),
            neuron_count as i32,
            adaptation_scale,
        )?;

        stream
            .synchronize()
            .map_err(|e| GpuError::LaunchFailed(format!("saaq_find_best_walker sync: {e:?}")))?;

        let best = state.best_walker.to_vec()?;
        Ok(best[0])
    }

    /// Recompute SAT clause flags and reduce scores to one `(best_score, best_walker)`
    /// pair using the two-pass atomics-free CUDA path.
    ///
    /// `best_score` and `best_walker` must each be length 1. Intermediate per-block
    /// partial buffers are allocated internally using `grid.x = ceil_div(n_walkers, 256)`.
    #[allow(clippy::too_many_arguments)]
    pub fn satsolver_aux_reduce_best(
        &self,
        assignment: &GpuBuffer<u8>,
        sat_flags: &mut GpuBuffer<u8>,
        scores: &GpuBuffer<i32>,
        best_score: &mut GpuBuffer<i32>,
        best_walker: &mut GpuBuffer<i32>,
        clauses: &GpuBuffer<i32>,
        n_walkers: i32,
        n_vars: i32,
        n_clauses: i32,
        clause_len: i32,
    ) -> GpuResult<()> {
        if n_walkers <= 0 {
            return Err(GpuError::LaunchFailed(format!(
                "satsolver_aux_reduce_best: n_walkers must be > 0, got {n_walkers}"
            )));
        }
        if n_vars < 0 {
            return Err(GpuError::LaunchFailed(format!(
                "satsolver_aux_reduce_best: n_vars must be >= 0, got {n_vars}"
            )));
        }
        if n_clauses < 0 {
            return Err(GpuError::LaunchFailed(format!(
                "satsolver_aux_reduce_best: n_clauses must be >= 0, got {n_clauses}"
            )));
        }
        if clause_len < 0 {
            return Err(GpuError::LaunchFailed(format!(
                "satsolver_aux_reduce_best: clause_len must be >= 0, got {clause_len}"
            )));
        }

        let n_walkers_usize = n_walkers as usize;
        let n_vars_usize = n_vars as usize;
        let n_clauses_usize = n_clauses as usize;
        let clause_len_usize = clause_len as usize;

        Self::expect_len(
            "assignment",
            assignment.len(),
            n_walkers_usize.saturating_mul(n_vars_usize),
        )?;
        Self::expect_len(
            "sat_flags",
            sat_flags.len(),
            n_walkers_usize.saturating_mul(n_clauses_usize),
        )?;
        Self::expect_len("scores", scores.len(), n_walkers_usize)?;
        Self::expect_len("best_score", best_score.len(), 1)?;
        Self::expect_len("best_walker", best_walker.len(), 1)?;
        Self::expect_len(
            "clauses",
            clauses.len(),
            n_clauses_usize.saturating_mul(clause_len_usize),
        )?;

        let kernels = self.kernels()?;
        let satsolver_aux_update = kernels.get_function("satsolver_aux_update")?;
        let satsolver_best_reduce_pass2 = kernels.get_function("satsolver_best_reduce_pass2")?;
        let stream = Self::new_stream()?;
        let grid_x = Self::ceil_div_u32(n_walkers as u32, SATSOLVER_BLOCK_SIZE);
        let block = SATSOLVER_BLOCK_SIZE;
        let partial_len = grid_x as usize;
        let partial_scores = GpuBuffer::<i32>::alloc(partial_len)?;
        let partial_walkers = GpuBuffer::<i32>::alloc(partial_len)?;

        unsafe {
            launch!(satsolver_aux_update<<<grid_x, block, SATSOLVER_SHARED_MEM_BYTES, stream>>>(
                assignment.as_device_ptr(),
                sat_flags.as_device_ptr(),
                scores.as_device_ptr(),
                partial_scores.as_device_ptr(),
                partial_walkers.as_device_ptr(),
                clauses.as_device_ptr(),
                n_walkers,
                n_vars,
                n_clauses,
                clause_len,
            ))
            .map_err(|e| GpuError::LaunchFailed(format!("satsolver_aux_update launch: {e:?}")))?;

            launch!(satsolver_best_reduce_pass2<<<1u32, block, SATSOLVER_SHARED_MEM_BYTES, stream>>>(
                partial_scores.as_device_ptr(),
                partial_walkers.as_device_ptr(),
                best_score.as_device_ptr(),
                best_walker.as_device_ptr(),
                partial_len as i32,
            ))
            .map_err(|e| {
                GpuError::LaunchFailed(format!("satsolver_best_reduce_pass2 launch: {e:?}"))
            })?;
        }

        stream.synchronize().map_err(|e| {
            GpuError::LaunchFailed(format!("satsolver_aux_reduce_best sync: {e:?}"))
        })?;
        Ok(())
    }

    fn new_stream() -> GpuResult<Stream> {
        Stream::new(StreamFlags::DEFAULT, None)
            .map_err(|e| GpuError::LaunchFailed(format!("stream creation failed: {e:?}")))
    }

    fn expect_len(name: &str, actual: usize, minimum: usize) -> GpuResult<()> {
        if actual < minimum {
            return Err(GpuError::MemoryError(format!(
                "{name} too small: need at least {minimum} elements, got {actual}"
            )));
        }
        Ok(())
    }

    fn ceil_div_u32(value: u32, divisor: u32) -> u32 {
        value.div_ceil(divisor)
    }

    fn build_temporal_state(neuron_count: usize) -> GpuResult<TemporalState> {
        let n_inputs = neuron_count; // full connectivity for weighted GIF SNN (sparse can be added later)
        let weight_size = neuron_count * n_inputs;
        let saaq_partials_len = TEMPORAL_GRID_SIZE as usize;
        Ok(TemporalState {
            neuron_count,
            n_inputs,
            membrane: GpuBuffer::<f32>::from_slice(&vec![0.0f32; neuron_count])?,
            refractory: GpuBuffer::<u32>::from_slice(&vec![0u32; neuron_count])?,
            spikes_out: GpuBuffer::<u32>::from_slice(&vec![0u32; neuron_count])?,
            input_current: GpuBuffer::<f32>::from_slice(&vec![0.0f32; neuron_count])?,
            input_spikes: GpuBuffer::<f32>::from_slice(&vec![0.0f32; n_inputs])?,
            adaptation: GpuBuffer::<f32>::from_slice(&vec![0.0f32; neuron_count])?,
            weights_f32: GpuBuffer::<f32>::from_slice(&vec![0.0f32; weight_size])?,
            weights_f16: None,
            synapse_precision: SynapsePrecision::None,
            synapse_signature: None,
            snapshot: GpuBuffer::<f32>::from_slice(&[0.0f32; SNAPSHOT_CHANNELS])?,
            best_walker: GpuBuffer::<u32>::from_slice(&[0u32; 1])?,
            saaq_partial_scores: GpuBuffer::<f32>::from_slice(&vec![0.0f32; saaq_partials_len])?,
            saaq_partial_walkers: GpuBuffer::<u32>::from_slice(&vec![0u32; saaq_partials_len])?,
        })
    }

    #[cfg(test)]
    pub(crate) fn new_stub_for_tests() -> Self {
        Self {
            _ctx: None,
            modules: None,
            temporal_state: None,
        }
    }
}

impl Default for GpuAccelerator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::wrappers::context::GpuContext;

    #[test]
    fn test_temporal_state_requires_gpu() {
        let mut accelerator = GpuAccelerator::new_stub_for_tests();
        let err = accelerator.ensure_temporal_state(2048).unwrap_err();
        assert!(matches!(err, GpuError::NoGpu));
    }

    #[test]
    #[ignore] // requires GPU + driver ≥ 570
    fn test_saaq_two_pass_reduction_selects_global_best_and_tie_breaks() {
        if !GpuContext::is_available() {
            return;
        }

        let mut accelerator = GpuAccelerator::new();
        if !accelerator.is_ready() {
            eprintln!("Skipping SAAQ ignored test because GPU accelerator is not fully ready");
            return;
        }

        let neuron_count = (TEMPORAL_GRID_SIZE * TEMPORAL_BLOCK_SIZE) as usize;
        accelerator
            .ensure_temporal_state(neuron_count)
            .expect("temporal state should allocate");

        let mut membrane = vec![0.0f32; neuron_count];
        let mut adaptation = vec![0.0f32; neuron_count];
        membrane[17] = 2.0;
        membrane[5 * TEMPORAL_BLOCK_SIZE as usize + 9] = 4.5;

        {
            let state = accelerator
                .temporal_state
                .as_mut()
                .expect("temporal state should exist");
            state
                .membrane
                .upload(&membrane)
                .expect("membrane upload should succeed");
            state
                .adaptation
                .upload(&adaptation)
                .expect("adaptation upload should succeed");
        }

        let stream = GpuAccelerator::new_stream().expect("stream should create");
        let best = accelerator
            .saaq_find_best_walker(&stream, neuron_count)
            .expect("SAAQ reduction should succeed");
        assert_eq!(best, 5 * TEMPORAL_BLOCK_SIZE + 9);

        membrane.fill(0.0);
        adaptation.fill(0.0);
        membrane[11] = 3.0;
        membrane[3 * TEMPORAL_BLOCK_SIZE as usize + 4] = 3.0;

        let stream = GpuAccelerator::new_stream().expect("stream should create");
        let tie_best = accelerator
            .saaq_find_best_walker(&stream, neuron_count)
            .expect("SAAQ reduction should succeed");
        assert_eq!(tie_best, 11);
    }
}
