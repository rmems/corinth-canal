//! Spike-to-expert projector — the heart of the neuromorphic-ANN fusion.
//!
//! The [`Projector`] sits between the Spikenaut SNN and OLMoE's expert router.
//! Its job is to compress the high-dimensional spike activity (neuron indices +
//! firing rates + membrane potentials) into a single dense embedding vector
//! that OLMoE can use as a context prefix.
//!
//! # Projection modes
//!
//! | Mode | Description | Best for |
//! |------|-------------|----------|
//! | [`RateSum`] | Per-neuron spike-count → linear projection | rate-coded telemetry |
//! | [`TemporalHistogram`] | Spikes binned over time → flatten | timing-sensitive HFT |
//! | [`MembraneSnapshot`] | Post-step membrane potentials → linear | smooth gradient flow |
//!
//! # No OLMoE dependency
//!
//! The projector is intentionally **pure** — it depends only on the spike
//! activity represented as spike indices, potentials, and adaptive voltages,
//! and emits a plain `Vec<f32>`.
//! This keeps it reusable with any LLM backend.
//!
//! [`RateSum`]: crate::types::ProjectionMode::RateSum
//! [`TemporalHistogram`]: crate::types::ProjectionMode::TemporalHistogram
//! [`MembraneSnapshot`]: crate::types::ProjectionMode::MembraneSnapshot

use crate::error::{HybridError, Result};
use crate::types::{EMBEDDING_DIM, ProjectionMode};

// ── Projection weight matrix ───────────────────────────────────────────────────

/// Number of neurons in the Spikenaut SNN.
const SNN_NEURONS: usize = 2048;
const TEMPORAL_BINS: usize = 4;

/// Number of Izhikevich neurons in the adaptive bank.
const IZ_NEURONS: usize = 5;

fn feature_dim_for(snn_neurons: usize) -> usize {
    snn_neurons + (snn_neurons * TEMPORAL_BINS) + snn_neurons + IZ_NEURONS
}

// ── Projector ─────────────────────────────────────────────────────────────────

/// Converts Spikenaut SNN output into a dense embedding for OLMoE.
///
/// Internally this is a **learned linear layer** `W ∈ ℝ^{EMBEDDING_DIM × FEATURE_DIM}`
/// plus a bias `b ∈ ℝ^{EMBEDDING_DIM}`, initialised with Xavier-uniform weights.
/// The weight matrix is updated only by Julia/E-prop via the spine; it is
/// **frozen from the Rust side**.
///
/// When [`ProjectionMode::SpikingTernary`] is selected the projection uses
/// GIF (Generalised Integrate-and-Fire) membrane integration and fires ternary
/// spikes (-1.0 / 0.0 / 1.0), producing a sparse event-driven embedding.
/// Membrane state persists across calls; call [`reset_membrane`](Self::reset_membrane)
/// to clear it (e.g. on episode boundaries).
///
/// ```no_run
/// use corinth_canal::hybrid::projector::Projector;
/// use corinth_canal::types::ProjectionMode;
///
/// let proj = Projector::new(ProjectionMode::RateSum);
/// ```
pub struct Projector {
    /// Projection strategy.
    mode: ProjectionMode,

    snn_neurons: usize,

    feature_dim: usize,

    /// Flat weight matrix `W`, row-major layout: `W[out * FEATURE_DIM + in]`.
    /// Shape: `[EMBEDDING_DIM, FEATURE_DIM]`.
    weights: Vec<f32>,

    /// Bias vector `b`.  Shape: `[EMBEDDING_DIM]`.
    bias: Vec<f32>,

    /// Running exponential moving average of firing rates (for normalisation).
    /// Updated each call to [`project`](Self::project).
    rate_ema: Vec<f32>,

    /// EMA decay constant for firing rate normalisation.
    ema_alpha: f32,

    // ── SpikingTernary state (GIF membrane) ───────────────────────────────────
    /// Per-output GIF membrane potential.  Shape: `[EMBEDDING_DIM]`.
    /// Only mutated when `mode == SpikingTernary`; always allocated so that
    /// switching modes at runtime has zero cost.
    membrane: Vec<f32>,

    /// Membrane firing threshold.  Crossed → ternary ±1 spike + soft reset.
    threshold: f32,

    /// GIF leak factor applied each integration step (0 < decay < 1).
    decay: f32,
}

impl Projector {
    /// Create a new Projector with Xavier-uniform initialised weights.
    ///
    /// # Arguments
    /// * `mode` — how to aggregate spike activity into a feature vector.
    pub fn new(mode: ProjectionMode) -> Self {
        Self::with_input_neurons(mode, SNN_NEURONS)
    }

    pub fn with_input_neurons(mode: ProjectionMode, snn_neurons: usize) -> Self {
        let snn_neurons = snn_neurons.max(1);
        let feature_dim = feature_dim_for(snn_neurons);
        let fan_in = feature_dim as f32;
        let fan_out = EMBEDDING_DIM as f32;
        let limit = (6.0_f32 / (fan_in + fan_out)).sqrt();
        const GOLDEN_RATIO_FRAC: f32 = 1.618_034;

        // Deterministic Xavier-uniform init (no external rng dep needed).
        let mut weights = Vec::with_capacity(EMBEDDING_DIM * feature_dim);
        for i in 0..(EMBEDDING_DIM * feature_dim) {
            // Simple deterministic pseudo-random from index hash.
            let t = ((i as f32 * GOLDEN_RATIO_FRAC) % 1.0) * 2.0 - 1.0;
            weights.push(t * limit);
        }

        Self {
            mode,
            snn_neurons,
            feature_dim,
            weights,
            bias: vec![0.0; EMBEDDING_DIM],
            rate_ema: vec![0.0; snn_neurons],
            ema_alpha: 0.1,
            membrane: vec![0.0; EMBEDDING_DIM],
            threshold: 0.8, // saliency threshold (SpikeLLM / NSLLM style)
            decay: 0.92,    // GIF leak (close to biological membrane RC)
        }
    }

    /// Project SNN spike activity into a dense embedding.
    ///
    /// # Arguments
    /// * `spike_train`  — per-step spike sets from `SpikingNetwork::step`.
    /// * `potentials`   — membrane potentials after the final SNN time-step.
    /// * `iz_potentials`— Izhikevich neuron voltages (5 adaptive neurons).
    ///
    /// # Returns
    /// Dense embedding `Vec<f32>` of length [`EMBEDDING_DIM`].
    pub fn project(
        &mut self,
        spike_train: &[Vec<usize>],
        potentials: &[f32],
        iz_potentials: &[f32],
    ) -> Result<Vec<f32>> {
        if potentials.len() < self.snn_neurons {
            return Err(HybridError::InputLengthMismatch {
                expected: self.snn_neurons,
                got: potentials.len(),
            });
        }

        let feature_vec = self.build_feature_vector(spike_train, potentials, iz_potentials);
        let embedding = match self.mode {
            ProjectionMode::SpikingTernary => self.spiking_linear_project(&feature_vec),
            _ => self.dense_linear_project(&feature_vec),
        };
        Ok(embedding)
    }

    // ── Feature construction ──────────────────────────────────────────────────

    fn build_feature_vector(
        &mut self,
        spike_train: &[Vec<usize>],
        potentials: &[f32],
        iz_potentials: &[f32],
    ) -> Vec<f32> {
        let n_steps = spike_train.len().max(1) as f32;

        // 1. Firing rates per neuron [16 dims]
        let mut rates = vec![0.0_f32; self.snn_neurons];
        for step in spike_train {
            for &idx in step {
                if idx < self.snn_neurons {
                    rates[idx] += 1.0;
                }
            }
        }
        for r in &mut rates {
            *r /= n_steps;
        }

        // Update EMA for normalisation
        for (ema, rate) in self.rate_ema.iter_mut().zip(rates.iter()) {
            *ema = self.ema_alpha * *rate + (1.0 - self.ema_alpha) * *ema;
        }

        // 2. Temporal histogram bins (4 equal-width bins) [64 dims]
        let bins = TEMPORAL_BINS;
        let mut hist = vec![0.0_f32; self.snn_neurons * bins];
        if !spike_train.is_empty() {
            let steps = spike_train.len();
            for (t, step) in spike_train.iter().enumerate() {
                let bin = ((t * bins) / steps).min(bins - 1);
                for &idx in step {
                    if idx < self.snn_neurons {
                        hist[idx * bins + bin] += 1.0;
                    }
                }
            }
            let total = n_steps / bins as f32;
            for h in &mut hist {
                *h /= total.max(1.0);
            }
        }

        // 3. Membrane potentials [16 dims] — clamped to [0, 1]
        let membrane: Vec<f32> = potentials[..self.snn_neurons]
            .iter()
            .map(|&v| v.clamp(0.0, 1.0))
            .collect();

        // 4. Izhikevich adaptive bank potentials [5 dims]
        let iz: Vec<f32> = iz_potentials
            .iter()
            .take(IZ_NEURONS)
            .map(|&v| (v / 30.0).clamp(-1.0, 1.0)) // Izhikevich Vpeak ≈ 30 mV
            .chain(std::iter::repeat(0.0))
            .take(IZ_NEURONS)
            .collect();

        // Mode-specific blending
        let mut features = Vec::with_capacity(self.feature_dim);
        match self.mode {
            ProjectionMode::RateSum => {
                features.extend_from_slice(&rates);
                features.extend_from_slice(&hist);
                features.extend_from_slice(&membrane);
                features.extend_from_slice(&iz);
            }
            ProjectionMode::TemporalHistogram => {
                // Weight histogram more heavily than raw rates
                let weighted_rates: Vec<f32> = rates.iter().map(|r| r * 0.3).collect();
                features.extend_from_slice(&weighted_rates);
                let weighted_hist: Vec<f32> = hist.iter().map(|h| h * 2.0).collect();
                features.extend_from_slice(&weighted_hist);
                features.extend_from_slice(&membrane);
                features.extend_from_slice(&iz);
            }
            ProjectionMode::MembraneSnapshot => {
                // Use membrane directly as primary signal
                let membrane_primary: Vec<f32> = membrane.iter().map(|v| v * 2.0).collect();
                features.extend_from_slice(&rates);
                features.extend_from_slice(&hist);
                features.extend_from_slice(&membrane_primary);
                features.extend_from_slice(&iz);
            }
            ProjectionMode::SpikingTernary => {
                // Same feature blend as RateSum — GIF integration happens in
                // spiking_linear_project, not in the feature vector itself.
                features.extend_from_slice(&rates);
                features.extend_from_slice(&hist);
                features.extend_from_slice(&membrane);
                features.extend_from_slice(&iz);
            }
        }

        // Pad or truncate to exactly FEATURE_DIM
        features.resize(self.feature_dim, 0.0);
        features
    }

    // ── Linear projections ────────────────────────────────────────────────────

    /// Dense W × f + b with tanh squash.  Used for all non-spiking modes.
    fn dense_linear_project(&self, features: &[f32]) -> Vec<f32> {
        let mut out = vec![0.0_f32; EMBEDDING_DIM];
        for (out_i, out_slot) in out.iter_mut().enumerate() {
            let mut acc = self.bias[out_i];
            let row_offset = out_i * self.feature_dim;
            for (in_j, feature) in features.iter().take(self.feature_dim).enumerate() {
                acc += self.weights[row_offset + in_j] * *feature;
            }
            // Layer norm approximation: tanh squash keeps embedding bounded
            *out_slot = acc.tanh();
        }
        out
    }

    /// GIF membrane integration → ternary spike output.
    ///
    /// For each output dimension:
    /// 1. Accumulate W × f + b into `acc`.
    /// 2. Build a bounded drive from the signed learned current plus the
    ///    peak feature activity magnitude.
    /// 3. Integrate: `membrane = membrane * decay + drive * 0.35`.
    /// 3. Fire +1 if membrane > threshold (soft reset: membrane -= threshold).
    /// 4. Fire -1 if membrane < -threshold (soft reset: membrane += threshold).
    /// 5. Otherwise output 0.0.
    fn spiking_linear_project(&mut self, features: &[f32]) -> Vec<f32> {
        let mut spikes = vec![0.0_f32; EMBEDDING_DIM];
        let activity_drive = features.iter().map(|v| v.abs()).fold(0.0_f32, f32::max);
        for (out_i, spike) in spikes.iter_mut().enumerate() {
            let mut acc = self.bias[out_i];
            let row_offset = out_i * self.feature_dim;
            for (in_j, feature) in features.iter().take(self.feature_dim).enumerate() {
                acc += self.weights[row_offset + in_j] * *feature;
            }
            // GIF integration step
            let drive = (acc.tanh() * 0.5 + activity_drive * 0.5).clamp(-1.0, 1.0);
            self.membrane[out_i] = self.membrane[out_i] * self.decay + drive * 0.35;
            if self.membrane[out_i] > self.threshold {
                *spike = 1.0;
                self.membrane[out_i] -= self.threshold; // reset-with-refractory
            } else if self.membrane[out_i] < -self.threshold {
                *spike = -1.0;
                self.membrane[out_i] += self.threshold;
            }
            // else spikes[out_i] remains 0.0
        }
        spikes
    }

    // ── Weight management (for spine / E-prop updates) ────────────────────────

    /// Replace the weight matrix with values received from `SpikenautDistill.jl`.
    ///
    /// Julia sends the updated projector weights as a flat `f32` slice via the
    /// spine after each E-prop step.
    ///
    /// # Errors
    /// Returns [`HybridError::InputLengthMismatch`] if the slice length ≠
    /// `EMBEDDING_DIM × FEATURE_DIM`.
    pub fn load_weights(&mut self, weights: &[f32]) -> Result<()> {
        let expected = EMBEDDING_DIM * self.feature_dim;
        if weights.len() != expected {
            return Err(HybridError::InputLengthMismatch {
                expected,
                got: weights.len(),
            });
        }
        self.weights.copy_from_slice(weights);
        Ok(())
    }

    /// Replace the bias vector.
    pub fn load_bias(&mut self, bias: &[f32]) -> Result<()> {
        if bias.len() != EMBEDDING_DIM {
            return Err(HybridError::InputLengthMismatch {
                expected: EMBEDDING_DIM,
                got: bias.len(),
            });
        }
        self.bias.copy_from_slice(bias);
        Ok(())
    }

    /// Reset GIF membrane state to zero.
    ///
    /// Call this on episode/session boundaries when using
    /// [`ProjectionMode::SpikingTernary`] to avoid stale membrane history
    /// bleeding across unrelated inputs.  No-op for other modes.
    pub fn reset_membrane(&mut self) {
        self.membrane.fill(0.0);
    }

    /// Current projection mode.
    pub fn mode(&self) -> ProjectionMode {
        self.mode
    }

    /// Dimensionality constants (useful for allocating buffers).
    pub fn dims(&self) -> (usize, usize) {
        (self.feature_dim, EMBEDDING_DIM)
    }

    pub fn input_neurons(&self) -> usize {
        self.snn_neurons
    }

    /// Firing rate EMA snapshot (useful for diagnostics / reward shaping).
    pub fn rate_ema(&self) -> &[f32] {
        &self.rate_ema
    }
}

impl Default for Projector {
    fn default() -> Self {
        Self::new(ProjectionMode::RateSum)
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn dummy_spike_train(n_steps: usize, neurons: usize) -> Vec<Vec<usize>> {
        (0..n_steps)
            .map(|t| vec![t % neurons, (t + 1) % neurons])
            .collect()
    }

    #[test]
    fn test_project_output_length() {
        let mut proj = Projector::new(ProjectionMode::RateSum);
        let spikes = dummy_spike_train(20, SNN_NEURONS);
        let potentials = vec![0.3; SNN_NEURONS];
        let iz_pots = vec![15.0; IZ_NEURONS];
        let embedding = proj.project(&spikes, &potentials, &iz_pots).unwrap();
        assert_eq!(embedding.len(), EMBEDDING_DIM);
    }

    #[test]
    fn test_project_values_bounded() {
        let mut proj = Projector::new(ProjectionMode::TemporalHistogram);
        let spikes = dummy_spike_train(10, SNN_NEURONS);
        let potentials = vec![0.5; SNN_NEURONS];
        let iz_pots = vec![30.0; IZ_NEURONS];
        let embedding = proj.project(&spikes, &potentials, &iz_pots).unwrap();
        for v in &embedding {
            assert!(
                v.abs() <= 1.0 + 1e-6,
                "embedding value {v} out of tanh range [-1, 1]"
            );
        }
    }

    #[test]
    fn test_spiking_ternary_output() {
        let mut proj = Projector::new(ProjectionMode::SpikingTernary);
        let spikes = dummy_spike_train(20, SNN_NEURONS);
        let potentials = vec![0.3; SNN_NEURONS];
        let iz_pots = vec![15.0; IZ_NEURONS];

        let embedding = proj.project(&spikes, &potentials, &iz_pots).unwrap();
        assert_eq!(embedding.len(), EMBEDDING_DIM);

        // All values must be ternary {-1, 0, +1}
        for &v in &embedding {
            assert!(
                (v - 1.0).abs() < 1e-6 || v.abs() < 1e-6 || (v + 1.0).abs() < 1e-6,
                "expected ternary value but got {v}"
            );
        }
    }

    #[test]
    fn test_spiking_ternary_fires_after_warmup() {
        let mut proj = Projector::new(ProjectionMode::SpikingTernary);
        let spikes = dummy_spike_train(20, SNN_NEURONS);
        let potentials = vec![0.9; SNN_NEURONS]; // high activity to charge membranes
        let iz_pots = vec![28.0; IZ_NEURONS];

        // Run several steps to let membranes charge past threshold
        let mut fired = 0usize;
        for _ in 0..10 {
            let emb = proj.project(&spikes, &potentials, &iz_pots).unwrap();
            fired += emb.iter().filter(|&&v| v.abs() > 0.5).count();
        }
        assert!(
            fired > 0,
            "SpikingTernary should have fired at least one spike across 10 steps"
        );
    }

    #[test]
    fn test_reset_membrane_clears_state() {
        let mut proj = Projector::new(ProjectionMode::SpikingTernary);
        let spikes = dummy_spike_train(20, SNN_NEURONS);
        let potentials = vec![0.9; SNN_NEURONS];
        let iz_pots = vec![28.0; IZ_NEURONS];

        // Charge membranes
        for _ in 0..10 {
            proj.project(&spikes, &potentials, &iz_pots).unwrap();
        }

        // After reset all membrane values should be 0
        proj.reset_membrane();
        assert!(
            proj.membrane.iter().all(|&v| v == 0.0),
            "membrane should be zeroed after reset"
        );
    }

    #[test]
    fn test_load_weights_length_check() {
        let mut proj = Projector::new(ProjectionMode::RateSum);
        let bad_weights = vec![0.0f32; 10]; // wrong length
        assert!(proj.load_weights(&bad_weights).is_err());
    }

    #[test]
    fn test_membrane_mode() {
        let mut proj = Projector::new(ProjectionMode::MembraneSnapshot);
        let spikes = dummy_spike_train(5, SNN_NEURONS);
        let potentials = vec![0.8; SNN_NEURONS];
        let iz_pots = vec![0.0; IZ_NEURONS];
        let embedding = proj.project(&spikes, &potentials, &iz_pots).unwrap();
        assert_eq!(embedding.len(), EMBEDDING_DIM);
    }

    #[test]
    fn test_custom_input_neuron_count() {
        let neurons = 512;
        let mut proj = Projector::with_input_neurons(ProjectionMode::RateSum, neurons);
        let spikes = dummy_spike_train(8, neurons);
        let potentials = vec![0.4; neurons];
        let iz_pots = vec![0.0; IZ_NEURONS];
        let embedding = proj.project(&spikes, &potentials, &iz_pots).unwrap();
        let (feature_dim, embedding_dim) = proj.dims();

        assert_eq!(embedding.len(), EMBEDDING_DIM);
        assert_eq!(embedding_dim, EMBEDDING_DIM);
        assert_eq!(feature_dim, feature_dim_for(neurons));
        assert_eq!(proj.input_neurons(), neurons);
    }

    #[test]
    fn test_dims() {
        let proj = Projector::default();
        let (feat, emb) = proj.dims();
        assert_eq!(feat, feature_dim_for(SNN_NEURONS));
        assert_eq!(emb, EMBEDDING_DIM);
    }
}
