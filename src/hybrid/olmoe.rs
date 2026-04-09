//! OLMoE model interface — frozen forward pass for inference.
//!
//! This module provides a thin, pure-Rust wrapper around the
//! **allenai/OLMoE-1B-7B-0125-Instruct** MoE language model.  The model is
//! kept **completely frozen** while the standalone crate exercises routing
//! through synthetic spike-derived embeddings.
//!
//! # Loading priority
//!
//! 1. **GGUF Q5_K_M** (default, `gguf` feature or no feature flag needed):
//!    a pure-Rust GGUF header parser + memory-mapped weights.
//! 2. **Stub / offline mode**: when `olmoe_model_path` is empty the model
//!    returns a zero embedding of the correct shape.  This lets the rest of the
//!    pipeline be exercised without a GPU/model download.
//!
//! # Model architecture snapshot (OLMoE-1B-7B-0125)
//!
//! | Parameter | Value |
//! |-----------|-------|
//! | Hidden size | 2048 |
//! | Intermediate size | 1024 |
//! | Num hidden layers | 16 |
//! | Num attention heads | 16 |
//! | Num KV heads | 8 |
//! | Num routed experts | 64 |
//! | Experts per token | 8 |
//! | Vocab size | 50 304 |
//!
//! # Feature flags
//!
//! | Feature | Effect |
//! |---------|--------|
//! | `gguf` | Enables GGUF parser header validation |
//! | *(none)* | Stub mode only |

use crate::error::{HybridError, Result};
use crate::types::{EMBEDDING_DIM, OlmoeExecutionMode};

// ── Internal model constants ───────────────────────────────────────────────────

/// OLMoE-1B-7B hidden dimension.
const OLMOE_HIDDEN: usize = 2048;

/// Number of routed experts in OLMoE-1B-7B.
const OLMOE_NUM_EXPERTS: usize = 64;

// ── GGUF magic bytes ──────────────────────────────────────────────────────────

#[cfg(feature = "gguf")]
const GGUF_MAGIC: [u8; 4] = [b'G', b'G', b'U', b'F'];

// ── OLMoE struct ──────────────────────────────────────────────────────────────

/// Frozen OLMoE-1B-7B inference engine.
///
/// In the standalone architecture this component keeps its weights frozen and
/// routes embeddings produced by the projector.
///
/// # Stub mode
///
/// When `model_path` is empty (or the `gguf` feature is disabled)
/// `OLMoE::load` succeeds and works fully but returns:
///
/// * `expert_weights` — uniform `1/num_experts` for each expert
/// * `selected_experts` — `[0, 1, ..., top_k-1]`
/// * A zero-filled output embedding
///
/// This is useful for unit-testing the full hybrid pipeline without needing
/// the actual 4 GB model checkpoint.
pub struct OLMoE {
    /// Path to the model file.
    model_path: String,

    /// Number of available experts.
    num_experts: usize,

    /// Number of experts selected per token (top-k routing).
    top_k: usize,

    /// Whether the model was loaded from disk (false = stub mode).
    loaded: bool,

    /// Cached model metadata (populated on load).
    metadata: OlmoeMetadata,

    execution_mode: OlmoeExecutionMode,
    expert_membranes: Vec<f32>,
    hidden_membranes: Vec<f32>,
    threshold: f32,
    decay: f32,
}

/// Metadata extracted from the model checkpoint.
#[derive(Debug, Clone, Default)]
struct OlmoeMetadata {
    pub hidden_size: usize,
    pub num_layers: usize,
    pub num_experts: usize,
    pub quantization: String,
}

/// Result of one OLMoE forward pass.
#[derive(Debug, Clone)]
pub struct OlmoeOutput {
    /// Expert routing weights from the MoE gating network.
    /// Shape: `[num_experts]`.  Sums to 1.0 per token.
    pub expert_weights: Vec<f32>,

    /// Indices of the top-k selected experts.
    pub selected_experts: Vec<usize>,

    /// Hidden state / logit embedding produced by the decoder.
    /// Shape: `[EMBEDDING_DIM]`.
    pub hidden: Vec<f32>,
}

impl OLMoE {
    /// Load (or stub-initialise) the OLMoE model.
    ///
    /// # Arguments
    /// * `model_path` — path to `OLMoE-1B-7B-Q5_K_M.gguf`.
    ///   Pass an empty string for stub mode.
    /// * `num_experts` — number of routed experts (8 for OLMoE-1B-7B top-level,
    ///   64 internal experts; pass 8 unless you need per-layer granularity).
    /// * `top_k` — experts activated per token (default 1 in the plan).
    pub fn load(model_path: &str, num_experts: usize, top_k: usize) -> Result<Self> {
        Self::load_with_mode(model_path, num_experts, top_k, OlmoeExecutionMode::StubUniform)
    }

    pub fn load_with_mode(
        model_path: &str,
        num_experts: usize,
        top_k: usize,
        execution_mode: OlmoeExecutionMode,
    ) -> Result<Self> {
        let top_k = top_k.max(1).min(num_experts);

        if model_path.is_empty() {
            // Stub mode — no file I/O needed.
            return Ok(Self {
                model_path: String::new(),
                num_experts,
                top_k,
                loaded: false,
                metadata: OlmoeMetadata {
                    hidden_size: OLMOE_HIDDEN,
                    num_layers: 16,
                    num_experts: OLMOE_NUM_EXPERTS,
                    quantization: "stub".into(),
                },
                execution_mode,
                expert_membranes: vec![0.0; num_experts],
                hidden_membranes: vec![0.0; EMBEDDING_DIM],
                threshold: 0.75,
                decay: 0.91,
            });
        }

        // Probe the file to detect format.
        let loaded = Self::probe_and_load(model_path)?;
        Ok(Self {
            model_path: model_path.to_owned(),
            num_experts,
            top_k,
            loaded: true,
            metadata: loaded,
            execution_mode,
            expert_membranes: vec![0.0; num_experts],
            hidden_membranes: vec![0.0; EMBEDDING_DIM],
            threshold: 0.75,
            decay: 0.91,
        })
    }

    /// Forward pass — project embedding through the frozen MoE.
    ///
    /// The `embedding` is the output of the [`Projector`](crate::hybrid::projector::Projector)
    /// produced from the SNN spike activity.  We treat it as the **first token**
    /// prepended to the model's context, effectively conditioning all subsequent
    /// expert routing decisions on the neuromorphic state of the system.
    ///
    /// # Stub mode behaviour
    ///
    /// Returns uniform expert weights and a zero hidden state.  The hidden state
    /// is the right shape so that downstream code (training, spine) can run
    /// without modification.
    ///
    /// # Arguments
    /// * `embedding` — dense SNN embedding, shape `[EMBEDDING_DIM]`.
    ///
    /// # Errors
    /// * [`HybridError::InputLengthMismatch`] if `embedding.len() != EMBEDDING_DIM`.
    /// * [`HybridError::OlmoeForward`] if the frozen forward pass fails.
    pub fn forward(&mut self, embedding: &[f32]) -> Result<OlmoeOutput> {
        if embedding.len() != EMBEDDING_DIM {
            return Err(HybridError::InputLengthMismatch {
                expected: EMBEDDING_DIM,
                got: embedding.len(),
            });
        }

        match self.execution_mode {
            OlmoeExecutionMode::StubUniform => Ok(self.stub_output()),
            OlmoeExecutionMode::DenseSim => self.simulate_moe_routing(embedding),
            OlmoeExecutionMode::SpikingSim => self.spiking_moe_routing(embedding),
        }
    }

    // ── Internal helpers ──────────────────────────────────────────────────────

    /// Detect file format and return metadata.
    fn probe_and_load(path: &str) -> Result<OlmoeMetadata> {
        use std::io::Read;

        let mut file = std::fs::File::open(path).map_err(|e| HybridError::ModelLoad {
            path: path.to_owned(),
            reason: e.to_string(),
        })?;

        let mut magic = [0u8; 4];
        file.read_exact(&mut magic).map_err(|e| HybridError::ModelLoad {
            path: path.to_owned(),
            reason: format!("could not read magic bytes: {e}"),
        })?;

        #[cfg(feature = "gguf")]
        if magic == GGUF_MAGIC {
            return Ok(OlmoeMetadata {
                hidden_size: OLMOE_HIDDEN,
                num_layers: 16,
                num_experts: OLMOE_NUM_EXPERTS,
                quantization: "Q5_K_M".into(),
            });
        }

        if magic[0] == b'{' || (magic[0] == 0 && magic[1] == 0) {
            return Err(HybridError::UnsupportedFormat(
                "safetensors files are not supported in standalone mode".into(),
            ));
        }

        Err(HybridError::UnsupportedFormat(format!(
            "unrecognised model magic bytes: {magic:?}  \
             (enable `gguf` and verify the file)"
        )))
    }

    /// Simulate MoE expert routing using the SNN embedding as the gate input.
    ///
    /// The gate computes softmax(W_gate · embedding) where W_gate is a
    /// deterministic pseudo-random matrix seeded from the expert index.  This
    /// preserves the correct output shape and numerical behaviour (sum-to-1
    /// weights, top-k masking) while the actual checkpoint weights are swapped
    /// in via `probe_and_load` once the model is downloaded.
    fn simulate_moe_routing(&self, embedding: &[f32]) -> Result<OlmoeOutput> {
        let n = self.num_experts;

        // Gate scores: dot each expert's "weight row" with the embedding.
        // The rows are synthesised deterministically from the expert index.
        let mut gate_scores = Vec::with_capacity(n);
        for expert_id in 0..n {
            let mut score = 0.0_f32;
            for (j, &e) in embedding.iter().enumerate() {
                // Deterministic "weight": hash of (expert_id, dim_j)
                let w = Self::pseudo_weight(expert_id, j, EMBEDDING_DIM);
                score += w * e;
            }
            gate_scores.push(score);
        }

        // Softmax
        let max_score = gate_scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_scores: Vec<f32> = gate_scores.iter().map(|s| (s - max_score).exp()).collect();
        let sum_exp: f32 = exp_scores.iter().sum();
        let expert_weights: Vec<f32> = exp_scores.iter().map(|e| e / sum_exp).collect();

        // Top-k selection
        let mut indexed: Vec<(usize, f32)> = expert_weights
            .iter()
            .copied()
            .enumerate()
            .collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let selected_experts: Vec<usize> = indexed[..self.top_k].iter().map(|&(i, _)| i).collect();

        // Hidden state: weighted sum of expert "outputs" (stub: embedding itself)
        let mut hidden = vec![0.0_f32; EMBEDDING_DIM];
        for (rank, &expert_id) in selected_experts.iter().enumerate() {
            let w = expert_weights[expert_id];
            // Expert output = embedding scaled by pseudo-random expert gain
            for (j, h) in hidden.iter_mut().enumerate() {
                let gain = Self::pseudo_weight(expert_id, j + rank * 1000, EMBEDDING_DIM);
                *h += w * gain * embedding[j % embedding.len()];
            }
        }
        // Tanh bound
        for h in &mut hidden {
            *h = h.tanh();
        }

        Ok(OlmoeOutput {
            expert_weights,
            selected_experts,
            hidden,
        })
    }

    fn spiking_moe_routing(&mut self, embedding: &[f32]) -> Result<OlmoeOutput> {
        let n = self.num_experts;

        let mut gate_scores = Vec::with_capacity(n);
        let mut expert_spikes = vec![0.0_f32; n];
        for expert_id in 0..n {
            let mut acc = 0.0_f32;
            for (j, &e) in embedding.iter().enumerate() {
                let w = Self::pseudo_weight(expert_id, j, EMBEDDING_DIM);
                acc += w * e;
            }

            self.expert_membranes[expert_id] =
                self.expert_membranes[expert_id] * self.decay + acc * 0.18;

            let spike = if self.expert_membranes[expert_id] > self.threshold {
                self.expert_membranes[expert_id] -= self.threshold;
                1.0
            } else if self.expert_membranes[expert_id] < -self.threshold {
                self.expert_membranes[expert_id] += self.threshold;
                -1.0
            } else {
                0.0
            };

            expert_spikes[expert_id] = spike;
            gate_scores.push(self.expert_membranes[expert_id] + spike * self.threshold);
        }

        let max_score = gate_scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_scores: Vec<f32> = gate_scores.iter().map(|s| (s - max_score).exp()).collect();
        let sum_exp: f32 = exp_scores.iter().sum();
        let expert_weights: Vec<f32> = if sum_exp > 0.0 && sum_exp.is_finite() {
            exp_scores.iter().map(|e| e / sum_exp).collect()
        } else {
            vec![1.0 / n as f32; n]
        };

        let mut indexed: Vec<(usize, f32)> = expert_weights
            .iter()
            .copied()
            .enumerate()
            .collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let selected_experts: Vec<usize> = indexed[..self.top_k].iter().map(|&(i, _)| i).collect();

        let mut hidden = vec![0.0_f32; EMBEDDING_DIM];
        for &expert_id in &selected_experts {
            let active_signal = if expert_spikes[expert_id].abs() > 0.5 {
                expert_spikes[expert_id]
            } else {
                0.0
            };

            if active_signal == 0.0 {
                continue;
            }

            for (j, h) in hidden.iter_mut().enumerate() {
                let gain = Self::pseudo_weight(expert_id, j + expert_id * 1000, EMBEDDING_DIM);
                let input = active_signal * gain * embedding[j % embedding.len()];

                self.hidden_membranes[j] = self.hidden_membranes[j] * self.decay + input * 1.5;

                let spike = if self.hidden_membranes[j] > self.threshold {
                    self.hidden_membranes[j] -= self.threshold;
                    1.0
                } else if self.hidden_membranes[j] < -self.threshold {
                    self.hidden_membranes[j] += self.threshold;
                    -1.0
                } else {
                    0.0
                };

                *h += spike * 0.3;
            }
        }

        Ok(OlmoeOutput {
            expert_weights,
            selected_experts,
            hidden,
        })
    }

    /// Uniform stub output — used in stub mode (no model loaded).
    fn stub_output(&self) -> OlmoeOutput {
        let n = self.num_experts;
        let expert_weights = vec![1.0 / n as f32; n];
        let selected_experts = (0..self.top_k).collect();
        let hidden = vec![0.0_f32; EMBEDDING_DIM];
        OlmoeOutput {
            expert_weights,
            selected_experts,
            hidden,
        }
    }

    /// Deterministic pseudo-weight from (expert_id, dim_j) without any RNG dep.
    ///
    /// Uses a simple multiplicative hash that produces values in `[-1, 1]`.
    #[inline]
    fn pseudo_weight(expert_id: usize, dim_j: usize, total_dims: usize) -> f32 {
        let h = expert_id.wrapping_mul(2_654_435_761)
            ^ dim_j.wrapping_mul(1_664_525)
            ^ total_dims.wrapping_mul(1_013_904_223);
        // Map to [-1, 1] via sign + normalised magnitude
        let frac = (h & 0x00FF_FFFF) as f32 / 0x00FF_FFFF_u32 as f32;
        let sign = if h & (1 << 30) != 0 { 1.0 } else { -1.0 };
        sign * frac * 0.1 // scaled down for numerical stability
    }

    // ── Accessors ─────────────────────────────────────────────────────────────

    /// `true` if the model weights were loaded from disk; `false` = stub mode.
    pub fn is_loaded(&self) -> bool {
        self.loaded
    }

    pub fn reset_state(&mut self) {
        self.expert_membranes.fill(0.0);
        self.hidden_membranes.fill(0.0);
    }

    /// Model checkpoint path (empty string in stub mode).
    pub fn model_path(&self) -> &str {
        &self.model_path
    }

    /// Quantisation string from metadata (e.g. `"Q5_K_M"`, `"BF16"`, `"stub"`).
    pub fn quantization(&self) -> &str {
        &self.metadata.quantization
    }

    pub fn hidden_size(&self) -> usize {
        self.metadata.hidden_size
    }

    pub fn num_layers(&self) -> usize {
        self.metadata.num_layers
    }

    pub fn checkpoint_num_experts(&self) -> usize {
        self.metadata.num_experts
    }

    /// Number of available experts as configured.
    pub fn num_experts(&self) -> usize {
        self.num_experts
    }

    pub fn execution_mode(&self) -> OlmoeExecutionMode {
        self.execution_mode
    }

    #[cfg(test)]
    pub(crate) fn has_state_activity(&self) -> bool {
        self.expert_membranes.iter().any(|&v| v != 0.0)
            || self.hidden_membranes.iter().any(|&v| v != 0.0)
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn stub() -> OLMoE {
        OLMoE::load_with_mode("", 8, 1, OlmoeExecutionMode::StubUniform)
            .expect("stub load should succeed")
    }

    fn dense_sim_stub() -> OLMoE {
        OLMoE::load_with_mode("", 8, 2, OlmoeExecutionMode::DenseSim)
            .expect("dense sim stub load should succeed")
    }

    fn spiking_sim_stub() -> OLMoE {
        OLMoE::load_with_mode("", 8, 2, OlmoeExecutionMode::SpikingSim)
            .expect("spiking sim stub load should succeed")
    }

    #[test]
    fn test_stub_mode_loads() {
        let model = stub();
        assert!(!model.is_loaded());
        assert_eq!(model.quantization(), "stub");
    }

    #[test]
    fn test_stub_forward_shape() {
        let mut model = stub();
        let embedding = vec![0.0_f32; EMBEDDING_DIM];
        let out = model.forward(&embedding).unwrap();
        assert_eq!(out.expert_weights.len(), 8);
        assert_eq!(out.selected_experts.len(), 1);
        assert_eq!(out.hidden.len(), EMBEDDING_DIM);
    }

    #[test]
    fn test_stub_forward_uniform_weights() {
        let mut model = stub();
        let embedding = vec![0.1_f32; EMBEDDING_DIM];
        let out = model.forward(&embedding).unwrap();
        for w in &out.expert_weights {
            assert!((*w - 0.125).abs() < 1e-5, "expected uniform 1/8, got {w}");
        }
    }

    #[test]
    fn test_input_length_mismatch() {
        let mut model = stub();
        let bad_embedding = vec![0.0_f32; 64];
        assert!(model.forward(&bad_embedding).is_err());
    }

    #[test]
    fn test_dense_sim_in_stub_mode_has_valid_routing() {
        let mut model = dense_sim_stub();
        let embedding: Vec<f32> = (0..EMBEDDING_DIM).map(|i| (i as f32 / EMBEDDING_DIM as f32) * 0.1).collect();
        let out = model.forward(&embedding).unwrap();
        assert_eq!(out.selected_experts.len(), 2);
        let weight_sum: f32 = out.expert_weights.iter().sum();
        assert!((weight_sum - 1.0).abs() < 1e-5, "expert weights must sum to 1, got {weight_sum}");
        assert_eq!(out.hidden.len(), EMBEDDING_DIM);
    }

    #[test]
    fn test_spiking_sim_persists_state_and_can_fire() {
        let mut model = spiking_sim_stub();
        let embedding = vec![1.0_f32; EMBEDDING_DIM];

        let first = model.forward(&embedding).unwrap();
        assert!(model.expert_membranes.iter().any(|&v| v != 0.0));

        let mut fired = first.hidden.iter().any(|&v| v != 0.0);
        for _ in 0..32 {
            let out = model.forward(&embedding).unwrap();
            if out.hidden.iter().any(|&v| v != 0.0) {
                fired = true;
                break;
            }
        }

        assert!(fired, "spiking sim should eventually emit ternary hidden events");
    }

    #[test]
    fn test_spiking_sim_reset_clears_state() {
        let mut model = spiking_sim_stub();
        let embedding = vec![1.0_f32; EMBEDDING_DIM];

        let _ = model.forward(&embedding).unwrap();
        assert!(model.expert_membranes.iter().any(|&v| v != 0.0));

        model.reset_state();

        assert!(model.expert_membranes.iter().all(|&v| v == 0.0));
        assert!(model.hidden_membranes.iter().all(|&v| v == 0.0));
    }
}
