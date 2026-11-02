// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Routing math and embedding resampling for the GGUF and Safetensors router bridge.

use super::checkpoint::MappedGgufCheckpoint;
use super::safetensors::MappedSafetensorsCheckpoint;
use crate::error::{HybridError, Result};
use crate::types::EMBEDDING_DIM;
use std::cmp::Ordering;

/// How gate scoring indexes a rank-2 routing tensor's flat weights.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum GateOrientation {
    /// `dims[0] == hidden_size`; a weight sits at `dim * stride + expert_id`.
    HiddenMajor { stride: usize },
    /// `dims[1] == hidden_size`; a weight sits at `expert_id * stride + dim`.
    ExpertMajor { stride: usize },
}

/// The routing-tensor shape rule for both GGUF and Safetensors: one dimension
/// must equal `hidden_size` and the other must be at least `num_experts`.
/// Load-time validation and gate scoring both go through this.
pub(super) fn gate_orientation(
    d0: usize,
    d1: usize,
    hidden_size: usize,
    num_experts: usize,
) -> Option<GateOrientation> {
    if d0 == hidden_size && d1 >= num_experts {
        return Some(GateOrientation::HiddenMajor { stride: d1 });
    }
    if d0 >= num_experts && d1 == hidden_size {
        return Some(GateOrientation::ExpertMajor { stride: d1 });
    }
    None
}

fn gate_scores_from_weights(
    weights: &[f32],
    orientation: GateOrientation,
    num_experts: usize,
    embedding: &[f32],
) -> Vec<f32> {
    let mut gate_scores = Vec::with_capacity(num_experts);
    for expert_id in 0..num_experts {
        let mut score = 0.0f32;
        for (dim, &value) in embedding.iter().enumerate() {
            let index = match orientation {
                GateOrientation::HiddenMajor { stride } => dim * stride + expert_id,
                GateOrientation::ExpertMajor { stride } => expert_id * stride + dim,
            };
            score += weights[index] * value;
        }
        gate_scores.push(score);
    }
    gate_scores
}

pub(super) fn checkpoint_gate_scores(
    checkpoint: &MappedGgufCheckpoint,
    model_path: &str,
    routing_tensor_name: &str,
    num_experts: usize,
    embedding: &[f32],
) -> Result<Vec<f32>> {
    let info = checkpoint.tensor_info(routing_tensor_name, model_path)?;
    let weights = checkpoint.f32_tensor(routing_tensor_name, model_path)?;
    if info.dims.len() != 2 {
        return Err(HybridError::UnsupportedFormat(format!(
            "routing tensor '{routing_tensor_name}' must be rank-2, got {:?}",
            info.dims
        )));
    }
    let orientation = gate_orientation(info.dims[0], info.dims[1], embedding.len(), num_experts)
        .ok_or_else(|| {
            HybridError::UnsupportedFormat(format!(
                "unsupported routing tensor orientation {:?} for hidden_size={}",
                info.dims,
                embedding.len()
            ))
        })?;
    Ok(gate_scores_from_weights(
        weights,
        orientation,
        num_experts,
        embedding,
    ))
}

pub(super) fn safetensors_gate_scores(
    checkpoint: &MappedSafetensorsCheckpoint,
    model_path: &str,
    routing_tensor_name: &str,
    num_experts: usize,
    embedding: &[f32],
) -> Result<Vec<f32>> {
    let weights = checkpoint.extract_tensor_f32(routing_tensor_name, model_path)?;
    let info =
        checkpoint
            .tensor_info(routing_tensor_name)
            .ok_or_else(|| HybridError::MissingTensor {
                name: routing_tensor_name.to_owned(),
                path: model_path.to_owned(),
            })?;
    let shape = info.1;
    if shape.len() != 2 {
        return Err(HybridError::UnsupportedFormat(format!(
            "routing tensor '{routing_tensor_name}' must be rank-2, got {:?}",
            shape
        )));
    }
    let orientation = gate_orientation(shape[0], shape[1], embedding.len(), num_experts)
        .ok_or_else(|| {
            HybridError::UnsupportedFormat(format!(
                "unsupported safetensors routing tensor orientation {:?} for hidden_size={}",
                shape,
                embedding.len()
            ))
        })?;
    Ok(gate_scores_from_weights(
        &weights,
        orientation,
        num_experts,
        embedding,
    ))
}

pub(super) fn synthetic_gate_scores(num_experts: usize, embedding: &[f32]) -> Vec<f32> {
    let width = embedding.len().max(1);
    let chunk = (width / num_experts.max(1)).max(1);
    let mut gate_scores = Vec::with_capacity(num_experts);
    for expert_id in 0..num_experts {
        let start = (expert_id * chunk) % width;
        let end = (start + chunk).min(width);
        gate_scores.push(embedding[start..end].iter().sum());
    }
    gate_scores
}

pub(super) fn softmax(scores: &[f32]) -> Vec<f32> {
    if scores.is_empty() {
        return Vec::new();
    }

    let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exp_scores: Vec<f32> = scores
        .iter()
        .map(|&score| (score - max_score).exp())
        .collect();
    let sum_exp: f32 = exp_scores.iter().sum();
    if sum_exp <= 0.0 || !sum_exp.is_finite() {
        return vec![1.0 / scores.len() as f32; scores.len()];
    }
    exp_scores
        .into_iter()
        .map(|value| value / sum_exp)
        .collect()
}

pub(super) fn top_k_indices(weights: &[f32], top_k: usize) -> Vec<usize> {
    let mut indexed: Vec<(usize, f32)> = weights.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
    indexed
        .into_iter()
        .take(top_k)
        .map(|(idx, _)| idx)
        .collect()
}

pub(super) fn resample_embedding(input: &[f32], target_len: usize) -> Vec<f32> {
    if target_len == 0 {
        return Vec::new();
    }
    if input.len() == target_len {
        return input.to_vec();
    }
    if input.is_empty() {
        return vec![0.0; target_len];
    }
    if target_len == 1 {
        return vec![input.iter().sum::<f32>() / input.len() as f32];
    }

    let scale = (input.len() - 1) as f32 / (target_len - 1) as f32;
    let mut out = Vec::with_capacity(target_len);
    for idx in 0..target_len {
        let source = idx as f32 * scale;
        let lo = source.floor() as usize;
        let hi = source.ceil().min((input.len() - 1) as f32) as usize;
        if lo == hi {
            out.push(input[lo]);
            continue;
        }
        let t = source - lo as f32;
        out.push(input[lo] * (1.0 - t) + input[hi] * t);
    }
    out
}

pub(super) fn normalize_l2(values: &mut [f32]) {
    let norm = values.iter().map(|value| value * value).sum::<f32>().sqrt();
    if norm > 1e-8 {
        for value in values {
            *value /= norm;
        }
    }
}

pub(super) fn normalize_to_internal_embedding_dim(input: &[f32]) -> Vec<f32> {
    let mut out = resample_embedding(input, EMBEDDING_DIM);
    normalize_l2(&mut out);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gate_orientation_requires_a_hidden_dim_and_enough_experts() {
        use GateOrientation::{ExpertMajor, HiddenMajor};

        // Exact shapes, in both orientations.
        assert_eq!(
            gate_orientation(16, 8, 16, 8),
            Some(HiddenMajor { stride: 8 })
        );
        assert_eq!(
            gate_orientation(8, 16, 16, 8),
            Some(ExpertMajor { stride: 16 })
        );
        // The expert dimension may hold more experts than are requested.
        assert_eq!(
            gate_orientation(16, 64, 16, 8),
            Some(HiddenMajor { stride: 64 })
        );
        assert_eq!(
            gate_orientation(64, 16, 16, 8),
            Some(ExpertMajor { stride: 16 })
        );
        // A square tensor resolves to the first rule.
        assert_eq!(
            gate_orientation(16, 16, 16, 8),
            Some(HiddenMajor { stride: 16 })
        );
        // No hidden dimension, or too few experts.
        assert_eq!(gate_orientation(32, 32, 16, 8), None);
        assert_eq!(gate_orientation(64, 4096, 2048, 8), None);
        assert_eq!(gate_orientation(4, 16, 16, 8), None);
        assert_eq!(gate_orientation(16, 4, 16, 8), None);
    }

    #[test]
    fn gate_scores_from_weights_indexes_both_orientations() {
        let weights = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let embedding = [10.0, 100.0];

        // Expert 0 reads indices 0 and 3; expert 1 reads indices 1 and 4.
        assert_eq!(
            gate_scores_from_weights(
                &weights,
                GateOrientation::HiddenMajor { stride: 3 },
                2,
                &embedding,
            ),
            vec![410.0, 520.0]
        );
        // Expert 0 reads indices 0 and 1; expert 1 reads indices 2 and 3.
        assert_eq!(
            gate_scores_from_weights(
                &weights,
                GateOrientation::ExpertMajor { stride: 2 },
                2,
                &embedding,
            ),
            vec![210.0, 430.0]
        );
    }

    #[test]
    fn resample_embedding_preserves_target_length() {
        let input = vec![0.0, 1.0, 0.0, -1.0];
        let out = resample_embedding(&input, 8);
        assert_eq!(out.len(), 8);
        assert!((out[0] - 0.0).abs() < 1e-6);
        assert!((out[7] + 1.0).abs() < 1e-6);
    }

    #[test]
    fn normalize_to_internal_dim_returns_2048() {
        let input = vec![0.5; 3072];
        let out = normalize_to_internal_embedding_dim(&input);
        assert_eq!(out.len(), EMBEDDING_DIM);
    }

    fn assert_close(actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len(), "{actual:?} vs {expected:?}");
        for (a, e) in actual.iter().zip(expected) {
            assert!((a - e).abs() < 1e-6, "{actual:?} vs {expected:?}");
        }
    }

    #[test]
    fn softmax_sums_to_one_and_is_shift_invariant() {
        let base = softmax(&[1.0, 2.0, 3.0]);
        let shifted = softmax(&[11.0, 12.0, 13.0]);
        assert!((base.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        assert!(base[0] < base[1] && base[1] < base[2]);
        assert_close(&base, &shifted);
    }

    #[test]
    fn softmax_of_empty_is_empty() {
        assert!(softmax(&[]).is_empty());
    }

    // Characterizes corinth's current behaviour, not an endorsement of it:
    // hybrid-fusion's softmax differs here on purpose.
    #[test]
    fn softmax_uniform_fallback_on_non_finite_input() {
        assert_close(&softmax(&[f32::NAN, 1.0]), &[0.5, 0.5]);
        assert_close(&softmax(&[f32::INFINITY; 3]), &[1.0 / 3.0; 3]);
    }

    #[test]
    fn top_k_indices_returns_descending_indices_and_clamps_k() {
        let weights = [0.1, 0.7, 0.2];
        assert_eq!(top_k_indices(&weights, 2), vec![1, 2]);
        assert_eq!(top_k_indices(&weights, 5), vec![1, 2, 0]);
        assert!(top_k_indices(&weights, 0).is_empty());
        // Ties resolve to the lower index.
        assert_eq!(top_k_indices(&[0.5, 0.9, 0.5, 0.9], 4), vec![1, 3, 0, 2]);
    }

    #[test]
    fn synthetic_gate_scores_sums_disjoint_chunks() {
        assert_eq!(synthetic_gate_scores(4, &[1.0; 8]), vec![2.0; 4]);

        // Characterization, not intent: with 8 values and 3 experts the chunk
        // width is 2, so indices 6 and 7 are currently dropped.
        let embedding: Vec<f32> = (0..8).map(|value| value as f32).collect();
        assert_eq!(synthetic_gate_scores(3, &embedding), vec![1.0, 5.0, 9.0]);
    }

    #[test]
    fn normalize_l2_scales_to_unit_norm_and_leaves_near_zero_untouched() {
        let mut values = [3.0, 4.0];
        normalize_l2(&mut values);
        assert_close(&values, &[0.6, 0.8]);

        let mut near_zero = [1e-9, 0.0];
        normalize_l2(&mut near_zero);
        assert_eq!(near_zero, [1e-9, 0.0]);
    }

    #[test]
    fn resample_embedding_edge_cases() {
        assert_eq!(resample_embedding(&[], 3), vec![0.0; 3]);
        assert!(resample_embedding(&[1.0, 2.0], 0).is_empty());
        assert_eq!(resample_embedding(&[1.0, 2.0, 6.0], 1), vec![3.0]);
        assert_eq!(resample_embedding(&[1.0, 2.0], 2), vec![1.0, 2.0]);
    }
}
