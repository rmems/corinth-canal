use crate::tensor::{Tensor, dot};

pub fn rms_norm(input: &[f32], eps: f32) -> Tensor {
    if input.is_empty() {
        return Vec::new();
    }

    let mean_sq = dot(input, input) / input.len() as f32;
    let inv = 1.0 / (mean_sq + eps).sqrt();
    input.iter().map(|v| v * inv).collect()
}
