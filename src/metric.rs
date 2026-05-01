#[allow(dead_code)]
pub(crate) fn mean_squared_error(output: &[f32], target: &[f32]) -> f32 {
    let len = output.len().min(target.len());

    if len == 0 {
        return 0.0;
    }
    output
        .iter()
        .zip(target.iter())
        .take(len)
        .map(|(actual, expected)| (actual - expected).powi(2))
        .sum::<f32>()
        / len as f32
}
