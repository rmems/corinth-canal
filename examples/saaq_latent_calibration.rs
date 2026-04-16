use corinth_canal::{
    HybridConfig, HybridModel, OlmoeExecutionMode, ProjectionMode, EMBEDDING_DIM,
    gpu::GpuAccelerator,
};
use std::io::Error;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_path = std::env::var("OLMOE_PATH").unwrap_or_default();
    if model_path.trim().is_empty() {
        eprintln!("OLMOE_PATH must point to a GGUF checkpoint");
        std::process::exit(1);
    }

    let mut accelerator = GpuAccelerator::new();
    let mut model = HybridModel::new(HybridConfig {
        olmoe_model_path: model_path.clone(),
        gpu_synapse_tensor_name: "blk.0.attn_q.weight".into(),
        num_experts: 8,
        top_k_experts: 1,
        olmoe_execution_mode: OlmoeExecutionMode::SpikingSim,
        snn_steps: 1,
        projection_mode: ProjectionMode::SpikingTernary,
    })?;

    if !model.olmoe_loaded() {
        return Err(Error::other("OLMoE model did not load from OLMOE_PATH").into());
    }
    if !accelerator.is_ready() {
        return Err(Error::other("GpuAccelerator is not ready").into());
    }

    // 1. Extract embeddings for token IDs: [1045, 2099, 450, 8000, 12]
    // These represent the prompt: "Let's teach this MoE model the language of SNN"
    let tokens = [1045, 2099, 450, 8000, 12];
    let mut pooled = vec![0.0f32; EMBEDDING_DIM];

    for &token in &tokens {
        let emb = model.extract_token_embedding(token)?;
        for i in 0..EMBEDDING_DIM {
            pooled[i] += emb[i];
        }
    }

    // 2. Mean-pool into a single [f32; 2048] vector
    for i in 0..EMBEDDING_DIM {
        pooled[i] /= tokens.len() as f32;
    }

    let target_neurons = model.projector_mut().input_neurons();
    if pooled.len() != target_neurons {
        return Err(Error::other(format!(
            "Dimension mismatch: pooled len {} != target_neurons {}",
            pooled.len(),
            target_neurons
        ))
        .into());
    }

    // 3. Prepare GPU temporal state
    model.prepare_gpu_temporal(&mut accelerator)?;

    // 4. Run the 10,000-tick loop using the single pooled context vector continuously
    for tick in 0..10_000usize {
        let started = Instant::now();
        let best_walker = model.tick_gpu_temporal(&mut accelerator, &pooled)?;
        let elapsed_us = started.elapsed().as_micros();
        println!(
            "tick={} best_walker={} elapsed_us={}",
            tick + 1,
            best_walker,
            elapsed_us
        );
    }

    // Maintain strict drop order
    drop(model);
    drop(accelerator);

    Ok(())
}
