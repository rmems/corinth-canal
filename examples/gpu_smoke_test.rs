use corinth_canal::{
    HybridConfig, HybridModel, OlmoeExecutionMode, ProjectionMode, gpu::GpuAccelerator,
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

    let target_neurons = model.projector_mut().input_neurons();
    println!(
        "startup model_path={} olmoe_loaded={} gpu_ready={} target_neurons={}",
        model_path,
        model.olmoe_loaded(),
        accelerator.is_ready(),
        target_neurons,
    );

    if !model.olmoe_loaded() {
        return Err(Error::other("OLMoE model did not load from OLMOE_PATH").into());
    }
    if !accelerator.is_ready() {
        return Err(Error::other("GpuAccelerator is not ready").into());
    }

    model.prepare_gpu_temporal(&mut accelerator)?;
    println!("prepared gguf-backed temporal path; beginning 10,000 direct GPU ticks");

    for tick in 0..10_000usize {
        let phase = tick as f32 * 0.31;
        let input_spikes: Vec<f32> = (0..target_neurons)
            .map(|i| {
                let wave = (i as f32 * 0.017 + phase).sin();
                0.1 * (wave + 1.0) * 0.5
            })
            .collect();

        let started = Instant::now();
        let best_walker = model.tick_gpu_temporal(&mut accelerator, &input_spikes)?;
        let elapsed_us = started.elapsed().as_micros();
        println!(
            "tick={} best_walker={} elapsed_us={}",
            tick + 1,
            best_walker,
            elapsed_us
        );
    }

    println!("completed 10,000 GPU ticks; dropping model before accelerator");
    drop(model);
    drop(accelerator);
    println!("gpu smoke test finished cleanly");

    Ok(())
}
