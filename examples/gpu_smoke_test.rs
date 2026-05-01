mod support;

use corinth_canal::{gpu::GpuAccelerator, model::Model};
use std::io::Error;
use std::time::Instant;
use support::{config::RunConfig, default_spiking_model_config};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let _ = dotenvy::from_filename(".env.local");
    let run_cfg = RunConfig::from_env();
    if run_cfg.gguf_checkpoint_path.trim().is_empty() {
        return Err(Error::other("GGUF_CHECKPOINT_PATH must point to a GGUF checkpoint").into());
    }
    let model_path = run_cfg.gguf_checkpoint_path.clone();

    let mut accelerator = GpuAccelerator::new();
    let mut model = Model::new(default_spiking_model_config(model_path.clone(), 1))?;

    let target_neurons = model.projector_mut().input_neurons();
    println!(
        "startup model_path={} router_loaded={} gpu_ready={} target_neurons={}",
        model_path,
        model.router_loaded(),
        accelerator.is_ready(),
        target_neurons,
    );

    if !model.router_loaded() {
        return Err(
            Error::other("OlmoeRouter model did not load from GGUF_CHECKPOINT_PATH").into(),
        );
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
