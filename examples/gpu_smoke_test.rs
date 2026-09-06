// SPDX-License-Identifier: Apache-2.0 OR MIT
mod support;

use corinth_canal::{gpu::GpuAccelerator, model::Model};
use std::io::Error;
use std::time::Instant;
use support::{
    RunConfig, default_spiking_model_config,
    observability::{self, CommandObserver, SafeDiagnosticData},
};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let _ = dotenvy::from_filename(".env.local");

    let _sentry_guard = observability::init_sentry("gpu_smoke_test");

    // Suppress broken pipe panics when stdout is piped to a short-lived process.
    // Registered after init_sentry so our hook wraps Sentry's panic integration
    // and broken-pipe events are not captured by Sentry.
    let default_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let is_broken_pipe = info
            .payload()
            .downcast_ref::<String>()
            .is_some_and(|s| s.contains("Broken pipe"))
            || info
                .payload()
                .downcast_ref::<&str>()
                .is_some_and(|s| s.contains("Broken pipe"));
        if is_broken_pipe {
            return;
        }
        default_hook(info);
    }));
    let observer = observability::start_command("gpu_smoke_test");
    let result = run(&observer);
    observer.finish(&result, SafeDiagnosticData::default());
    result
}

fn run(observer: &CommandObserver) -> Result<(), Box<dyn std::error::Error>> {
    let run_cfg = RunConfig::from_env();
    let smoke_ticks = gpu_smoke_ticks();
    let mut safe = SafeDiagnosticData::default();
    if let Some(model_slug) = observability::checkpoint_slug(&run_cfg.checkpoint_path) {
        safe = safe.with_model_slug(&model_slug);
        observer.annotate(safe);
    } else {
        observer.annotate(safe);
    }
    if run_cfg.checkpoint_path.trim().is_empty() {
        return Err(Error::other("CHECKPOINT_PATH must point to a GGUF checkpoint").into());
    }
    let model_path = run_cfg.checkpoint_path.clone();
    let model_label = observability::checkpoint_slug(&model_path)
        .unwrap_or_else(|| "configured_checkpoint".to_owned());

    let mut accelerator = GpuAccelerator::new();
    let mut model = Model::new(default_spiking_model_config(model_path.clone(), 1))?;

    let target_neurons = model.projector_mut().input_neurons();
    println!(
        "startup model_slug={} router_loaded={} gpu_ready={} target_neurons={}",
        model_label,
        model.router_loaded(),
        accelerator.is_ready(),
        target_neurons,
    );

    if !model.router_loaded() {
        return Err(Error::other("Router model did not load from CHECKPOINT_PATH").into());
    }
    if !accelerator.is_ready() {
        return Err(Error::other("GpuAccelerator is not ready").into());
    }

    model.prepare_gpu_temporal(&mut accelerator)?;
    println!("prepared gguf-backed temporal path; beginning {smoke_ticks} direct GPU ticks");

    for tick in 0..smoke_ticks {
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
        if should_validate_tick(tick, smoke_ticks) {
            validate_gpu_tick_state(&accelerator, target_neurons, best_walker, tick + 1)?;
        }
    }

    println!("completed {smoke_ticks} GPU ticks; dropping model before accelerator");
    drop(model);
    drop(accelerator);
    println!("gpu smoke test finished cleanly");

    Ok(())
}

fn gpu_smoke_ticks() -> usize {
    std::env::var("GPU_SMOKE_TICKS")
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .filter(|ticks| *ticks > 0)
        .unwrap_or(10_000)
}

fn should_validate_tick(tick: usize, smoke_ticks: usize) -> bool {
    tick == 0 || tick + 1 == smoke_ticks || (tick + 1).is_multiple_of(1_000)
}

fn validate_gpu_tick_state(
    accelerator: &GpuAccelerator,
    target_neurons: usize,
    best_walker: u32,
    tick: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    if best_walker as usize >= target_neurons {
        return Err(Error::other(format!(
            "tick {tick}: best_walker {best_walker} outside neuron range 0..{target_neurons}"
        ))
        .into());
    }

    let spikes = accelerator.temporal_spikes_to_vec(target_neurons)?;
    if spikes.len() != target_neurons {
        return Err(Error::other(format!(
            "tick {tick}: spike output length mismatch: expected {target_neurons}, got {}",
            spikes.len()
        ))
        .into());
    }
    if let Some((idx, value)) = spikes.iter().enumerate().find(|(_, value)| **value > 1) {
        return Err(Error::other(format!(
            "tick {tick}: spike output at neuron {idx} is {value}, expected binary 0/1"
        ))
        .into());
    }

    let membrane = accelerator.temporal_membrane_to_vec(target_neurons)?;
    if membrane.len() != target_neurons {
        return Err(Error::other(format!(
            "tick {tick}: membrane length mismatch: expected {target_neurons}, got {}",
            membrane.len()
        ))
        .into());
    }
    if let Some((idx, value)) = membrane
        .iter()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(Error::other(format!(
            "tick {tick}: membrane at neuron {idx} is non-finite ({value})"
        ))
        .into());
    }

    Ok(())
}
