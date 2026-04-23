//! Standalone demo for the deterministic dummy front-end.

mod support;

use corinth_canal::{
    EMBEDDING_DIM, model::Model, moe::RoutingMode, telemetry::TelemetrySnapshot,
};
use support::{RunConfig, default_spiking_model_config};

fn main() -> corinth_canal::Result<()> {
    let _ = dotenvy::from_filename(".env.local");
    let run_cfg = RunConfig::from_env();
    let routing_mode = run_cfg
        .routing_mode_override
        .unwrap_or(RoutingMode::SpikingSim);

    let mut cfg = default_spiking_model_config(run_cfg.gguf_checkpoint_path.clone(), 20);
    cfg.routing_mode = routing_mode;

    let mut model = Model::new(cfg)?;
    println!(
        "router_loaded={} routing_mode={:?}",
        model.router_loaded(),
        model.config().routing_mode
    );
    let mut total_loss = 0.0_f32;

    for step in 0..25usize {
        let phase = step as f32 / 25.0;
        let snap = TelemetrySnapshot {
            gpu_temp_c: 55.0 + phase * 30.0,
            gpu_power_w: 180.0 + phase * 120.0,
            cpu_tctl_c: 48.0 + phase * 20.0,
            cpu_package_power_w: 70.0 + phase * 40.0,
            heartbeat_signal: 0.0,
            heartbeat_enabled: false,
            timestamp_ms: (step as u64) * 1000,
        };

        let output = model.forward(&snap)?;
        let mean_embed = output.embedding.iter().sum::<f32>() / EMBEDDING_DIM as f32;
        let target = vec![mean_embed * 0.9; EMBEDDING_DIM];
        let loss = output
            .embedding
            .iter()
            .zip(target.iter())
            .map(|(hidden, expected)| (hidden - expected).powi(2))
            .sum::<f32>()
            / EMBEDDING_DIM as f32;
        total_loss += loss;

        if step % 5 == 0 {
            println!(
                "step={:>2} spikes={} top_expert={:?} loss={:.6}",
                step + 1,
                output.spike_train.iter().map(|s| s.len()).sum::<usize>(),
                output
                    .selected_experts
                    .as_ref()
                    .and_then(|v| v.first())
                    .copied(),
                loss
            );
        }
    }

    println!("avg_loss={:.6}", total_loss / 25.0);
    println!("global_step={}", model.global_step());
    println!("projector_dims={:?}", model.projector_mut().dims());

    Ok(())
}
