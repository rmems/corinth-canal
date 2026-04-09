//! Standalone demo for the deterministic dummy front-end.

use corinth_canal::{
    EMBEDDING_DIM, HybridConfig, HybridModel, OlmoeExecutionMode, ProjectionMode,
    TelemetrySnapshot,
};

fn main() -> corinth_canal::Result<()> {
    let model_path = std::env::var("OLMOE_PATH").unwrap_or_default();
    let olmoe_execution_mode = match std::env::var("OLMOE_MODE")
        .unwrap_or_else(|_| "spiking".into())
        .to_ascii_lowercase()
        .as_str()
    {
        "dense" => OlmoeExecutionMode::DenseSim,
        "stub" => OlmoeExecutionMode::StubUniform,
        _ => OlmoeExecutionMode::SpikingSim,
    };

    let cfg = HybridConfig {
        olmoe_model_path: model_path,
        snn_steps: 20,
        context_length: 512,
        num_experts: 8,
        top_k_experts: 1,
        olmoe_execution_mode,
        projection_mode: ProjectionMode::SpikingTernary,
    };

    let mut model = HybridModel::new(cfg)?;
    let mut total_loss = 0.0_f32;

    for step in 0..25usize {
        let phase = step as f32 / 25.0;
        let snap = TelemetrySnapshot {
            timestamp_ms: (step as u64) * 1000,
            gpu_temp_c: 55.0 + phase * 30.0,
            gpu_power_w: 180.0 + phase * 120.0,
            gpu_clock_mhz: 1800.0 + phase * 300.0,
            mem_util_pct: 50.0 + phase * 30.0,
            cpu_tctl_c: 48.0 + phase * 20.0,
            cpu_package_power_w: 70.0 + phase * 40.0,
            workload_throughput: (0.005 + phase as f64 * 0.01),
            workload_efficiency: (0.6 + phase as f64 * 0.2),
            auxiliary_signal: (phase as f64).sin().abs(),
        };

        let output = model.forward(&snap)?;
        let mean_embed = output.embedding.iter().sum::<f32>() / EMBEDDING_DIM as f32;
        let target = vec![mean_embed * 0.9; EMBEDDING_DIM];
        let loss = model.train_step(&snap, &target)?;
        total_loss += loss;

        if step % 5 == 0 {
            println!(
                "step={:>2} spikes={} top_expert={:?} loss={:.6}",
                step + 1,
                output.spike_train.iter().map(|s| s.len()).sum::<usize>(),
                output.selected_experts.as_ref().and_then(|v| v.first()).copied(),
                loss
            );
        }
    }

    println!("avg_loss={:.6}", total_loss / 25.0);
    println!("global_step={}", model.global_step());
    println!("projector_dims={:?}", model.projector_mut().dims());

    Ok(())
}
