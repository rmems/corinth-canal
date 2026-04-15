# corinth-canal

Standalone SNN-logic quantization crate focused on the spiking projector and OLMoE routing path.

[![License: GPL-3.0](https://img.shields.io/badge/license-GPL--3.0-blue)](LICENSE)

## Overview

`corinth-canal` keeps the real projector and OLMoE simulation logic, while the old telemetry/SNN front-end is replaced by a deterministic in-repo spike generator. That keeps the crate self-contained and runnable from a fresh clone.

## Origin

This repository originated from the `spikenaut-hybrid` codebase and was reorganized into `corinth-canal` with a consolidated `src/hybrid`, `src/tensor`, and `src/transformer` structure.

## Architecture

### CPU Fallback + GPU GIF Path

```text
TelemetrySnapshot
       |
       v  TelemetryEncoder (delta modulation) / project_snapshot_current (GPU)
[i8; 4] ternary spikes (+1/0/-1)
       |
       v  SignedSplitBankBridge or GPU input_spikes
2048 input neurons
       |
       v  SparseGifHiddenLayer (CPU) or gif_step_weighted (GPU with adaptation)
2048 GIF hidden neurons with adaptive thresholds + history-aware SAAQ
       |
       v  Projector (SpikingTernary GIF mode for 2048-neuron input)
dense embedding [2048]
       |
       v  OLMoE (stub/dense/spiking sim) + SAT solver for routing
expert_weights + selected_experts + hidden + spike_count telemetry
```

### GPU Acceleration (NVIDIA Blackwell sm_120+ with GIF)

When a compatible GPU is detected, the crate offloads the 2048-neuron GIF SNN to CUDA via `GpuAccelerator` (temporal tick loop with `gif_step_weighted`, adaptation buffer, dynamic thresholds). This provides history-aware spiking for better SAAQ quantization and sparsity telemetry to optimize 16GB VRAM/power usage.

```text
TelemetrySnapshot
       |
       v  project_snapshot_current (to input_current)
       |
       v  GpuAccelerator::reset_temporal_state() + load_synapse_weights()
       |
       v  gif_step_weighted_tick loop (adaptation decay, adaptive threshold = base + scale*adaptation, weighted synapses, refractory)
       |
       v  temporal_*_to_vec (membrane, adaptation, spikes)
       |
       v  forward_activity + Projector (GIF mode) + OLMoE + SAAQ/sparsity CSV (spike_count, mean_adaptation, active_fraction)
```

## Quick start

```rust
use corinth_canal::{
    HybridConfig, HybridModel, TelemetryFunnel, TelemetrySnapshot,
    FUNNEL_HIDDEN_NEURONS,
};

// Configure the telemetry funnel
let thresholds = [1.0, 5.0, 1.0, 5.0];
let snn_steps = 20;
let mut funnel = TelemetryFunnel::new(thresholds, snn_steps);

// Create a telemetry snapshot
let snap = TelemetrySnapshot {
    gpu_temp_c: 60.0,
    gpu_power_w: 260.0,
    cpu_tctl_c: 77.0,
    cpu_package_power_w: 153.0,
    timestamp_ms: 0,
};

// Run the full funnel: encoder -> bridge -> 512-neuron GIF hidden layer
let activity = funnel.encode_snapshot(&snap);

// Or use the complete hybrid pipeline
let cfg = HybridConfig::default();
let mut model = HybridModel::new_with_projector_neurons(cfg, FUNNEL_HIDDEN_NEURONS)?;
let output = model.forward_activity(
    &activity.spike_train,
    &activity.potentials,
    &activity.iz_potentials,
)?;

println!("Selected experts: {:?}", output.selected_experts);
```

## TelemetryEncoder

The `TelemetryEncoder` converts continuous telemetry values into ternary spike vectors using delta modulation. This bridges raw telemetry data into the spiking projection path.

### Delta modulation

For each telemetry channel (GPU temp, GPU power, CPU temp, CPU power), the encoder:

- Stores a baseline value (seeded from the first sample)
- Compares each new value to its baseline
- Emits `+1` if the change exceeds the positive threshold
- Emits `-1` if the change exceeds the negative threshold
- Emits `0` otherwise
- Updates the baseline only when a spike fires

### Usage

```rust
use corinth_canal::{TelemetryEncoder, TelemetrySnapshot};

// Configure thresholds: [temp, power, temp, power]
let thresholds = [1.0, 5.0, 1.0, 5.0];
let mut encoder = TelemetryEncoder::new(thresholds);

let snap = TelemetrySnapshot {
    gpu_temp_c: 60.0,
    gpu_power_w: 260.0,
    cpu_tctl_c: 77.0,
    cpu_package_power_w: 153.0,
    timestamp_ms: 0,
};

// First call seeds the baseline, returns [0, 0, 0, 0]
let spikes = encoder.encode(&snap);

// Subsequent calls emit spikes based on delta
let spikes = encoder.encode(&snap);
```

## TelemetryFunnel

The `TelemetryFunnel` orchestrates the full spiking pipeline from raw telemetry to hidden-layer activity ready for projection.

### Architecture

```
TelemetrySnapshot
       |
       v  TelemetryEncoder (delta modulation)
[i8; 4] ternary spikes
       |
       v  SignedSplitBankBridge
16 input neurons (4 channels × 2 polarities × 2 neurons)
       |
       v  SparseGifHiddenLayer
512 GIF neurons with adaptive thresholds (sparse 16→512 weights)
       |
       v  FunnelActivity
spike_train + potentials + iz_potentials
```

### Components

| Component | Description |
|-----------|-------------|
| `SignedSplitBankBridge` | Expands 4-channel ternary spikes into 16 input neurons using signed split banks (positive/negative pairs per channel) |
| `SparseGifHiddenLayer` | Pure-Rust Generalized Integrate-and-Fire (GIF) layer with adaptive thresholds. Uses sparse 4-edge connectivity per hidden neuron for fast inference |
| `FunnelActivity` / `HybridOutput` | Output struct containing the 2048-neuron GIF spike train, membrane potentials, adaptation stats, and Izhikevich potentials. Extended with spike_count telemetry for SAAQ. |

### GIF neuron parameters (synced in GPU kernel, funnel.rs, projector.rs)

| Parameter | Value | Description |
|-----------|-------|-------------|
| `leak` | 0.92 | Membrane leak factor (GIF_LEAK) |
| `threshold_base` | 0.65 | Base firing threshold (GIF_THRESHOLD_BASE) |
| `adaptation_scale` | 0.22 | Adaptive threshold scaling (GIF_ADAPTATION_SCALE) |
| `adaptation_decay` | 0.94 | Adaptation variable decay (GIF_ADAPTATION_DECAY) |
| `reset_ratio` | 0.35 | Post-spike membrane reset (GIF_RESET_RATIO) |
| `drive_scale` | 0.75 | Input drive scaling |

### Usage with GIF GPU Path

```rust
use corinth_canal::{HybridModel, HybridConfig, GpuAccelerator, TelemetrySnapshot};

let mut model = HybridModel::new(HybridConfig::default()).unwrap();
let mut accelerator = GpuAccelerator::new();  // falls back gracefully
let output = model.forward_gpu_temporal(&mut accelerator, &TelemetrySnapshot::default()).unwrap();

// GIF adaptation drives sparsity pruning; CSV now logs spike_count, mean_adaptation, active_fraction for VRAM optimization and Julia symbolic regression
```

## Features

| Feature | Effect |
|---------|--------|
| `gguf` | Enables GGUF header validation for model-file probing |

## Run example

```bash
cargo run --example telemetry_bridge
```

With GGUF probing enabled:

```bash
OLMOE_PATH=/models/OLMoE-1B-7B-Q5_K_M.gguf \
  cargo run --example telemetry_bridge --features gguf --release
```

## CSV Replay Contract

`corinth-canal` consumes canonical CSV exported by `gaming-telemetry`:

`timestamp_ms,gpu_temp_c,gpu_power_w,cpu_tctl_c,cpu_package_power_w`

Producer/consumer flow:

`collector -> gpu_telemetry_v1_batch_N.parquet -> export_csv -> csv_replay`

Replay command:

```bash
cargo run --example csv_replay /path/to/canonical.csv
```

`csv_replay` validates the header strictly, counts malformed rows, and prints:
- `rows_processed`
- `rows_skipped`
- `global_step`
- `olmoe_loaded`

## Latent Telemetry Calibration

The crate provides a separate calibration-only path for harvesting latent SNN metrics from canonical hardware telemetry. This maintains strict separation of concerns: the standard hardware telemetry file remains a pure, lightweight record of the physical machine, while latent metrics are generated on-demand during calibration passes for symbolic regression.

### Latent Snapshot

`SnnLatentSnapshot` captures internal SNN state and teacher targets:

| Field | Description |
|-------|-------------|
| `timestamp_ms` | Wall-clock timestamp from hardware telemetry |
| `avg_pop_firing_rate_hz` | Population-average firing rate of hidden neurons |
| `membrane_dv_dt` | Row-to-row change in mean membrane potential |
| `routing_entropy` | Normalized entropy of OLMoE expert weights |
| `saaq_delta_q_prev` | Previous-step quantization delta (teacher history) |
| `saaq_delta_q_target` | Target quantization delta (teacher signal) |

### Latent Calibrator

`SnnLatentCalibrator` derives latent features from runtime state:

- **`avg_pop_firing_rate_hz`**: Computed from hidden-layer spike count over elapsed wall-clock time
- **`membrane_dv_dt`**: Computed from row-to-row mean hidden membrane change
- **`routing_entropy`**: Derived from normalized entropy of `HybridOutput.expert_weights`
- **`saaq_delta_q_prev` / `saaq_delta_q_target`**: Deterministic first-pass calibration policy for bootstrapping symbolic regression

### Latent CSV Exporter

`SnnLatentCsvExporter` writes latent snapshots to CSV with the header:

```text
timestamp_ms,avg_pop_firing_rate_hz,membrane_dv_dt,routing_entropy,saaq_delta_q_prev,saaq_delta_q_target
```

This format is compatible with Julia symbolic regression workflows (e.g., `SymbolicRegression.jl`).

### Calibration Runner

The `saaq_latent_calibration` example replays canonical hardware CSV through the funnel/model, computes latent metrics, and writes `snn_latent_telemetry.csv`:

```bash
cargo run --example saaq_latent_calibration /path/to/canonical.csv [output.csv]
```

Default output path: `snn_latent_telemetry.csv`

The runner validates the canonical hardware CSV header strictly and prints:
- `rows_processed`
- `rows_skipped`
- `global_step`
- `olmoe_loaded`
- Sample latent metrics for first 5 rows and every 100th row

### Separation of Concerns

The latent telemetry path:
- Does not modify `TelemetrySnapshot` or the canonical hardware telemetry schema
- Does not affect normal inference or replay performance
- Runs only during calibration passes when generating symbolic regression datasets
- Keeps the standard `telemetry.csv` contract lightweight and pure

## GPU Routing Telemetry

For high-performance expert routing analysis, the crate utilizes the NVIDIA Blackwell (RTX 5080) GPU to perform two-pass SAT reductions directly in VRAM. This telemetry path captures the final winning scores and walkers for every token processed.

### GPU Routing Snapshot

| Field | Description |
|-------|-------------|
| `token_idx` | Sequential index of the processed token |
| `best_score` | Final optimized SAT score from the second-pass reduction |
| `best_walker` | ID of the winning walker that achieved the best score |

### Mechanism: Two-Pass Reduction

Unlike the latent calibration path which runs on the CPU, the routing telemetry is powered by the `GpuAccelerator`:

1.  **SAT Extraction**: `satsolver_extract` prepares the initial state in VRAM.
2.  **Best-Reduction**: `satsolver_aux_reduce_best` performs a massively parallel reduction across the GPU grid to find the global optimum.
3.  **8-Byte Transfer**: Only the final `best_score` and `best_walker` (8 bytes total) are transferred back to the CPU for logging.

### Output

Rows are appended to `snn_gpu_routing_telemetry.csv` with the header:

```text
token_idx,best_score,best_walker
```

This file is created automatically on the first call to `compute_routing_telemetry` and grows sequentially as more tokens are processed.

## Project layout

| Path | Responsibility |
|------|----------------|
| `Cargo.toml` | crate config and dependencies |
| `src/lib.rs` | public API and crate docs |
| `src/types.rs` | `TelemetrySnapshot`, config, enums, output types |
| `src/error.rs` | `HybridError`, `Result` |
| `src/telemetry.rs` | `TelemetryEncoder` delta-modulation state machine |
| `src/tensor/mod.rs` | candle-free tensor utilities |
| `src/transformer/mod.rs` | transformer helpers |
| `src/hybrid/mod.rs` | hybrid module switchboard |
| `src/hybrid/projector.rs` | 2-bit spiking projector logic |
| `src/hybrid/olmoe.rs` | GGUF-aware OLMoE simulation |
| `src/funnel.rs` | TelemetryFunnel: encoder + split-bank bridge + sparse GIF hidden layer |
| `src/hybrid/hybrid.rs` | deterministic front-end + projector + OLMoE orchestration |
| `src/latent.rs` | `SnnLatentSnapshot`, `SnnLatentCalibrator`, `SnnLatentCsvExporter` for symbolic regression |
| `src/gpu/mod.rs` | GPU accelerator public API and module switchboard |
| `src/gpu/kernels/` | CUDA kernels (`.cu`, `.cuh`) for GPU-accelerated tasks |
| `src/gpu/wrappers/` | Safe Rust wrappers for CUDA context, memory, and kernel launches |
| `build.rs` | Build script to compile CUDA kernels to PTX |
| `examples/telemetry_bridge.rs` | end-to-end standalone example |
| `examples/csv_replay.rs` | canonical CSV replay adapter (funnel-driven) |
| `examples/saaq_latent_calibration.rs` | calibration-only latent telemetry exporter |

## Acknowledgments

This work builds directly on prior research in spiking and spike-driven LLMs.

- Xing et al. (2024), *SpikeLLM: Scaling up Spiking Neural Network to Large Language Models via Saliency-based Spiking*.
  - arXiv: [2407.04752](https://arxiv.org/abs/2407.04752)
  - DOI: [10.48550/arXiv.2407.04752](https://doi.org/10.48550/arXiv.2407.04752)
- Xu et al. (2025), *Neuromorphic spike-based large language model* (`NSLLM`).
  - Journal page: [National Science Review](https://academic.oup.com/nsr/advance-article/doi/10.1093/nsr/nwaf551/8365570)
  - DOI: [10.1093/nsr/nwaf551](https://doi.org/10.1093/nsr/nwaf551)
- Qiu et al. (2025), *Quantized Spike-driven Transformer* (`QSD-Transformer`).
  - ICLR poster: [OpenReview / ICLR 2025](https://iclr.cc/virtual/2025/poster/30954)
  - arXiv: [2501.13492](https://arxiv.org/abs/2501.13492)
- The Allen Institute for AI for releasing the open OLMoE models used as the primary testbed.

## Citation

If you use `corinth-canal` or the SNN-logic quantization approach in your research, please cite:

```bibtex
@misc{corinth-canal2026,
  title        = {corinth-canal: Turning MOE Architecture into SNN Quantization},
  author       = {Raul Montoya Cardenas},
  year         = {2026},
  howpublished = {\url{https://github.com/Spikenaut/corinth-canal}},
  note         = {SNN-logic quantization with GIF-Ternary spiking for MoE models}
}
```

## License

GPL-3.0-or-later. See [LICENSE](LICENSE).
