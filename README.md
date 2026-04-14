# corinth-canal

Standalone SNN-logic quantization crate focused on the spiking projector and OLMoE routing path.

[![License: GPL-3.0](https://img.shields.io/badge/license-GPL--3.0-blue)](LICENSE)

## Overview

`corinth-canal` keeps the real projector and OLMoE simulation logic, while the old telemetry/SNN front-end is replaced by a deterministic in-repo spike generator. That keeps the crate self-contained and runnable from a fresh clone.

## Origin

This repository originated from the `spikenaut-hybrid` codebase and was reorganized into `corinth-canal` with a consolidated `src/hybrid`, `src/tensor`, and `src/transformer` structure.

## Architecture

```text
TelemetrySnapshot
       |
       v  TelemetryEncoder (delta modulation)
[i8; 4] ternary spikes (+1/0/-1)
       |
       v  SignedSplitBankBridge
16 input neurons (4 ch × 2 signs × 2 bank width)
       |
       v  SparseGifHiddenLayer (sparse 16 → 512)
512 GIF hidden neurons with adaptive thresholds
       |
       v  Projector (generalized for 512-neuron input)
dense embedding [2048]
       |
       v  OLMoE (stub/dense/spiking sim)
expert_weights + selected_experts + hidden
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
| `FunnelActivity` | Output struct containing the 512-neuron spike train, membrane potentials, and Izhikevich potentials |

### GIF neuron parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `leak` | 0.92 | Membrane leak factor |
| `threshold_base` | 0.65 | Base firing threshold |
| `adaptation_scale` | 0.22 | Adaptive threshold scaling |
| `adaptation_decay` | 0.94 | Adaptation variable decay |
| `reset_ratio` | 0.35 | Post-spike membrane reset |

### Usage

```rust
use corinth_canal::{TelemetryFunnel, TelemetrySnapshot, FUNNEL_HIDDEN_NEURONS};

let mut funnel = TelemetryFunnel::new(
    [1.0, 5.0, 1.0, 5.0], // thresholds
    20,                   // snn_steps
);

let activity = funnel.encode_snapshot(&TelemetrySnapshot::default());

// activity.spike_train: 512-neuron spike activity over 20 steps
// activity.potentials: final membrane potentials (512 values)
// activity.iz_potentials: Izhikevich potentials (5 values)
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
| `examples/telemetry_bridge.rs` | end-to-end standalone example |
| `examples/csv_replay.rs` | canonical CSV replay adapter (funnel-driven) |

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
