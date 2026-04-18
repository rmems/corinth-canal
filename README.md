# corinth-canal

SNN-logic quantization repo focused on the spiking projector and OLMoE routing path.

[![License: GPL-3.0](https://img.shields.io/badge/license-GPL--3.0-blue)](LICENSE)

## Overview

`corinth-canal` keeps the real projector and first-block OLMoE routing bridge, while the old telemetry/SNN front-end is replaced by a deterministic in-repo spike generator. That keeps the crate self-contained and runnable from a fresh clone while still supporting a real GGUF-backed path.

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
       v  OLMoE (stub/dense/spiking sim) with GGUF-backed first-block routing
expert_weights + selected_experts + hidden + spike_count telemetry
```

### GPU Acceleration (NVIDIA Blackwell sm_120+ with GIF)

When a compatible GPU is detected, the crate offloads the 2048-neuron GIF SNN to CUDA via `GpuAccelerator` (temporal tick loop with `gif_step_weighted` / `gif_step_weighted_f16`, adaptation buffer, dynamic thresholds). The temporal path now supports a GGUF-backed first-layer synapse upload: `blk.0.attn_q.weight` is memory-mapped, host-registered with CUDA, uploaded once as FP16, and then reused across forwards. The SAAQ winner selection now runs as a race-free two-pass on-device reduction: pass 1 emits one winner per block, pass 2 reduces those 8 partials to a single best walker before the 4-byte result is copied back to Rust.

```text
TelemetrySnapshot
       |
       v  project_snapshot_current (to input_current)
       |
       v  mmap GGUF tensor + cuMemHostRegister_v2 + resident synapse upload
       |
       v  gif_step_weighted_tick loop (adaptation decay, adaptive threshold = base + scale*adaptation, weighted synapses, refractory)
       |
       v  two-pass SAAQ reduction (block partials -> final best walker)
       |
       v  temporal_*_to_vec (membrane, adaptation, spikes)
       |
       v  forward_activity + Projector (GIF mode) + OLMoE routing + SAAQ/sparsity CSV (spike_count, mean_adaptation, active_fraction)
```

## GGUF First-Block Bridge

The current GGUF integration is intentionally scoped to a first-block bridge:

- `blk.0.ffn_gate_inp.weight` is preferred as the real routing matrix for `DenseSim` and `SpikingSim`
- `blk.0.ffn_gate.weight` is accepted when it exposes an explicit expert axis compatible with the checkpoint metadata
- `blk.0.attn_q.weight` remains the default recurrent synapse tensor when the checkpoint exposes a square F16 slice
- if the checkpoint does not expose a compatible recurrent synapse tensor, the temporal path falls back to the synthetic SNN weight matrix while still using the real GGUF routing bridge

This phase does not implement the full expert MLP block. Hidden outputs remain a route-weighted passthrough of the projector embedding so the repo can exercise real routing without pretending to be a full multi-layer MoE runtime.

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

To point the model at a real GGUF checkpoint while keeping the default GPU synapse tensor:

```rust
use corinth_canal::{HybridConfig, HybridModel};

let cfg = HybridConfig {
    olmoe_model_path: "/models/olmoe.gguf".into(),
    gpu_synapse_tensor_name: "blk.0.attn_q.weight".into(),
    ..Default::default()
};

let mut model = HybridModel::new(cfg)?;
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
2048 GIF neurons with adaptive thresholds (sparse 16→2048 weights)
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

## GPU Smoke Test

To validate the actual Blackwell temporal path on hardware, use the dedicated smoke test:

```bash
MOE_GGUF_PATH=/path/to/gguf cargo run --release --example gpu_smoke_test
```

This example exercises:

- GGUF mmap plus `cuMemHostRegister_v2` host registration for `blk.0.attn_q.weight`
- resident FP16 synapse upload into the GPU temporal state
- `gif_step_weighted_f16`
- the two-pass SAAQ reduction returning `best_walker`
- per-tick timing output in microseconds

This is the correct example for validating the real GPU temporal path.

## OLMoE Modes

| Mode | Behavior |
|------|----------|
| `StubUniform` | No checkpoint required. Returns uniform expert weights and zero hidden output |
| `DenseSim` | Uses the real GGUF first-block routing tensor when a model path is provided |
| `SpikingSim` | Uses the same real routing tensor but integrates expert and hidden membrane state over time |

## Run example

```bash
cargo run --example telemetry_bridge
```

With a real GGUF checkpoint:

```bash
MOE_GGUF_PATH=/models/model.gguf \
  cargo run --example telemetry_bridge --release
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

The crate provides a separate calibration path for driving the GPU temporal SNN logic with real context embeddings. This ensures the MoE gating and temporal threshold adaptation are tested against actual semantic distributions rather than uniform noise.

### Direct Token Embedding Extraction

Rather than bringing in a heavy tokenizer dependency, the `saaq_latent_calibration` example accepts model-specific token IDs and works directly from the GGUF checkpoint:

1. It probes the checkpoint metadata and selects a compatible first-block routing tensor.
2. It extracts token embeddings from `token_embd.weight`, truncating wider checkpoints down to the 2048-neuron SNN width when necessary.
3. It mean-pools the selected token rows into a single `[f32; 2048]` context vector.
4. It feeds this context vector into the direct GPU temporal loop and exports labeled research artifacts for each run.

### Calibration Runner

Run the calibration test by pointing it to your local GGUF checkpoint:

```bash
MOE_GGUF_PATH=/path/to/model.gguf \
PROMPT_SLUG=math_logic \
PROMPT_TEXT="The derivative of a constant is mathematically zero." \
PROMPT_TOKEN_IDS=402,11492,286,257,4568,318,12056,4202,13 \
RUN_OUTPUT_ROOT=/path/to/Metis-SMoE-Latent-Telemetry/routing \
cargo run --example saaq_latent_calibration --release
```

The runner writes a labeled run directory:

- `tick_telemetry.txt`
- `latent_telemetry.csv`
- `run_manifest.json`
- expected `routing_map.png` target path for the downstream Julia plot step

It also streams the per-tick performance and the winning SAAQ walker ID selected by the on-device reduction:

```text
tick=1 best_walker=12 elapsed_us=850
tick=2 best_walker=45 elapsed_us=842
...
```

This path exercises the pure Spikenaut physics engine (GIF membrane + adaptive thresholds + two-pass SAT reduction) driven by realistic, pooled semantic pressure.

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
