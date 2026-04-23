# corinth-canal

SNN-logic quantization repo focused on the spiking projector and OlmoeRouter routing path.

[![License: GPL-3.0](https://img.shields.io/badge/license-GPL--3.0-blue)](LICENSE)

## Overview

`corinth-canal` keeps the real projector and first-block OlmoeRouter routing bridge, while the old telemetry/SNN front-end is replaced by a deterministic in-repo spike generator. That keeps the crate self-contained and runnable from a fresh clone while still supporting a real GGUF-backed path.

## Origin

This repository originated from the `spikenaut-hybrid` codebase and is now organized around feature modules: `src/model`, `src/moe`, `src/projector.rs`, `src/funnel.rs`, `src/telemetry.rs`, and `src/latent.rs`.

Going forward, `corinth-canal` is the **single-crate reference implementation** for the `rmems` modular line. Proven components will graduate into separate `rmems-*` crates per `docs/PROMOTION_RULES.md`; `corinth-canal` itself stays as one crate.

## Documentation

- `docs/ARCHITECTURE.md` — block diagram, module map, and hidden control flow (CWD writes, env-resolved paths, fallback stamping).
- `docs/RUN_PROFILES.md` — catalog of validated `(prompt, telemetry, heartbeat, SAAQ rule)` tuples with the exact `just` command for each.
- `docs/PROMOTION_RULES.md` — criteria for graduating a module out of this reference repo.
- `docs/MODULE_STATUS.md` — live status table; machine-readable mirror in `manifests/proven_components.toml`.
- `manifests/known_good_runs.md` — append-only log of blessed run IDs.

## Module Map

- `src/model/*`: end-to-end runtime orchestration, GPU temporal logic, and routing telemetry helpers
- `src/moe/*`: GGUF checkpoint parsing, router math, and token embedding extraction
- `src/projector.rs`: spike-to-embedding projection and `ProjectionMode`
- `src/funnel.rs`: telemetry funnel and GIF hidden layer
- `src/telemetry.rs`: `TelemetryEncoder` plus `TelemetrySnapshot` re-export
- `src/latent.rs`: latent calibration snapshots and CSV export
- `examples/support/mod.rs`: shared example bootstrap helpers

## Migration Note

Core runtime imports now come from feature modules instead of flat crate-root exports:

```rust
use corinth_canal::model::{Model, ModelConfig};
use corinth_canal::moe::RoutingMode;
use corinth_canal::projector::ProjectionMode;
use corinth_canal::telemetry::TelemetrySnapshot;
```

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
       v  OlmoeRouter (stub/dense/spiking sim) with GGUF-backed first-block routing
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
       v  forward_activity + Projector (GIF mode) + OlmoeRouter routing + SAAQ/sparsity CSV (spike_count, mean_adaptation, active_fraction)
```

## GGUF First-Block Bridge

The current OlmoeRouter integration is intentionally scoped to a first-block bridge:

- `blk.0.ffn_gate_inp.weight` is used as the real routing matrix for `DenseSim` and `SpikingSim`
- `blk.0.attn_q.weight` is the default recurrent synapse matrix for `forward_gpu_temporal`
- the GGUF file is memory-mapped and the GPU synapse slice is registered with CUDA via `cuMemHostRegister_v2`
- the temporal GPU path reuses resident device weights across forwards instead of rebuilding dummy weights

This phase does not implement the full expert MLP block. Hidden outputs remain a route-weighted passthrough of the projector embedding so the repo can exercise real routing without pretending to be a full multi-layer OlmoeRouter runtime.

## Quick start

```rust
use corinth_canal::{
    FUNNEL_HIDDEN_NEURONS,
    funnel::TelemetryFunnel,
    model::{Model, ModelConfig},
    telemetry::TelemetrySnapshot,
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
let cfg = ModelConfig::default();
let mut model = Model::new_with_projector_neurons(cfg, FUNNEL_HIDDEN_NEURONS)?;
let output = model.forward_activity(
    &activity.spike_train,
    &activity.potentials,
    &activity.iz_potentials,
)?;

println!("Selected experts: {:?}", output.selected_experts);
```

To point the model at a real GGUF checkpoint while keeping the default GPU synapse tensor:

```rust
use corinth_canal::model::{Model, ModelConfig};

let cfg = ModelConfig {
    gguf_checkpoint_path: "/models/olmoe.gguf".into(),
    gpu_synapse_tensor_name: "blk.0.attn_q.weight".into(),
    ..Default::default()
};

let mut model = Model::new(cfg)?;
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
use corinth_canal::{telemetry::TelemetryEncoder, telemetry::TelemetrySnapshot};

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
| `FunnelActivity` / `ModelOutput` | Output struct containing the 2048-neuron GIF spike train, membrane potentials, adaptation stats, and Izhikevich potentials. Extended with spike_count telemetry for SAAQ. |

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
use corinth_canal::{
    gpu::GpuAccelerator,
    model::{Model, ModelConfig},
    telemetry::TelemetrySnapshot,
};

let mut model = Model::new(ModelConfig::default()).unwrap();
let mut accelerator = GpuAccelerator::new();  // falls back gracefully
let output = model.forward_gpu_temporal(&mut accelerator, &TelemetrySnapshot::default()).unwrap();

// GIF adaptation drives sparsity pruning; CSV now logs spike_count, mean_adaptation, active_fraction for VRAM optimization and Julia symbolic regression
```

## GPU Smoke Test

To validate the actual Blackwell temporal path on hardware, use the dedicated smoke test:

```bash
GGUF_CHECKPOINT_PATH=/path/to/gguf cargo run --release --example gpu_smoke_test
```

This example exercises:

- GGUF mmap plus `cuMemHostRegister_v2` host registration for `blk.0.attn_q.weight`
- resident FP16 synapse upload into the GPU temporal state
- `gif_step_weighted_f16`
- the two-pass SAAQ reduction returning `best_walker`
- per-tick timing output in microseconds

This is the correct example for validating the real GPU temporal path. `saaq_latent_calibration` is CPU-side calibration logic and does not exercise the direct GPU temporal loop.

## OlmoeRouter Modes

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
GGUF_CHECKPOINT_PATH=/models/olmoe.gguf \
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
- `router_loaded`

## Latent Telemetry Calibration

The crate provides a separate calibration path for driving the GPU temporal SNN logic with real context embeddings. This ensures the MoE gating and temporal threshold adaptation are tested against actual semantic distributions rather than uniform noise.

### Direct Token Embedding Extraction

Rather than bringing in a heavy text tokenizer dependency, the `saaq_latent_calibration` example bypasses text parsing entirely:

1. It memory-maps the `token_embd.weight` tensor from the provided OlmoeRouter GGUF checkpoint.
2. It extracts the raw 2048-dimensional embedding rows for a hard-coded sequence of token IDs (representing the prompt: *"Let's teach this MoE model the language of SNN"*).
3. It mean-pools these tokens into a single `[f32; 2048]` context vector.
4. It feeds this single context vector continuously into the direct GPU temporal loop (`tick_gpu_temporal`) for 10,000 ticks.

### Calibration Runner

Run the calibration test by pointing it to your local OlmoeRouter checkpoint:

```bash
GGUF_CHECKPOINT_PATH=/path/to/olmoe.gguf cargo run --example saaq_latent_calibration --release
```

The runner will output the per-tick performance and the winning SAAQ walker ID selected by the on-device reduction:

```text
tick=1 best_walker=12 elapsed_us=850
tick=2 best_walker=45 elapsed_us=842
...
```

This path exercises the pure SNN physics engine (GIF membrane + adaptive thresholds + two-pass SAT reduction) driven by realistic, pooled semantic pressure.

### Validation Sweeps (dual-SAAQ, CSV-replay, repeat-aware)

The `saaq_latent_calibration` runner is a sweep orchestrator over every discovered GGUF model. Per-run artifacts land under a fully-labeled directory tree:

```text
<VALIDATION_OUTPUT_ROOT>/<model_slug>/<telemetry_source>/<heartbeat_slug>/<run_id>/
    tick_telemetry.txt
    latent_telemetry.csv
    run_manifest.json
```

where `<run_id>` = `<YYYYMMDDTHHMMSS>_<prompt_slug>_r<repeat_idx>` (UTC, sortable).

#### Environment knobs

| Env | Default | Purpose |
|-----|---------|---------|
| `TELEMETRY_SOURCE` | `synthetic` | `synthetic` runs the in-process sinusoid; `csv` replays a canonical telemetry CSV. Must be set explicitly to `csv` for real-corpus runs — defaults are dependency-light on purpose. |
| `TELEMETRY_CSV_PATH` | _(unset)_ | Canonical-format CSV (`timestamp_ms,gpu_temp_c,gpu_power_w,cpu_tctl_c,cpu_package_power_w`). Required when `TELEMETRY_SOURCE=csv`. Missing/malformed → fallback to synthetic, stamped `synthetic_fallback` in the manifest. |
| `TICKS` | `512` | Per-run tick count. Special case: when `TICKS=0` *and* `TELEMETRY_SOURCE=csv`, uses the exact CSV row count so a SymbolicRegression.jl corpus run covers exactly one loop with zero wraparound contamination. |
| `REPEAT_COUNT` | `1` | Number of repeats per (model, telemetry, heartbeat) tuple. Deterministic inputs mean repeats should bit-match — divergence indicates scheduler noise. |
| `VALIDATION_OUTPUT_ROOT` | `./artifacts` | Top of the per-run output tree. Repo-relative by default; set to an absolute path to redirect runs into an external consumer. |
| `HEARTBEAT_MATRIX` | `off,on` | Kept as-is for this round; amplitude/period sweep is future work. |
| `SAAQ_RULE` | `saaq_v1_5` | Selects which rule fills the legacy `saaq_delta_q_{prev,target}` columns. Both rules are always emitted in the dual-SAAQ columns regardless. |

#### Wraparound hygiene

For `TELEMETRY_SOURCE=csv`, if `TICKS > rows.len()` the runner emits a warning:

```text
wraparound: ticks={N} > rows={M}; regression corpus may be contaminated by looped endings. Prefer TICKS=0 or TICKS<={M}.
```

The manifest records `wraparound_enabled`, `wraparound_loops`, and `ticks_effective` so any downstream filter can reject contaminated traces with one predicate. **For SymbolicRegression.jl corpus generation, always use `TICKS=0` (or `TICKS ≤ CSV row count`).** Wraparound is intended for smoke tests only.

#### Dual-SAAQ CSV schema

`latent_telemetry.csv` preserves the original 12 columns for backward compatibility with existing SR.jl scripts, and appends 4 new columns so both SAAQ rules can be fitted without rerunning:

```text
timestamp_ms, avg_pop_firing_rate_hz, membrane_dv_dt, routing_entropy,
saaq_delta_q_prev, saaq_delta_q_target,           # primary rule (selected via SAAQ_RULE)
heartbeat_signal, heartbeat_enabled,
gpu_temp_c, gpu_power_w, cpu_tctl_c, cpu_package_power_w,
saaq_delta_q_legacy_prev, saaq_delta_q_legacy_target,   # SAAQ 1.0 trajectory
saaq_delta_q_v15_prev, saaq_delta_q_v15_target          # SAAQ 1.5 trajectory
```

The feature columns fed to SR.jl (`avg_pop_firing_rate_hz`, `membrane_dv_dt`, `routing_entropy`) are computed from activity/routing only and are therefore identical across rules — this makes dual-emission paired data for free. See `SnnDualLatentCalibrator` in `src/latent.rs`.

#### Recipes

Synthetic smoke (no CSV dependency, fast wiring check):

```bash
TICKS=64 HEARTBEAT_MATRIX=off \
  cargo run --release --example saaq_latent_calibration
```

RE4 corpus run (exact CSV length, zero wraparound):

```bash
TELEMETRY_SOURCE=csv TICKS=0 REPEAT_COUNT=1 HEARTBEAT_MATRIX=off \
  cargo run --release --example saaq_latent_calibration
```

Repeatability check (3 runs per model per heartbeat, deterministic inputs):

```bash
TELEMETRY_SOURCE=csv TICKS=0 REPEAT_COUNT=3 \
  cargo run --release --example saaq_latent_calibration
```

#### CWD routing-CSV caveat

`snn_gpu_routing_telemetry.csv` is hardcoded to the current working directory by `forward_gpu_temporal` / `compute_routing_telemetry` (see `src/model/core.rs`). `saaq_latent_calibration` drives the GPU via `tick_gpu_temporal`, which does **not** write that file, so it is safe today. If a future change starts invoking `forward_gpu_temporal` from the validation runner, every model's writes will silently merge into the same CWD-scoped file — move the sink into the per-run directory before doing that.

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

## Project Layout

| Path | Responsibility |
|------|----------------|
| `Cargo.toml` | crate config and dependencies |
| `src/lib.rs` | public API and crate docs |
| `src/types.rs` | shared scalar types and metadata structs |
| `src/error.rs` | `HybridError`, `Result` |
| `src/telemetry.rs` | `TelemetryEncoder` and `TelemetrySnapshot` |
| `src/projector.rs` | 2-bit spiking projector logic |
| `src/moe/mod.rs` | public MoE router shell |
| `src/moe/checkpoint.rs` | GGUF parsing and mapped tensor access |
| `src/moe/routing.rs` | routing math and expert selection |
| `src/funnel.rs` | TelemetryFunnel: encoder + split-bank bridge + sparse GIF hidden layer |
| `src/model/mod.rs` | public runtime shell and re-exports |
| `src/model/core.rs` | deterministic front-end + projector + OlmoeRouter orchestration |
| `src/model/temporal.rs` | GPU temporal tick orchestration |
| `src/model/telemetry_io.rs` | routing telemetry CSV helpers |
| `src/latent.rs` | `SnnLatentSnapshot`, `SnnLatentCalibrator`, `SnnLatentCsvExporter` for symbolic regression |
| `src/gpu/mod.rs` | GPU accelerator public API and module switchboard |
| `src/gpu/kernels/` | CUDA kernels (`.cu`, `.cuh`) for GPU-accelerated tasks |
| `src/gpu/wrappers/` | Safe Rust wrappers for CUDA context, memory, and kernel launches |
| `build.rs` | Build script to compile CUDA kernels to PTX |
| `examples/support/mod.rs` | shared example bootstrap and prompt helpers |
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
- The Allen Institute for AI for releasing the open OlmoeRouter models used as the primary testbed.

## Citation

If you use `corinth-canal` or the SNN-logic quantization approach in your research, please cite:

```bibtex
@misc{corinth-canal2026,
  title        = {corinth-canal: Turning MoE Architectures into SNN Quantization},
  author       = {Raul Montoya Cardenas},
  year         = {2026},
  howpublished = {\url{https://github.com/rmems/corinth-canal}},
  note         = {SNN-logic quantization reference crate with GIF-Ternary spiking for MoE models}
}
```

## License

GPL-3.0-or-later. See [LICENSE](LICENSE).
