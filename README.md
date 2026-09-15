# corinth-canal

Single-crate reference implementation of the `rmems` SNN-logic quantization
bridge.

[![License: Apache 2.0 OR MIT](https://img.shields.io/badge/license-Apache%202.0%20OR%20MIT-blue.svg)](LICENSE-APACHE)

This project is dual-licensed under the Apache 2.0 License or the MIT License (see `LICENSE-APACHE` and `LICENSE-MIT`).

## Overview

`corinth-canal` keeps the telemetry encoder, spiking hidden layer, projector,
GGUF-backed routing bridge, and SAAQ validation loop in one repository so the
full research path can be exercised end to end.

The current runtime flow is:

```text
TelemetrySnapshot
       |
       v  TelemetryEncoder (CPU) / project_snapshot_current (GPU)
ternary telemetry events (+1 / 0 / -1)
       |
       v  SignedSplitBankBridge (CPU) / GPU input_spikes
input spike train
       |
       v  SparseGifHiddenLayer (CPU) / gif_step_weighted_tick (GPU)
hidden spike train + membrane state
       |
       v  Projector
embedding [2048]
       |
       v  Router
expert_weights + selected_experts + routed hidden state
       |
       v  SAAQ latent calibration / telemetry export
```

## Scope

This repository is intentionally kept as a single crate. Proven components are
expected to graduate later into `rmems-*` crates according to
`docs/PROMOTION_RULES.md`, but `corinth-canal` itself remains the reference repo
for the full end-to-end loop.

## Implementation highlights

### Custom GGUF-backed routing bridge

The model-loading interface is custom to this repository.

`Router`:

- parses GGUF checkpoints in-repo
- resolves a supported model family from checkpoint metadata
- extracts routing tensors and token embeddings
- selects a GPU synapse source from one of:
  - real `F16`
  - dequantized `Q8_0`
  - dequantized `Q5_K`
  - dequantized `Q6_K`
  - dequantized `IQ3_M`
  - dequantized `INT4` (safetensors backend)
  - `routing-f32` (GGUF fallback when no preferred tensor matched)
  - `synthetic-fallback`

  These eight strings are what land in `synapse_source()` and the diagnostics
  JSON. `src/lib.rs` documents the priority order; treat it as the source of
  truth rather than duplicating the list again.

Supported families in code:

The `ModelFamily` enum in `src/types.rs` is the list — 21 variants at the time
of writing, including `Olmoe`, `Qwen3Moe`, `Gemma4`, `DeepSeek2`, `LlamaMoe`,
`Zaya`, `Glm4`, `Moonlight16BA3B`, `Granite31A800M`, `Nemotron`, `Lfm2Moe`,
`SlimMoe`, `GptOss`, `Step`, `MiniMax`, `Cohere`, `Grin`, `Skyworks`,
`Trinity`, `Grok` and `NemotronLegacy` — the last being a serde-only back-compat alias (`nemotron3nano4b`) with no human-facing name of its own. Consult the enum rather than this paragraph: prose lists
here have drifted before, and `ModelFamily::from_alias` is the authority.

### Model onboarding and cloud lineup

- `configs/saaq15_moe_lineup.toml` — shareable GGUF lineup template for SAAQ 1.5
- `configs/saaq15_cloud_lineup.toml` — cloud model metadata stubs with fail-fast
  env var guards (execution delegated to Dioscuri-Cloud)
- `configs/safetensors_lineup.template.toml` — shareable safetensors lineup template for
  manifest inspection (header-only, no tensor payload reads)
- `docs/CLOUD_MODELS.md` — cloud model delegation model and provider reference
- `docs/model_lineup.md` — rollout batch structure and required metadata

### Projection modes

- `RateSum`
- `TemporalHistogram`
- `MembraneSnapshot`
- `SpikingTernary`

### Routing modes

- `StubUniform`
- `DenseSim`
- `SpikingSim`

### Telemetry

`TelemetrySnapshot` carries:

- `gpu_temp_c`
- `gpu_power_w`
- `cpu_tctl_c`
- `cpu_package_power_w`
- `timestamp_ms`

### SAAQ calibration

The validation path uses `SnnDualLatentCalibrator`, which emits both the legacy
and v1.5 SAAQ trajectories in latent telemetry output while preserving legacy
compatibility columns.

## Primary docs

- `docs/ARCHITECTURE.md` — runtime architecture, module map, hidden control flow
- `docs/RUN_PROFILES.md` — validated commands and per-run outputs
- `docs/CUDA_VALIDATION.md` — tiered CUDA smoke, sanitizer, profiler, and hardware validation ladder
- `docs/CLOUD_MODELS.md` — cloud model delegation model and provider reference
- `docs/model_lineup.md` — rollout batch structure and required onboarding metadata
- `docs/MODEL_SOURCE_VERIFICATION_CHECKLIST.md` — pre-onboarding source verification gate
- `docs/PROMOTION_RULES.md` — module promotion rules
- `docs/MODULE_STATUS.md` — current module stability/promotion status
- `manifests/proven_components.toml` — machine-readable module status mirror

## Main entrypoints

- `examples/saaq_latent_calibration.rs` — primary research / validation loop
- `examples/gpu_smoke_test.rs` — GPU temporal smoke validation
- `examples/csv_replay.rs` — canonical telemetry CSV replay
- `examples/telemetry_bridge.rs` — routing demonstration
- `examples/safetensors_manifest.rs` — safetensors header inspection and manifest generation

## Validation outputs

The validation runner writes per-run artifacts including:

- `tick_telemetry.txt`
- `latent_telemetry.csv`
- `run_manifest.json`
- `summary.json`

`snn_gpu_routing_telemetry.csv` is conditional: it is produced only by GPU
routing paths that append telemetry rows (for example
`Model::forward_gpu_temporal`). The normal `saaq_latent_calibration` loop uses
`Model::tick_gpu_temporal`, so this CSV is not created in standard validation
runs.

When emitted, the GPU routing telemetry CSV schema is:

```text
token_idx,best_score,best_walker,spike_count,mean_adaptation,active_fraction
```

## Quick start

### Check the repo

```bash
just setup
cargo check --all-targets --no-default-features
cargo test --no-default-features
```

On CUDA-equipped setups with `nvcc` available, `just check` and `just test`
exercise the default feature set.

Snyk security scans (SCA/SAST/etc.) are available via Snyk MCP tools instead of CI workflow (GH#108).

### Run the primary validation loop

```bash
just saaq
```

### Run the GPU smoke path with a real checkpoint

```bash
CHECKPOINT_PATH=/path/to/model.gguf just smoke
```

For the full CUDA validation ladder, including host sanity checks, Compute
Sanitizer, Nsight Systems, Nsight Compute, and DCGM diagnostics, see
`docs/CUDA_VALIDATION.md`.

## Notes

- CPU-only buildability is preserved.
- CUDA/GPU behavior is preserved behind the `cuda` feature.
- Machine-local checkpoint discovery under `$HOME/Downloads/SNN_Quantization`
  is a reference-repo convention, not a portable interface.
