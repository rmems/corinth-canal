# Architecture

`corinth-canal` is the single-crate reference implementation of the `rmems`
SNN-logic quantization bridge. It keeps the telemetry encoder, spiking hidden
layer, projector, GGUF-backed routing bridge, and validation artifacts in one
repository so the full research loop can be exercised within the required
single-crate layout.

## Block diagram

```text
TelemetrySnapshot
       │
       ▼  TelemetryEncoder (CPU) / project_snapshot_current (GPU)
[i8; 4] ternary telemetry events (+1 / 0 / -1)
       │
       ▼  SignedSplitBankBridge (CPU) / GPU input_spikes buffer
input spike train
       │
       ▼  SparseGifHiddenLayer (CPU) / gif_step_weighted_tick (GPU)
hidden spike train + membrane/adaptation state
       │
       ▼  Projector
embedding [EMBEDDING_DIM = 2048]
       │
       ▼  Router (stub | dense | spiking sim)
expert_weights + selected_experts + routed hidden state
       │
       ▼  SAAQ latent calibration / telemetry export
```

## CPU and GPU paths

### CPU path

The CPU path is assembled from pure Rust components:

- `TelemetryEncoder` converts a `TelemetrySnapshot` into `[i8; 4]` ternary
  events using per-channel delta thresholds.
- `SignedSplitBankBridge` expands those ternary events into an input spike
  train.
- `SparseGifHiddenLayer` runs a 2048-neuron GIF hidden layer with adaptive
  thresholds.
- `Projector` converts hidden activity into a 2048-dimensional embedding.
- `Router` consumes the embedding and produces expert weights / selected
  experts.

### GPU path

The GPU temporal path is orchestrated by `Model::prepare_gpu_temporal`,
`Model::tick_gpu_temporal`, and `Model::forward_gpu_temporal`.

The focused CUDA validation ladder for this path is documented in
`docs/CUDA_VALIDATION.md`.

Key pieces:

- `project_snapshot_current` projects 4-channel telemetry into the GPU temporal
  input buffer.
- `gif_step_weighted_tick` advances the resident GIF temporal state.
- GPU synapse weights are loaded from a GGUF-backed source when available.
- The GPU path reuses resident synapse weights across calls using a source
  signature cached in `GpuAccelerator`.
- `forward_gpu_temporal` writes routing telemetry through the shared CSV helper.
- `tick_gpu_temporal` advances GPU state but does not itself append routing CSV
  rows.

### GPU Observability

CUDA kernel launch failures are captured and sent to Sentry after the caller
initializes Sentry (the examples do this via `examples/support/observability.rs`
when `SENTRY_DSN` is configured). Each failure event includes:

- Kernel name and launch type (PTX/fatbin or C ABI shim)
- Grid and block dimensions
- Shared memory allocation
- CUDA error codes and messages
- JIT compilation logs (when applicable for module load failures)

Events are fingerprinted by `[kernel_name, error_category]` for efficient
grouping in Sentry dashboards. This telemetry is opt-in: when no Sentry client
is active, the capture layer is a no-op and introduces no runtime overhead.

Instrumented launch sites:

- **PTX/fatbin kernels** (via `cust::launch!` macro):
  - `project_snapshot_current` — telemetry projection
  - `lif_step` — LIF neuron step
  - `gif_step_weighted` — GIF step with F32 synapses
  - `reset_membrane` — membrane potential reset
  - `satsolver_extract` — SAT solution extraction
  - `satsolver_aux_update` — SAT auxiliary update
  - `satsolver_best_reduce_pass2` — SAT reduction pass 2

- **C ABI shim kernels** (Blackwell-critical F16 paths):
  - `gif_step_weighted_f16` — GIF step with F16 synapses
  - `saaq_find_best_walker` — SAAQ on-device reduction

- **Module load failures**:
  - Fatbin/PTX JIT compilation errors with full diagnostic logs

## Module map

| Module | Role |
|--------|------|
| `src/model/core.rs` | Runtime orchestration, config validation, forward paths |
| `src/model/temporal.rs` | GPU temporal loop (`prepare_gpu_temporal`, `tick_gpu_temporal`, `forward_gpu_temporal`) |
| `src/model/telemetry_io.rs` | Shared CSV writer for GPU routing telemetry |
| `src/moe/mod.rs` | `Router` host with routing-mode dispatch |
| `src/moe/checkpoint.rs` | Compatibility façade re-exporting `src/moe/gguf/` |
| `src/moe/gguf/` | GGUF parse, mmap, tensor slicing, dequantization, CUDA host register |
| `src/moe/adapter.rs` | Model-family adapter resolution and tensor selection |
| `src/moe/routing.rs` | Router math (gate scores, resampling, normalization, top-k) |
| `src/moe/safetensors/` | Safetensors façade: inspection/manifest (`discovery`/`json`/`paths`/`manifest`/`validate`) + HF config + mmap load (`config`/`map`) |
| `src/projector.rs` | `ProjectionMode` spike-to-embedding projection |
| `src/funnel.rs` | Telemetry funnel, signed split banks, GIF hidden layer |
| `src/telemetry.rs` | `TelemetryEncoder` and `TelemetrySnapshot` bridge |
| `src/latent.rs` | SAAQ 1.0 / 1.5 calibration and CSV export |
| `src/gpu/` | CUDA wrappers, buffers, kernel launchers |
| `examples/support/config.rs` | Example-only environment/config resolution |

## Model loading and routing bridge

The runtime model-loading interface is custom and GGUF-backed. Safetensors are
supported as an inspection/import surface for checkpoint anatomy manifests; they
do not replace GGUF-backed routing in the current runtime path.

Important nuance: the adapter chooses the GPU synapse path from the selected
tensor's actual `ggml_type`, not from the checkpoint filename or the GGUF-wide
quantization label. Mixed-quant checkpoints are therefore expected. For
example, a file named `...IQ4_NL.gguf` can still drive the `dequantized-q8_0`
path when `blk.0.attn_q.weight` is stored as `Q8_0` inside the GGUF.

`Router`:

- memory-maps GGUF checkpoints in-repo
- resolves a supported model family from GGUF metadata
- locates the routing tensor and token embedding tensor
- exposes token embedding extraction for validation workflows
- inspects `blk.0.attn_q.weight` directly when selecting the GPU synapse source
- selects a GPU synapse source from one of:
  - real `F16`
  - dequantized `Q8_0`
  - dequantized `Q5_K`
  - synthetic fallback

`moe::safetensors` (directory split for #116 extract boundary):

- **Inspection half** (`manifest`/`validate`/`json`/`paths`/`discovery`): reads
  only Safetensors headers, not tensor payload bytes; accepts a file, HF index,
  or directory of shards; labels MoE router/expert candidates; emits deterministic
  JSON manifests
- **Load half** (`config`/`map`): HF `config.json` parse + mmap'd
  `MappedSafetensorsCheckpoint` for router bridge tensor extraction

When the selected tensor is dequantized and not already square, the GPU
temporal path resamples it to the neuron grid instead of rejecting the
checkpoint outright.

Supported families in code today:

- `Olmoe`
- `Qwen3Moe`
- `Gemma4`
- `DeepSeek2`
- `LlamaMoe`
- `Zaya`
- `Glm4`

### Cloud model metadata

`CloudModelSpec` (in `src/types.rs`) carries metadata stubs for cloud-hosted
models that cannot be executed locally. Cloud execution is delegated to
Dioscuri-Cloud. corinth-canal is responsible for candidate selection,
manifest stamping, and fail-fast validation when required cloud provider env
vars are unset — not for infrastructure provisioning.

The cloud lineup lives in `configs/saaq_cloud_lineup.toml`. Each entry
records: `cloud_model_id`, `source_url`, architecture class, parameter counts,
provider format, and the env var names required for cloud execution. No
secrets or absolute paths are stored.

## Routing / projection modes

### Projection modes

- `RateSum`
- `TemporalHistogram`
- `MembraneSnapshot`
- `SpikingTernary` (default when `PROJECTION_MODE` is unset)

Operators select a mode via `PROJECTION_MODE` (same env surface as
`ROUTING_MODE`). The calibration runner stamps the resolved label onto
`run_manifest.json` / `summary.json` as `projection_mode`
(`rate_sum`, `temporal_histogram`, `membrane_snapshot`,
`spiking_ternary`). An unrecognised value fails fast.

### Routing modes

- `StubUniform`
- `DenseSim`
- `SpikingSim`

## Validation entrypoint and artifacts

The primary research/validation loop is `examples/saaq_latent_calibration.rs`.
For each run it writes artifacts including:

- `tick_telemetry.txt`
- `latent_telemetry.csv`
- `run_manifest.json`
- `summary.json`

`snn_gpu_routing_telemetry.csv` is conditional: it is produced only by GPU
routing paths that append telemetry rows. The normal
`examples/saaq_latent_calibration.rs` loop uses `Model::tick_gpu_temporal`, so
this CSV is not created in standard validation runs.

When emitted, the GPU routing telemetry CSV schema is:

```text
token_idx,best_score,best_walker,spike_count,mean_adaptation,active_fraction
```

The latent telemetry CSV includes both SAAQ trajectories via
`SnnDualLatentCalibrator`; one rule is selected as the primary/legacy
compatibility projection while the legacy and v1.5 columns are both emitted.

## Hidden control flow

A few control paths are easy to miss from a top-level read.

### Routing telemetry CSV path behavior

`Model::forward_gpu_temporal` and `Model::compute_routing_telemetry` resolve the
routing telemetry sink through `ModelConfig::gpu_routing_telemetry_path`.
When this field is `None`, the runtime falls back to the legacy
CWD-relative filename `snn_gpu_routing_telemetry.csv`.

Implications:

- `examples/saaq_latent_calibration.rs` sets the path explicitly into the run
  directory, so any routing telemetry emitted by a writing path stays per-run.
- The same runner currently advances the GPU loop through
  `Model::tick_gpu_temporal`, which does not append routing CSV rows on its own.
- Other call sites may still rely on the legacy fallback when they do not set
  the path explicitly.

### Env-resolved paths

Machine-specific path discovery belongs in `examples/support/config.rs`.
The library code under `src/` does not perform environment-variable path
resolution for checkpoints, telemetry CSVs, or artifact roots.

### Telemetry source stamping

The validation runner stamps a source label into `run_manifest.json` using one
of:

- `synthetic`
- `synthetic_fallback`
- `csv_<stem>`

This makes fallback behavior explicit in artifacts instead of hiding it behind a
successful run.

The same runner stamps `routing_mode` and `projection_mode` so a later
reproduce does not have to guess which projector or router path produced
the latent series.

## Observability

The example binaries share observability helpers under
`examples/support/observability.rs`.

- `command_start` and `command_finish` tracing events are emitted for every
  example command.
- Sentry is opt-in only. If `SENTRY_DSN` is unset or blank, the examples remain
  local/offline.
- The wrappers attach only safe diagnostic fields such as `repo`, `command`,
  `git_sha`, `model_slug`, `telemetry_source`, `validation_status`, and
  `error_category`.
- Absolute checkpoint paths and artifact paths are not attached as Sentry tags
  by the wrappers.
