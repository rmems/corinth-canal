# Run Profiles

Every validated tuple of `(example, telemetry source, SAAQ rule)`
is catalogued here with the exact command. Anything not in this table has
not been blessed and should be considered experimental.

## Prerequisites

- Fill in `.env.local` from `.env.example`.
- `just setup` — sanity check that the scaffolding is in place.

## SAAQ latent calibration (primary research loop)

| Profile | Telemetry | SAAQ rule | Command |
|---------|-----------|-----------|---------|
| Smoke, synthetic | synthetic | 1.5 | `just saaq` |
| SR.jl corpus, full loop | csv (RE4) | 1.5 + 1.0 dual | `TICKS=0 just saaq-csv` |
| Wraparound sanity | csv (RE4) | 1.5 | `TICKS=2000 just saaq-csv` |
| Legacy rule parity | synthetic | 1.0 | `SAAQ_RULE=legacy just saaq` |

Dual-SAAQ columns are emitted regardless of `SAAQ_RULE`; the env var only
selects which rule fills the legacy compatibility columns.

### Validation outputs

`examples/saaq_latent_calibration.rs` writes a per-run directory under the
configured output root. Each run directory contains:

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

The latent telemetry CSV includes both SAAQ trajectories via
`SnnDualLatentCalibrator`:

- legacy / primary compatibility columns
- explicit legacy trajectory columns
- explicit v1.5 trajectory columns

### Operational notes

- `TICKS=0` with `TELEMETRY_SOURCE=csv` uses the full CSV row count so corpus
  runs cover exactly one telemetry loop.
- When `TICKS > csv_row_count`, the runner warns about wraparound contamination.
- `STRICT_REPEAT_CHECK=true` enables repeat-to-repeat comparison in the
  validation workflow.
- `run_manifest.json` stamps the actual telemetry source label, including
  `synthetic_fallback` when CSV replay degrades.
- `PROJECTION_MODE` selects the projector the same way `ROUTING_MODE`
  selects the router. Unset / blank keeps `SpikingTernary`. Accepted
  values: `RateSum`, `TemporalHistogram`, `MembraneSnapshot`,
  `SpikingTernary` (also snake_case and kebab-case forms such as
  `rate_sum` / `rate-sum`, `temporal_histogram` / `temporal-histogram`,
  `membrane_snapshot` / `membrane-snapshot`,
  `spiking_ternary` / `spiking-ternary`). An unrecognised value — or one
  that is not valid Unicode — fails fast. `run_manifest.json` and
  `summary.json` stamp `projection_mode` so a run is reproducible from
  the manifest.
- `PROJECTION_MODE=TemporalHistogram` is **rejected** by
  `saaq_latent_calibration`. That loop feeds the projector a single-tick
  spike train, so the four histogram bins collapse onto bin 0 and the mode
  degenerates into a rescaled `RateSum` carrying no temporal information.
  Rather than stamp `temporal_histogram` on a run that did not perform it,
  the runner exits 1. Supporting it needs a real multi-tick window in the
  tick loop, which changes what the calibrator observes per row and is
  tracked as follow-up to GH#162. `telemetry_bridge` is unaffected.

## GPU smoke test

| Profile | Command |
|---------|---------|
| 10k GPU ticks, real checkpoint | `CHECKPOINT_PATH=... just smoke` |
| Short deterministic GPU smoke | `GPU_SMOKE_TICKS=1 CHECKPOINT_PATH=... cargo run --release --example gpu_smoke_test` |

This example is the direct GPU-temporal smoke path. It is the correct entrypoint
for validating resident synapse upload, GIF weighted temporal stepping, and the
on-device best-walker reduction.

The full Tier 0-5 CUDA validation ladder, including Compute Sanitizer, Nsight
Systems, Nsight Compute, and DCGM commands, lives in `docs/CUDA_VALIDATION.md`.

## Synapse diagnostic

| Profile | Command |
|---------|---------|
| Probe preferred synapse tensor selection only | `just synapse-diag` |

`examples/synapse_diagnostic.rs` is the cheapest way to explain why a checkpoint
selected any of the eight sources — `real`, `dequantized-q8_0`,
`dequantized-q5_k`, `dequantized-q6_k`, `dequantized-iq3_m`,
`dequantized-int4`, `routing-f32` or `synthetic-fallback`. It does not run SAAQ ticks or bring up the GPU temporal
loop; it only loads the GGUF metadata and the preferred synapse tensor facts
that drive `src/moe/adapter.rs::resolve_adapter`.

The console line reports both:

- the file-level family / architecture / quantization context
- the selected tensor's actual `ggml_type`, dimensions, and derived synapse
  source

This matters for mixed-quant checkpoints. If the example prints a file whose
name contains `IQ4_NL` but the `type=` field shows `Q8_0` and the `source=`
field shows `dequantized-q8_0`, that is expected: the adapter branches on the
actual `ggml_type` of `blk.0.attn_q.weight`, not on the filename suffix.

The example also writes `<output_root>/synapse_diagnostic.json` for a structured
record of the same fields.

## Cloud model lineup

Cloud model execution is delegated to Dioscuri-Cloud. corinth-canal is
responsible for model selection, experiment metadata stamping, and fail-fast
validation — not for infrastructure provisioning.

### Cloud lineup config

| Profile | Command |
|---------|---------|
| Validate cloud lineup metadata file | `cargo test --no-default-features cloud_lineup -- --nocapture` |

Each cloud entry carries:

- `cloud_model_id` — provider-qualified identifier passed to Dioscuri-Cloud
- `source_url` — canonical model card
- `target` — always `"cloud"`
- `architecture` — `"moe"` or `"dense"`
- `active_params` / `total_params` — informational parameter counts
- `provider_format` — expected runtime format (`nvcf-nim`, `openai-compat`, `vertex-ai`, `watsonx-saas`, `fp8-safetensors`)
- `required_env_vars` — env var names that must be set for execution

`CLOUD_LINEUP_CONFIG` parsing and cloud execution guards currently live in
`examples/support/mod.rs`. The main `just saaq` runner path is not yet wired to
consume cloud lineup config directly.

## Safetensors manifest

| Profile | Command |
|---------|---------|
| Inspect a single Safetensors checkpoint | `cargo run --example safetensors_manifest --no-default-features -- <checkpoint-or-dir> artifacts/safetensors_manifest.json` |

The safetensors lineup template (`configs/safetensors_lineup.template.toml`) can be copied to `configs/safetensors_lineup.toml`; helper utilities in `examples/support/mod.rs` parse the local copy. The `safetensors_manifest` example
currently uses positional CLI arguments for single-checkpoint inspection.

Local entries onboarded:

| Slug | Model | Shards |
|------|-------|--------|
| `nemotron_3_nano_4b` | nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16 | 1 |
| `granite_3_1_3b_a800m` | ibm-granite/granite-3.1-3b-a800m-base | 2 |
| `trinity_nano_base` | arcee-ai/Trinity-Nano-Base | 7 |
| `phi_tiny_moe_instruct` | microsoft/Phi-tiny-MoE-instruct | 2 |
| `moonlight_16b_a3b_bnb_4bit` | slowfastai/Moonlight-16B-A3B-bnb-4bit | 2 |

Use Safetensors manifests when the goal is checkpoint anatomy: tensor names,
dtypes, shapes, byte sizes, source shards, and recognizable MoE router/expert
candidates. The manifest generator reads only Safetensors headers plus optional
Hugging Face `.safetensors.index.json` metadata; it does not create a project,
run activation tracing, or load tensor payload bytes.

Use GGUF when the goal is current `corinth-canal` runtime routing. The
`Router` path remains GGUF-backed for token embedding extraction, routing
tensor access, and GPU synapse-source selection. Safetensors support is a setup
and inspection adapter for experiment preparation, not a replacement for the
runtime GGUF bridge.

## CSV replay

| Profile | Command |
|---------|---------|
| Ingest canonical telemetry CSV | `just replay /path/to/telemetry.csv` |

Canonical CSV schema consumed by replay and validation:

```text
timestamp_ms,gpu_temp_c,gpu_power_w,cpu_tctl_c,cpu_package_power_w
```

## Telemetry bridge demo

| Profile | Command |
|---------|---------|
| Spiking routing | `just bridge` |
| Dense routing | `ROUTING_MODE=dense just bridge` |
| Stub routing | `ROUTING_MODE=stub just bridge` |
| Rate-sum projection | `PROJECTION_MODE=RateSum just bridge` |
| Temporal-histogram projection | `PROJECTION_MODE=TemporalHistogram just bridge` |
| Membrane-snapshot projection | `PROJECTION_MODE=MembraneSnapshot just bridge` |

## Model discovery

If `CHECKPOINT_PATH` is unset, `saaq_latent_calibration` auto-discovers
up to five MoE families under `$HOME/Downloads/SNN_Quantization/`:

- `olmoe-0125-gguf/OLMoE-1B-7B-0125-Instruct-F16.gguf`
- `models/qwen3-moe-i1-GGUF/qwen3-moe.i1-IQ3_M.gguf`
- `models/gemma-4-26B-A4B-it-GGUF/gemma-4-26B-A4B-it-UD-IQ4_NL.gguf`
- `models/DeepSeek-Coder-V2-Lite-Instruct-GGUF/DeepSeek-Coder-V2-Lite-Instruct-Q6_K_L.gguf`
- `models/Llama-3.2-8X3B-MOE-Dark-Champion-GGUF/L3.2-8X3B-MOE-Dark-Champion-Inst-18.4B-uncen-ablit_D_AU-q5_k_m.gguf`

This discovery root is a machine-local convention on the author's Fedora
box. CI and contributor machines should set `CHECKPOINT_PATH`
explicitly.

Do not assume the filename suffix tells you which GPU synapse path will be
used. Synapse selection is per-tensor: the adapter inspects the actual
`ggml_type` of `blk.0.attn_q.weight`. A checkpoint labeled `IQ4_NL` at the file
level may still route through `dequantized-q8_0` if that tensor is stored as
`Q8_0` in the GGUF payload.

## Supported routing/model surfaces reflected in code

Routing modes:

- `StubUniform`
- `DenseSim`
- `SpikingSim`

Projection modes (selected via `PROJECTION_MODE`, default `SpikingTernary`):

- `RateSum`
- `TemporalHistogram`
- `MembraneSnapshot`
- `SpikingTernary`

Model families supported by the GGUF adapter layer:

- `Olmoe`
- `Qwen3Moe`
- `Gemma4`
- `DeepSeek2`
- `LlamaMoe`
- `Zaya`
- `Glm4`
