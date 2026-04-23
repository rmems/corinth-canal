# Architecture

`corinth-canal` is a single-crate reference implementation of the SNN-logic
quantization bridge. It is intentionally *not* split into modular crates
yet — the goal is to keep one working codebase where every component can be
exercised end-to-end, then promote proven modules into the `rmems` modular
repos (see `docs/PROMOTION_RULES.md` and `docs/MODULE_STATUS.md`).

## Block diagram

```text
TelemetrySnapshot
       │
       ▼  TelemetryEncoder (delta modulation) / project_snapshot_current (GPU)
[i8; 4] ternary spikes (+1 / 0 / -1)
       │
       ▼  SignedSplitBankBridge (CPU) or GPU input_spikes buffer
2048 input neurons
       │
       ▼  SparseGifHiddenLayer (CPU) or gif_step_weighted_tick (GPU, adaptive)
2048 GIF hidden neurons with adaptive thresholds + history-aware SAAQ
       │
       ▼  Projector (SpikingTernary)
dense embedding [EMBEDDING_DIM = 2048]
       │
       ▼  OlmoeRouter (stub | dense | spiking sim) with GGUF-backed routing
expert_weights + selected_experts + hidden + spike_count telemetry
```

## Module map

| Module | Role |
|--------|------|
| `src/model/core.rs` | Runtime orchestration, config validation, forward paths |
| `src/model/temporal.rs` | GPU temporal loop (`prepare_gpu_temporal`, `tick_gpu_temporal`, `forward_gpu_temporal`) |
| `src/model/telemetry_io.rs` | Shared CSV writer for routing telemetry |
| `src/moe/mod.rs` | `OlmoeRouter` host with routing-mode dispatch |
| `src/moe/checkpoint.rs` | GGUF parse, mmap, tensor slicing, `GgufMetadata` |
| `src/moe/adapter.rs` | Model-family adapter resolution (Olmoe, Qwen3Moe, Gemma4, DeepSeek2, LlamaMoe) |
| `src/moe/routing.rs` | Router math (gate scores, normalize, resample) |
| `src/projector.rs` | `ProjectionMode` spike → embedding |
| `src/funnel.rs` | Telemetry funnel + GIF hidden layer |
| `src/telemetry.rs` | `TelemetryEncoder`, `TelemetrySnapshot` |
| `src/latent.rs` | SAAQ 1.0 solo + dual 1.0/1.5 calibrators + CSV export |
| `src/gpu/` | cust-based kernel wrappers |
| `examples/support/config.rs` | `RunConfig::from_env()` — single source of env truth for examples |

## Hidden control flow

A few control paths are not obvious from a top-level read. They are called
out here so future promotion out of this reference repo does not lose them.

### CWD-relative write: `snn_gpu_routing_telemetry.csv`

`Model::forward_gpu_temporal` (and `Model::forward` → GPU routing) call
`append_gpu_routing_telemetry_row` with a path resolved from
`ModelConfig::gpu_routing_telemetry_path`. When that field is `None`,
the path falls back to the bare filename `snn_gpu_routing_telemetry.csv`,
which `std::fs` resolves against the process CWD.

- `saaq_latent_calibration` sets the path explicitly into
  `<run_dir>/snn_gpu_routing_telemetry.csv`, so it is self-contained.
- `telemetry_bridge` / `csv_replay` inherit the default (CWD). Launching
  them from the repo root is the intended workflow.
- `tick_gpu_temporal` does **not** touch this CSV; only the
  `forward_gpu_temporal` path does.

### Env-resolved paths

Every machine-specific path is resolved in `examples/support/config.rs::RunConfig::from_env()`.
The runtime crate itself (under `src/`) never reads environment variables
for paths. If you are promoting a module out of this repo, the `RunConfig`
surface is the contract to preserve.

### Telemetry source fallback

`resolve_telemetry_source()` stamps `source_label` into every
`run_manifest.json` with one of three values:

- `synthetic` — explicit or default stub
- `synthetic_fallback` — requested CSV but it was missing / malformed / empty
- `csv_<stem>` — canonical CSV path, e.g. `csv_re4` for `telemetry.csv`

This means the manifest is always truthful about what actually fed the run,
even when the configured source silently degraded.
