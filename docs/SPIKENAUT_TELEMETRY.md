# Spikenaut telemetry → corinth-canal SAAQ

Adapter for [rmems/Spikenaut-SNN-Telemetry](https://huggingface.co/datasets/rmems/Spikenaut-SNN-Telemetry)
(Vault JSONL: `neuromorphic_data`, `node_sync_harvest`, `ghost_market_log`,
`qubic_ticks_snn`, plus v3 `state_telemetry`) onto the canonical corinth replay
CSV consumed by `TELEMETRY_SOURCE=csv`.

This is **ingest only**. It does not change `TelemetrySnapshot` or the five-column
CSV schema. Gaming-telemetry mix (already `export_csv` to the same header) remains
a follow-up.

## Canonical sink

```text
timestamp_ms,gpu_temp_c,gpu_power_w,cpu_tctl_c,cpu_package_power_w
```

`TelemetryEncoder` / `TelemetryFunnel` then turn those four channels into ternary
events with thresholds `[1.0, 5.0, 1.0, 5.0]` (°C, W, °C, W). Units on the CSV
are therefore **encoder coordinates**, not a claim that every source literally
measured a CPU Tctl.

## Commands

```bash
# JSONL → canonical CSV (CPU-only; no CUDA)
just spikenaut-ingest tests/fixtures/spikenaut/gpu_sample.jsonl artifacts/spikenaut_gpu.csv

# Same conversion plus CPU dual-SAAQ (1.0 + 1.5) smoke + run_manifest.json
just spikenaut-smoke tests/fixtures/spikenaut/gpu_sample.jsonl

# GPU campaign replay of a converted corpus (needs a checkpoint)
TELEMETRY_SOURCE=csv TELEMETRY_CSV_PATH=artifacts/spikenaut_gpu.csv just saaq-csv
```

Domain override: `--domain gpu|mining|hft|qubic|state` (default `auto`).

## Timestamp policy

Copied from the dataset contract: **never synthesize a wall clock**.

| Source | `timestamp_ms` |
|--------|----------------|
| GPU (`neuromorphic_data`) | `row_index` (ordinal; the collector emitted no time) |
| Mining | parsed `timestamp` when it is a datetime; otherwise the 0-based JSONL ordinal. `dynex:919876`-style stamps are not datetimes and stay null in the published JSONL |
| HFT / Qubic | parsed ISO-8601 (`T` or space separator, optional fractional seconds and `Z`/`±HH:MM`) |
| v3 state | `ts_utc` (ns → ms) when present; else `step_idx` |

A missing timestamp is an ordinal, never `base + 10s × index`.

## Field mapping

Missing stays missing: a null or absent required channel **drops the row**.
No `0.0` fill.

### GPU — `neuromorphic_data.jsonl` / Hub `gpu_telemetry`

Physical sensors exist for GPU temp and power. The capture has no CPU package
channels, so the remaining two slots are the other on-device analogs.

| CSV column | JSONL field | Notes |
|------------|-------------|--------|
| `timestamp_ms` | `row_index` | ordinal, not unix time |
| `gpu_temp_c` | `gpu_temp_c` | measured |
| `gpu_power_w` | `power_w` | measured |
| `cpu_tctl_c` | `vram_temp_c` | memory thermal analog |
| `cpu_package_power_w` | `mem_util_pct` | utilization analog into the power channel (delta encoder) |

Interleaved `qubic_*` columns on this file are **not** mapped here; use the
`qubic` domain (or Hub `qubic_signals`) when those are the fuel.

### Mining — `node_sync_harvest.jsonl` / Hub `mining`

| CSV column | JSONL field | Notes |
|------------|-------------|--------|
| `timestamp_ms` | `timestamp` | real datetimes only |
| `gpu_temp_c` | `gpu_temp_c` | measured |
| `gpu_power_w` | `power_w` | measured |
| `cpu_tctl_c` | `hashrate_mh * 80` | published hashrate is 0–1 |
| `cpu_package_power_w` | `reward_hint * 200` | published reward hint is 0–1 |

### HFT — `ghost_market_log.jsonl` / Hub `hft`

No physical thermals. These are **axon-style affine surrogates** so the existing
4-channel `TelemetryEncoder` can run without a new snapshot schema and without
a dependency on `Limen-Neural/axon-encoder`. Scales are chosen so a typical
1-unit move sits near the encoder thresholds (`1.0` / `5.0`).

| CSV column | JSONL field | Affine |
|------------|-------------|--------|
| `timestamp_ms` | `timestamp` | ISO-8601 |
| `gpu_temp_c` | `portfolio_value` | identity (~$1 ≈ 1 ° analog) |
| `gpu_power_w` | `trade_value_usdt * 100` | ~$0.05 ≈ 5 W analog |
| `cpu_tctl_c` | `70 + cumulative_pnl * 10` | ~0.1 PnL ≈ 1 ° analog |
| `cpu_package_power_w` | `price_usd / 100` | asset switches spike hard |

A later axon-encoder pass can re-encode the raw HFT columns (rate / delta /
population) for a different simulator; this adapter only produces valid
corinth replay CSV.

### Qubic — `qubic_ticks_snn.jsonl` / Hub `qubic_ticks`

Independent signals are `tick_rate` and `qubic_tick_trace` only.
`*_derived` columns are a fixed function of `tick_rate` and are **never**
consumed as measurements.

| CSV column | JSONL field | Affine |
|------------|-------------|--------|
| `timestamp_ms` | `timestamp` | ISO-8601 |
| `gpu_temp_c` | `qubic_tick_trace * 100` | population channel A |
| `gpu_power_w` | `tick_rate * 400` | population channel B |
| `cpu_tctl_c` | `qubic_tick_trace * 80` | second affine of A |
| `cpu_package_power_w` | `tick_rate * 200` | second affine of B |

### v3 state — Hub `state_telemetry`

When CPU/board collectors are present this is the preferred physical mapping.

| CSV column | JSONL field | Fallback |
|------------|-------------|----------|
| `timestamp_ms` | `ts_utc` (ns) | `step_idx` |
| `gpu_temp_c` | `gpu_temp_c` | — |
| `gpu_power_w` | `power_w` | — |
| `cpu_tctl_c` | `cpu_temp_c` | `vram_temp_c` |
| `cpu_package_power_w` | `board_power_w` | `cpu_util_pct` |

Raw Vault envelopes of the form `{"telemetry": {…}}` are flattened before
classification.

## Domain label

`run_manifest.json` stamps:

- `telemetry_source`: `csv_spikenaut_<domain>` (`gpu` / `mining` / `hft` / `qubic` / `state`)
- `run_tag`: `spikenaut_<domain>`

GPU campaign replay of a file named `spikenaut_gpu.csv` gets the same
`csv_spikenaut_gpu` label from `examples/support/telemetry_csv.rs`.

## Dual-SAAQ smoke

`--smoke` runs `SnnDualLatentCalibrator` on the CPU funnel + projector +
`RoutingMode::StubUniform` router (no CUDA, no checkpoint). It writes the
usual validation files:

- `tick_telemetry.txt`
- `latent_telemetry.csv` (legacy + v1.5 columns)
- `run_manifest.json` (`saaq_dual_emit: true`)
- `summary.json`

That is the ingest acceptance smoke. Full GPU campaigns still go through
`examples/saaq_latent_calibration.rs` with `TELEMETRY_SOURCE=csv`.
