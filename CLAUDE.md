# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

`AGENTS.md` is the authoritative policy document for this repo (non-negotiable rules, approved tooling, git workflow, beads/session-close protocol). Read it too; this file covers build commands and the architectural context that requires reading several files to reconstruct.

## Build, test, lint

Two build worlds exist and they are not interchangeable:

- **CPU-only** (`--no-default-features`): no `nvcc`, no CUDA. The `cuda` feature is *default-on*, and it gates `pub mod gpu` and `pub mod model` in `src/lib.rs`. So under `--no-default-features` there is **no `Model` type at all** — only telemetry/funnel/projector/moe/latent/experiment. The five GPU examples declare `required-features = ["cuda"]` and silently vanish from CPU builds; the four that remain runnable are `validate_matrix`, `validate_local_saaq`, `summarize_local_saaq`, and `safetensors_manifest` (each documents its argv in its module doc comment).
- **CUDA** (default features): `build.rs` shells out to `nvcc`, compiles `src/gpu/kernels/*.cu` to `sm_120` fatbins plus the `myelin_shim` C-ABI object, and panics if `nvcc` is missing. `--features gpu-stub` writes empty fatbins as an optional non-CUDA fallback for local development; CI uses `--no-default-features` (CPU) or the default `cuda` feature on a CUDA runner.
- The `gguf` and `ci` features in `Cargo.toml` are **declared but referenced nowhere** in `src/`, `examples/`, or `build.rs`. `--features gguf` does not enable GGUF support — GGUF parsing is unconditional.

```bash
just setup                                   # scaffolding sanity check; warns if .env.local missing
cargo check --all-targets --no-default-features
cargo test --no-default-features
cargo fmt --all -- --check
cargo clippy --all-targets --no-default-features -- -D warnings -A dead_code   # CI's exact lint gate
```

On a CUDA box with `nvcc`, `just check` / `just test` exercise the default feature set, and `cargo build --examples` becomes meaningful.

Single test / narrower runs:

```bash
cargo test --no-default-features gate_scores          # substring filter across all targets
cargo test --lib --no-default-features -- --nocapture
cargo test --no-default-features --test examples_support_lineup     # #[path] into examples/support/lineup.rs
cargo test --no-default-features --test examples_support_telemetry  # #[path] into examples/support/telemetry_csv.rs
cargo test --no-default-features --test examples_support_embedding  # #[path] into examples/support/embedding.rs
cargo test --test gpu_sentry_telemetry            # default features; whole file is #[cfg(feature = "cuda")]
```

Unit tests live in `#[cfg(test)] mod tests` inside each `src/` file (two are split out: `src/moe/tests.rs`, `src/moe/safetensors/tests.rs`). `tests/` holds integration files that reach into `examples/support/*.rs` via `#[path]`: `examples_support_lineup`, `examples_support_telemetry`, `examples_support_embedding`. Those extracted-file tests run under self-hosted / local `--all-targets` (and plain `cargo test --no-default-features` locally); hosted PR CI is `--lib` only, so only library unit tests such as `ModelFamily::from_alias` run on every PR.

Coverage mirrors CI with `cargo llvm-cov --lib --no-default-features --locked --lcov --output-path lcov.info`.

## Running the research loop

`RunConfig`-based examples read config from env (`.env.local`, auto-loaded via `dotenvy`; copy from `.env.example`, which documents every key). `validate_matrix`, `validate_local_saaq`, `summarize_local_saaq`, and `safetensors_manifest` take positional arguments; `csv_replay` requires a positional CSV path and also reads its remaining configuration from env. Recipes in `justfile`:

```bash
just saaq                          # primary SAAQ latent calibration loop
just saaq-csv                      # forces TELEMETRY_SOURCE=csv (needs TELEMETRY_CSV_PATH)
just saaq-campaign                 # 2-phase synthetic + csv baseline campaign
CHECKPOINT_PATH=/path/model.gguf just smoke   # direct GPU temporal smoke path
just synapse-diag                  # print preferred GPU synapse tensor + ggml_type per model
just replay /path/telemetry.csv
just clean-artifacts
```

`docs/RUN_PROFILES.md` is the table of blessed `(example, telemetry source, SAAQ rule)` tuples — anything absent from it is experimental. `docs/CUDA_VALIDATION.md` holds the Tier 0–5 sanitizer/Nsight/DCGM ladder.

## Architecture

The crate is deliberately a **single crate** (see `AGENTS.md`); do not extract modules into `rmems-*` crates. `docs/MODULE_STATUS.md` + `manifests/proven_components.toml` track each module's promotion readiness, `docs/PROMOTION_RULES.md` the rules.

Pipeline (telemetry projection and the temporal loop have CPU and GPU counterparts; Projector, Router, and SAAQ calibration currently run on the CPU):

```text
TelemetrySnapshot  →  TelemetryEncoder (CPU) / project_snapshot_current (GPU)
  → ternary events [i8; 4]
  → SignedSplitBankBridge (CPU) / GPU input_spikes
  → SparseGifHiddenLayer (CPU) / gif_step_weighted_tick (GPU)
  → Projector → embedding [EMBEDDING_DIM = 2048]
  → Router → expert_weights + selected_experts + routed hidden state
  → SAAQ latent calibration / telemetry export
```

Reading order: `src/lib.rs` (exported surface) → `src/model/mod.rs` before `core.rs`/`temporal.rs` → `src/moe/mod.rs` before `adapter.rs`/`gguf/`/`routing.rs` → `examples/saaq_latent_calibration.rs` as the research entrypoint. `docs/ARCHITECTURE.md` is the long-form version of this section.

Module boundaries worth knowing before editing:

| Path | Role |
|------|------|
| `src/model/core.rs` | `Model` construction, config validation, forward paths |
| `src/model/temporal.rs` | `prepare_gpu_temporal` / `tick_gpu_temporal` / `forward_gpu_temporal` |
| `src/model/telemetry_io.rs` | the one CSV writer for GPU routing telemetry |
| `src/moe/mod.rs` | `Router` host + routing-mode dispatch; `moe` and `moe::safetensors` are public, plus `RoutingMode`, `ggml_type_label`, and `synapse_dequant_path_supported` re-exports; internal helpers live in private `gguf/`, `adapter.rs`, and `routing.rs` |
| `src/moe/gguf/` | GGUF parse, mmap, dequant, CUDA host-register (split out of `checkpoint.rs`) |
| `src/moe/checkpoint.rs` | thin compatibility façade re-exporting `gguf/` |
| `src/moe/adapter.rs` | model-family resolution + synapse/routing tensor selection |
| `src/moe/safetensors.rs` + `safetensors/` | header inspection + manifest generation, plus `MappedSafetensorsCheckpoint` token-embedding extraction and `safetensors_gate_scores` runtime routing |
| `src/experiment/schema.rs` | `RunMatrix` / `ExperimentManifest` / `ExperimentSummary` — the TOML matrix + `run_manifest.json` schemas the `validate_*` examples check |
| `src/types.rs` | `TelemetrySnapshot`, `ModelFamily`, `RoutingMode`, `ProjectionMode`, `CloudModelSpec`, `EMBEDDING_DIM` |
| `src/tensor/mod.rs`, `src/metric.rs` | tiny shared helpers (`Tensor = Vec<f32>`, dot, MSE); `metric` is `pub(crate)` |
| `examples/support/config.rs` | the env-truth surface; see boundary rule below |

### Invariants that are easy to break

- **`src/` never reads environment variables for paths.** All machine-local resolution (checkpoints, telemetry CSVs, artifact roots) lives in `examples/support/config.rs` and `examples/support/mod.rs`. Never introduce `/home/...` or env-based path discovery into `src/`. Grepping `env::var` in `src/` does hit two *non-path* readiness flags that are legitimate — `CloudModelSpec::cloud_provider_available` (`src/types.rs`) checks that declared credential vars are non-empty, and `RunMatrix::validate` (`src/experiment/schema.rs`) gates Grok-1 on `GROK1_ARTIFACT_READY=1`. Neither resolves a filesystem path.
- **Routing telemetry CSV has a legacy fallback.** When `ModelConfig::gpu_routing_telemetry_path` is `None`, `forward_gpu_temporal` / `compute_routing_telemetry` write to a CWD-relative `snn_gpu_routing_telemetry.csv`. The SAAQ runner sets the path explicitly; new call sites should too. Also note `tick_gpu_temporal` does *not* append routing rows, so standard validation runs never produce that CSV.
- **Synapse path is chosen from the tensor's actual `ggml_type`, not the filename.** The adapter inspects `blk.0.attn_q.weight`; a file named `...IQ4_NL.gguf` legitimately drives the `dequantized-q8_0` path. Mixed-quant checkpoints are expected. The eight `SynapseSource` variants (`src/moe/adapter.rs`) are `real`, `dequantized-q8_0`, `dequantized-q5_k`, `dequantized-q6_k`, `dequantized-iq3_m`, `dequantized-int4`, `routing-f32`, `synthetic-fallback` — those strings are what lands in `synapse_source()` and the diagnostics JSON. Non-square dequantized tensors are resampled to the neuron grid rather than rejected.
- **`ModelFamily` has 21 variants** (`src/types.rs`), including `Moonlight16BA3B`, `Granite31A800M`, `Nemotron`/`NemotronLegacy` (serde alias `Nemotron3Nano4B`), `Lfm2Moe`, `SlimMoe`, `GptOss`, `Step`, `MiniMax`, `Cohere`, `Grin`, `Skyworks`, `Trinity`, `Grok`. Read the enum — every prose list is stale: `src/lib.rs`'s doc comment names five, `README.md` names seven.
- **Do not change CSV schemas** (`latent_telemetry.csv`, the routing telemetry header, canonical telemetry CSV header) unless the task says so. Do not silently change what `src/lib.rs` re-exports.
- **Sentry/OTel stay disabled when `SENTRY_DSN` / `NR_INSERT_KEY` is blank.** When credentials are unset, no client is created, no network call is made, and the run continues. Wrappers attach the safe fields (`repo`, `command`, `git_sha`, `run_id`, `model_slug`, `telemetry_source`, `validation_status`, `error_category`, `prompt_profile`, `saaq_rule`) and avoid absolute checkpoint or artifact paths. `run_id` is also set as an OTel span attribute.
- **`configs/` filenames.** The files on disk are `hybrid_moe_lineup.toml`, `local_gguf_lineup.toml` (gitignored; copy from `local_gguf_lineup.template.toml`), `local_safetensors_lineup.template.toml`, `model_adapter_configs.toml`, and `saaq_cloud_lineup.toml` (`safetensors_lineup.toml` is a gitignored local copy created from the template). The old `saaq15_*` names were repaired across the `justfile`, `.env.example` and `docs/` — check `configs/` before quoting a path in new prose. `just saaq-campaign` now falls back to `configs/local_gguf_lineup.toml`, which is machine-local, so it still needs that file to exist or `LINEUP_CONFIG` set.

### Model selection precedence (examples only)

`RunConfig::from_env` validates any optional lineups that are set (`CLOUD_LINEUP_CONFIG`, `SAFETENSORS_LINEUP_CONFIG`) before resolving model precedence. It resolves models with the precedence `LINEUP_CONFIG` (errors and aborts only when set but unreadable; unset it to skip) → `SAFETENSORS_LINEUP_CONFIG` → `CHECKPOINT_PATH` → autodiscovery scan of the machine-local default root (configured in `examples/support/mod.rs`) for a hardcoded candidate list. Because validation runs before precedence resolution, a stale optional lineup can still abort the run even when a higher-priority config is the one used; unset optional lineups you are not actively using to avoid this. The autodiscovery root is a reference-repo convention only; do not copy it into a promoted crate unless you explicitly intend to maintain that machine-local scan.

Per-run artifacts land under `VALIDATION_OUTPUT_ROOT` (default `./artifacts`): `tick_telemetry.txt`, `latent_telemetry.csv`, `run_manifest.json`, `summary.json`. `run_manifest.json` stamps the *actual* telemetry source (`synthetic`, `synthetic_fallback`, `csv_<stem>`) so degradation is visible instead of silent.

## Conventions

- Rust edition 2024; every source file starts with `// SPDX-License-Identifier: Apache-2.0 OR MIT`.
- `[workspace]` in `Cargo.toml` is intentionally empty — it stops Cargo walking up into an unrelated parent workspace. Leave it.
- Dependencies are deliberately minimal; adding one needs clear justification. `rustls` is pinned to the `ring` backend on purpose (avoids `aws-lc-rs` C build deps).
- Commit style is conventional-commits with issue refs, e.g. `refactor(moe): split checkpoint.rs into private gguf/ modules (GH#118 PR-4) (#128)`. Keep behavioral changes separate from structural refactors.
- Prefer `git` CLI over MCP tools for branch/PR operations here. All work stays in the repo root — no extra worktrees.
- Task tracking goes through **beads** (`bd create` / `bd ready` / `bd close`), not TodoWrite or markdown TODO lists; `bd remember` holds cross-session notes. Run `bd prime` to reload that context. A session is not finished until changes are committed *and* pushed.
- CI passes `--locked`, so a `Cargo.lock` that drifts from `Cargo.toml` fails the build before any test runs.

## Code quality tooling

- **Codacy** (when available): run Codacy static analysis on changed files and dependency manifests, and fix the issues it reports. Do not use it to chase duplication, complexity metrics, or coverage. `.codacy.yml` excludes the intentionally-complex core (`src/model/**`, `src/gpu/**`, `src/moe/**`, `src/funnel.rs`, `src/telemetry.rs`, `src/latent.rs`, and the large examples).
- **Aikido** (when available): run Aikido scans on generated or modified first-party code and fix the findings.
- **Snyk** (when available): run Snyk SCA/SAST; the project does not rely on a dedicated Snyk CI workflow.

## CI

GitHub Actions is primary: `ci.yml` (CPU — fmt, clippy, `cargo test --lib --no-default-features`, `cargo check --examples`, llvm-cov → Codecov; then a self-hosted Ryzen job with a fork guard), `gpu-tests.yml` (CUDA ≥ 13.2 / sm_120 build validation), `docker-build.yml`, `sentry-release.yml`, `snyk-security.yml`. `scripts/*` is gitignored by design, so CI checkouts do not contain it.

Note the split in test scope: the hosted job runs only `--lib`, so `tests/` and example-target tests are exercised solely by the Ryzen job (`cargo test --all-targets --no-default-features --locked`), which is skipped for fork PRs. Run `--all-targets` locally before pushing rather than trusting the hosted job.
