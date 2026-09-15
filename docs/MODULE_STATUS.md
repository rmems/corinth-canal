# Module Status

Live snapshot of every `src/` module's position on the promotion ladder.
Machine-readable mirror: `manifests/proven_components.toml`.

Status legend: `reference` · `stabilizing` · `proven` · `frozen`
(see `docs/PROMOTION_RULES.md`).

| Module | Status | Target `rmems` crate | Notes |
|--------|--------|----------------------|-------|
| `src/model/core.rs` | reference | `rmems-model` | Orchestration layer still couples runtime behavior, artifact wiring, and GGUF-backed routing. Not yet ready to promote as an isolated surface. |
| `src/model/temporal.rs` | stabilizing | `rmems-model` | GPU temporal loop is tight and proven. Legacy fallback to a CWD-relative routing CSV still exists when `ModelConfig::gpu_routing_telemetry_path` is unset, so callers must keep the sink explicit. |
| `src/model/telemetry_io.rs` | stabilizing | `rmems-model` | Pure helper. Public behavior is stable; promotion depends mainly on the surrounding runtime/API cleanup. |
| `src/moe/mod.rs` | stabilizing | `rmems-moe` | Host entry for `Router`; surface is clean. Pending full model-family validation matrix. |
| `src/moe/checkpoint.rs` | reference | `rmems-moe` | Thin façade re-exporting `src/moe/gguf/` (metadata/map/dequant/cuda_register). GGUF parser/mmap/dequant works; needs broader format test battery before promotion. The parse layer is separately migrating to an `rmems/engram-parser` **dependency** (#115, blocked by #144 / rmems/engram-parser#45) — the GGUF parse layer uses a **real dependency**, whereas safetensors is a **one-way copy with no dependency**. |
| `src/moe/gguf/` | reference | `rmems-moe` | Internal split of former monolithic checkpoint: parse, mmap access, dequant, CUDA host register. Only the parse half is in scope for the engram-parser dependency in #115; dequant and CUDA host-register stay here. |
| `src/moe/adapter.rs` | stabilizing | `rmems-moe` | Adapter resolution covers all `ModelFamily` variants (21 at time of writing — consult the enum, not this note); needs broader validation coverage per family. |
| `src/moe/routing.rs` | stabilizing | `rmems-moe` | Stateless routing math. Low-risk promotion candidate. |
| `src/moe/safetensors/` | reference | `rmems/engram-parser` (feature `safetensors`; inspect half) | Directory split (#147) drew the boundary: extractable `manifest`/`validate`/`discovery`/`json`/`paths` vs corinth-specific `config`/`map` mmap load. #116 copies the inspect half **into engram-parser behind a cargo feature** — superseding the earlier dedicated `safetensors-parser` crate plan (rmems/engram-parser#10). **One-way copy:** corinth keeps this reference copy and takes **no** dependency on engram-parser. Status stays `reference` — see PROMOTION_RULES "One-way extractions". |
| `src/projector.rs` | stabilizing | `rmems-projector` | `ProjectionMode` surface is stable; `SpikingTernary` remains the live research path. |
| `src/funnel.rs` | reference | `rmems-funnel` | CPU GIF hidden layer is still shared with the broader runtime and validation path. |
| `src/telemetry.rs` | stabilizing | `rmems-telemetry` | Telemetry encoding surface is small and stable. `TelemetrySnapshot` carries physical telemetry channels and timestamps. |
| `src/latent.rs` | stabilizing | `rmems-latent` | Dual-SAAQ emission is in place. Determinism and campaign validation remain the main graduation gate. |
| `src/gpu/*` | reference | `rmems-gpu` | Kernel sources and cust wrappers remain coupled to the reference repo build/runtime assumptions. Promotion is still blocked on portability and validation breadth. |
| `src/types.rs` | stabilizing | `rmems-types` | The externally-visible vocabulary: `TelemetrySnapshot`, `ModelFamily` (21 variants), `RoutingMode`, `ProjectionMode`, `CloudModelSpec`, `EMBEDDING_DIM`. Serde aliases are load-bearing for existing matrices, so any rename is a breaking change. |
| `src/error.rs` | stabilizing | `rmems-types` | `HybridError` is the crate-wide error type. Two `ProjectionMode` helpers in `types.rs` still return `Result<_, String>`, and `src/gpu` has an unbridged `GpuError`; unifying those is the graduation gate. |
| `src/experiment/` | reference | `rmems-experiment` | `RunMatrix` / `ExperimentManifest` / `ExperimentSummary` — the TOML matrix and `run_manifest.json` schemas. `ExperimentBundle` and `ExperimentWarning` are declared but never constructed. |
| `src/tensor/` | reference | n/a | `Tensor = Vec<f32>` plus `zeros`/`dot`. Currently has no caller; kept as a shared-helper placeholder rather than promoted. |
| `src/metric.rs` | reference | n/a | `pub(crate)` MSE helper. Its only caller is cuda-gated, and `examples/csv_replay.rs` carries a second copy. |
| `examples/support/config.rs` | reference | n/a | Intentionally stays here — it is the env-truth surface for the reference repo only. |

## Known blockers

- **Machine-local discovery root.** `examples/support/mod.rs` walks
  `$HOME/Downloads/SNN_Quantization`. Fine for the reference repo; must
  not be copied into any `rmems` crate.
- **Legacy routing-CSV fallback.** The runtime now supports
  `ModelConfig::gpu_routing_telemetry_path`, but any caller that leaves it
  unset still falls back to the CWD-relative filename
  `snn_gpu_routing_telemetry.csv`.
- **`build.rs` fatbin compilation.** Assumes nvcc + `sm_120` targets on the
  author's box. `gpu-stub` covers the CI / non-CUDA case.
