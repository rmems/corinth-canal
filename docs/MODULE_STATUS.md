# Module Status

Live snapshot of every `src/` module's position on the promotion ladder.
Machine-readable mirror: `manifests/proven_components.toml`.

Status legend: `reference` · `stabilizing` · `proven` · `frozen`
(see `docs/PROMOTION_RULES.md`).

| Module | Status | Target `rmems` crate | Notes |
|--------|--------|----------------------|-------|
| `src/model/core.rs` | reference | `rmems-model` | Orchestration layer; still entangled with `Model::forward_gpu_temporal` CWD write. Gated on Stage E of the reference-repo cleanup. |
| `src/model/temporal.rs` | stabilizing | `rmems-model` | GPU temporal loop is tight and proven; needs its routing-CSV path made explicit before freezing. |
| `src/model/telemetry_io.rs` | stabilizing | `rmems-model` | Pure helper, ready to move once the caller stops passing a CWD-relative path. |
| `src/moe/mod.rs` | stabilizing | `rmems-moe` | Host entry for `OlmoeRouter`; surface is clean. Pending full matrix. |
| `src/moe/checkpoint.rs` | reference | `rmems-moe` | Pre-existing WIP in the parser is patched. Needs a proper test battery before promotion. |
| `src/moe/adapter.rs` | stabilizing | `rmems-moe` | Five-family support validated on the author's machine; needs CI matrix. |
| `src/moe/routing.rs` | stabilizing | `rmems-moe` | Stateless math. Low-risk promotion candidate. |
| `src/projector.rs` | stabilizing | `rmems-projector` | `ProjectionMode` surface frozen; SpikingTernary path is the live one. |
| `src/funnel.rs` | reference | `rmems-funnel` | GIF hidden layer still shared between CPU and GPU callers. |
| `src/telemetry.rs` | stabilizing | `rmems-telemetry` | Schema frozen (`timestamp_ms,gpu_temp_c,gpu_power_w,cpu_tctl_c,cpu_package_power_w`). |
| `src/latent.rs` | stabilizing | `rmems-latent` | Dual-SAAQ emission is working; needs a REPEAT_COUNT=2 determinism check to graduate. |
| `src/gpu/*` | reference | `rmems-gpu` | Kernel sources and cust wrappers. Promotion blocked on build.rs portability. |
| `examples/support/config.rs` | reference | n/a | Intentionally stays here — it is the env-truth surface for the reference repo only. |

## Known blockers

- **Machine-local discovery root.** `examples/support/mod.rs` walks
  `$HOME/Downloads/SNN_Quantization`. Fine for the reference repo; must
  not be copied into any `rmems` crate.
- **`GPU_ROUTING_TELEMETRY_PATH` CWD default.** Stage E of the cleanup
  makes this configurable via `ModelConfig`; freezing `src/model/*` is
  blocked until that lands.
- **`build.rs` fatbin compilation.** Assumes nvcc + sm_120 targets on the
  author's box. `gpu-stub` feature covers the CI / non-CUDA case.
