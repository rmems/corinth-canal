# Promotion Rules

Rules for graduating code paths out of this reference repo into one of the
`rmems` modular crates. Status tracking lives in
`manifests/proven_components.toml`; blessed run IDs live in
`manifests/known_good_runs.md`.

## Status ladder

1. **reference** — code lives here because it is the only place it works
   end-to-end. Not yet portable.
2. **stabilizing** — API has stopped thrashing and is covered by at least
   one blessed run, but still depends on helpers that have not been
   promoted.
3. **proven** — the component has run green against the full matrix in
   `docs/RUN_PROFILES.md` and its external surface is frozen.
4. **frozen** — copied into its target `rmems` crate. Further changes
   happen there; `corinth-canal` holds an unmodified reference copy.

## Promotion criteria

A module moves from **stabilizing** to **proven** only after all of:

1. **Router-family matrix green.** Every family the module claims to support
   completes a SAAQ latent calibration run with `validation_status:
   "completed"` in its `run_manifest.json`.

   The authoritative family list is the `ModelFamily` enum in `src/types.rs`
   (21 variants at time of writing), **not** a prose list. This gate
   previously named five families, which meant a module could be certified
   "proven" against a quarter of the families the adapter actually resolves.
   A module that is only validated against a subset must say which subset in
   its `docs/MODULE_STATUS.md` note rather than silently inheriting a short
   list from here.
2. **Dual-SAAQ parity.** The run emits both `saaq_delta_q_*_v1_0` and
   `saaq_delta_q_*_v1_5` columns, and the two rules diverge in the
   expected regime (1.5 < 1.0 on large deltas by the sqrt-rate rule).
3. **GPU validation and determinism check.** The relevant CUDA path has Tier 1
   repo-native evidence from `docs/CUDA_VALIDATION.md`, and a `REPEAT_COUNT=2`
   run produces bit-matching `tick_telemetry.txt` between repeat 0 and repeat 1.
   Non-determinism is a blocker.
4. **CSV schema frozen.** No structural change to
   `latent_telemetry.csv` or the telemetry CSV input header
   (`timestamp_ms,gpu_temp_c,gpu_power_w,cpu_tctl_c,cpu_package_power_w`)
   for at least one full validated matrix sweep.
5. **No machine-local paths.** `grep -rE '/home/[^/]+' src/<module>` is
   empty. Every path must come from `ModelConfig`, `RunConfig`, or an
   explicit argument.
6. **Blessed run logged.** A run ID is appended to
   `manifests/known_good_runs.md` with checkpoint slug + telemetry source
   + SAAQ rule + a one-line conclusion.

## Freezing

After promotion, `corinth-canal` keeps the reference copy unchanged. Any
bugfix that needs to apply to both the reference and the modular crate
must be mirrored by hand — the modular crate is the source of truth once
the component is **frozen**.

## One-way extractions (no reverse dependency)

The ladder above assumes a module eventually promotes into a target crate that
`corinth-canal` may then depend on, with the modular crate becoming the source
of truth. Two module families deliberately depart from that, in opposite
directions:

- **`src/moe/safetensors/` (inspect half) → `rmems/engram-parser`, feature
  `safetensors`.** This is a **one-way copy from inspiration**, not a
  promotion. `corinth-canal` will **never** take an `engram-parser` dependency
  for safetensors: it keeps this reference copy, keeps `config`/`map`, and
  keeps using them in the Router / `CheckpointBackend` experiment paths.
  `engram-parser` likewise takes **no** dependency on `corinth-canal`. The
  module therefore stays at **reference** even after the copy lands
  downstream — "frozen" would assert a source-of-truth handoff that is not
  happening, and the two copies are expected to diverge. Tracking: #116,
  rmems/engram-parser#10.
- **`src/moe/gguf/` (parse half) → `rmems/engram-parser`, as a real
  dependency.** Here corinth *does* intend to consume the modular crate, once
  it grows an mmap-backed reader and K-quant dequant (#115, blocked by #144 /
  rmems/engram-parser#45). Dequant and CUDA host-register stay here regardless.

**Rule of thumb:** a status in `docs/MODULE_STATUS.md` describes *this repo's*
readiness to hand code off. It does not by itself imply corinth will consume
the result, and `Target crate` names the destination of the code, not a future
`Cargo.toml` entry.
