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

1. **Router-family matrix green.** Every family listed in
   `docs/RUN_PROFILES.md#model-discovery` (Olmoe, Qwen3Moe, Gemma4,
   DeepSeek2, LlamaMoe) completes a SAAQ latent calibration run with
   `validation_status: "completed"` in its `run_manifest.json`.
2. **Dual-SAAQ parity.** The run emits both `saaq_delta_q_*_v1_0` and
   `saaq_delta_q_*_v1_5` columns, and the two rules diverge in the
   expected regime (1.5 < 1.0 on large deltas by the sqrt-rate rule).
3. **GPU determinism check.** A `REPEAT_COUNT=2` run produces bit-matching
   `tick_telemetry.txt` between repeat 0 and repeat 1. Non-determinism is
   a blocker.
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
