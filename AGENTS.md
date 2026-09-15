# AGENTS.md

## Project identity

`corinth-canal` is the single-crate reference implementation for the `rmems`
SNN-logic quantization line.

The core pipeline is:

TelemetrySnapshot
-> TelemetryEncoder / GPU snapshot projection
-> ternary spikes
-> SignedSplitBankBridge or GPU input_spikes
-> SparseGifHiddenLayer / GPU GIF temporal loop
-> Projector
-> Router
-> routing telemetry / SAAQ latent calibration

This repository is intentionally not split into modular crates yet. Do not
extract modules into separate crates unless explicitly asked. Proven components
graduate later according to `docs/PROMOTION_RULES.md` and
`docs/MODULE_STATUS.md`.

## Agent and executor tooling

This repository is agent-agnostic. Which assistant, IDE, or CLI you drive it
with is a personal choice, not repository policy, so no inventory of approved
executors is kept here. Such a list dates quickly, and it publishes the
maintainer's tooling and subscription arrangements without giving a contributor
anything they can act on.

What applies regardless of what you use:

- Do not commit or document secrets, tokens, DSNs, API keys, private telemetry,
  or local absolute paths.
- Do not assume optional third-party APIs are configured. Local and offline
  behaviour must stay correct when every external integration is unset.
- Keep tooling documentation high level. Machine-specific setup — install
  locations, subscription tiers, per-workstation permission grants — belongs in
  local, untracked configuration, not in tracked docs.
- Keep tooling changes limited to markdown and closely related repo docs unless
  a task explicitly asks for implementation work.

`CLAUDE.md` is the exception worth naming: it carries repository context
(architecture, build worlds, invariants) rather than a tooling roster, which is
why it is tracked.

## Non-negotiable rules

- Keep this as a single Rust crate unless the task explicitly says otherwise.
- Do not delete or bypass the SNN/GIF/SAAQ routing logic.
- Do not replace real GGUF-backed routing with stubs except in tests or explicit
  fallback paths.
- Do not introduce machine-local absolute paths such as `/home/...` into `src/`.
- Do not hardcode checkpoint paths, telemetry CSV paths, CUDA paths, or artifact
  output paths.
- Do not add new dependencies unless they are clearly justified.
- Preserve CPU-only buildability.
- Preserve CUDA/GPU behavior when touching GPU code.
- Keep diffs small and reviewable.
- Add or update tests when behavior changes.
- Do not change CSV schemas unless explicitly instructed.
- Do not silently change public APIs exported from `src/lib.rs`.
- Do not touch generated artifacts unless the task is specifically about run
  manifests, validation outputs, or known-good run logs.

## Repository structure

Important paths:

- `src/model/core.rs`
  - Runtime orchestration, config validation, forward paths.
- `src/model/temporal.rs`
  - GPU temporal loop:
    `prepare_gpu_temporal`, `tick_gpu_temporal`, `forward_gpu_temporal`.
- `src/model/telemetry_io.rs`
  - Routing telemetry CSV helpers.
- `src/moe/checkpoint.rs`
  - GGUF parsing, mmap access, tensor slicing, metadata.
- `src/moe/adapter.rs`
  - Model-family adapter resolution.
- `src/moe/routing.rs`
  - Router math, gate scores, expert selection.
- `src/projector.rs`
  - Spike-to-embedding projection.
- `src/funnel.rs`
  - Telemetry funnel and GIF hidden layer.
- `src/telemetry.rs`
  - `TelemetryEncoder` and `TelemetrySnapshot`.
- `src/latent.rs`
  - SAAQ latent calibration and CSV export.
- `src/gpu/`
  - CUDA/cust wrapper layer and GPU kernels.
- `src/gpu/kernels/`
  - CUDA `.cu` / `.cuh` sources.
- `examples/support/config.rs`
  - Environment/config resolution for examples.
- `examples/support/observability.rs`
  - Shared tracing/Sentry wrapper for example binaries.
- `docs/ARCHITECTURE.md`
  - Architecture, hidden control flow, path behavior.
- `docs/RUN_PROFILES.md`
  - Validated run profiles.
- `docs/PROMOTION_RULES.md`
  - Rules for graduating modules into `rmems-*` crates.
- `docs/MODULE_STATUS.md`
  - Current status of each module.
- `manifests/proven_components.toml`
  - Machine-readable promotion status.
- `manifests/known_good_runs.md`
  - Blessed run ID log.

## Build and validation commands

Prefer the checked-in `justfile` recipes.

Basic commands:

```bash
just setup
cargo check --all-targets --no-default-features
cargo test --no-default-features
```

On CUDA-equipped setups with `nvcc` available, `just check` and `just test`
exercise the default feature set.

### Cross-platform and security CI

- GitHub Actions: primary (see `.github/workflows/*.yml` for CPU, GPU, Docker, Sentry).
- Snyk: security scans (SCA for open-source deps, Code for SAST, etc.) are performed via the Snyk MCP server tools (e.g. `snyk__snyk_sca_scan`, `snyk__snyk_code_scan`, `snyk__snyk_container_scan`, `snyk__snyk_iac_scan`) instead of a dedicated CI workflow. Use during development, reviews, or agent-assisted tasks. GH#108. Aikido is tracked separately (see #70).

Run `cargo check --no-default-features && cargo test --no-default-features` locally before substantial Rust changes (per Git Workflow below). Docs-only, license-only, CI-only, or similar non-behavioral/non-Rust edits do not require full cargo validation (use targeted checks or skip as escape hatch). GHA provides additional platform coverage. Snyk MCP provides on-demand security scanning.

## Entry Order

- Start at `src/lib.rs` for the exported crate surface.
- Read `src/model/mod.rs` before descending into `core.rs` and `temporal.rs`.
- Read `src/moe/mod.rs` before descending into `adapter.rs`, `checkpoint.rs`, and `routing.rs`.
- Use `examples/saaq_latent_calibration.rs` as the research loop entrypoint.

## Workflow Policy

- All `corinth-canal` work stays in the repository root.
- Do not create or use additional `corinth-canal` worktrees outside the main repo.
- Do not edit files outside this repository, and do not add dependencies, unless
  the task explicitly asks for it. This previously read "limited to approved
  research tools", which depended on an inventory this document no longer keeps;
  the constraint was always about scope, not about which tool you drive.

## Git Workflow

- Prefer `git` commands over MCP tools for branch and PR operations in this environment.
- Keep behavioral changes separate from structural refactors when possible.
- Prefer basing PRs on `main`; avoid opening a PR whose base is another feature branch. When work
  is stacked, retarget child PRs to `main` before merging the parent. GH#126 was based on
  `refactor/gh118-unify-infer-family`; that branch was squash-merged via GH#125 and deleted, so
  GitHub marked GH#126 "merged" while none of its code reached `main` — the refactor was silently
  lost and had to be reconstructed. After squash-merging any stack, verify the child landed:
  `git grep <new-symbol> origin/main -- <file>`.
- Run `cargo check --no-default-features` and `cargo test --no-default-features` before closing substantial Rust changes. Docs-only, license-only, or similar non-Rust edits do not require cargo validation.
- On CUDA-equipped setups with `nvcc` available, also run the default-feature path and `cargo build --examples`.
- Snyk scans via MCP tools (see Cross-platform and security CI section) for on-demand use in development/agent flows (GH#108).

## Repository Context

- **Repo**: `rmems/corinth-canal`
- **Main branch**: `main`
- **Language**: Rust (edition 2024)
- **Key concepts**: SNN-LLM hybrid, GGUF checkpoints, MoE routing, GPU temporal simulation, SAAQ validation

<!-- BEGIN BEADS INTEGRATION v:1 profile:minimal hash:7510c1e2 -->
## Beads Issue Tracker

This project uses **bd (beads)** for issue tracking. Run `bd prime` to see full workflow context and commands.

### Quick Reference

```bash
bd ready              # Find available work
bd show <id>          # View issue details
bd update <id> --claim  # Claim work
bd close <id>         # Complete work
```

### Rules

- Use `bd` for ALL task tracking — do NOT use TodoWrite, TaskCreate, or markdown TODO lists
- Run `bd prime` for detailed command reference and session close protocol
- Use `bd remember` for persistent knowledge — do NOT use MEMORY.md files

**Architecture in one line:** issues live in a local Dolt DB; sync uses `refs/dolt/data` on your git remote; `.beads/issues.jsonl` is a passive export. See https://github.com/gastownhall/beads/blob/main/docs/SYNC_CONCEPTS.md for details and anti-patterns.

## Session Completion

**When ending a work session**, you MUST complete ALL steps below. Work is NOT complete until `git push` succeeds.

**MANDATORY WORKFLOW:**

1. **File issues for remaining work** - Create issues for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update issue status** - Close finished work, update in-progress items
4. **PUSH TO REMOTE** - This is MANDATORY:
   ```bash
   git pull --rebase
   git push
   git status  # MUST show "up to date with origin"
   ```
5. **Clean up** - Clear stashes, prune remote branches
6. **Verify** - All changes committed AND pushed
7. **Hand off** - Provide context for next session

**CRITICAL RULES:**
- Work is NOT complete until `git push` succeeds
- NEVER stop before pushing - that leaves work stranded locally
- NEVER say "ready to push when you are" - YOU must push
- If push fails, resolve and retry until it succeeds
<!-- END BEADS INTEGRATION -->
