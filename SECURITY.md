# Security Policy

## Scope

`corinth-canal` is a research reference implementation, not a deployed service.
The security-relevant surface is:

- **Checkpoint parsing.** `src/moe/gguf/` and `src/moe/safetensors/` parse GGUF
  and safetensors files. These are untrusted input: the autodiscovery scan walks
  a directory of downloaded checkpoints, so a malformed or hostile file reaches
  the parser without review.
- **CUDA FFI.** `src/gpu/` passes buffers to hand-written kernels through a
  C ABI shim.
- **Telemetry egress.** Sentry and OpenTelemetry are wired into the example
  runners. Both stay fully disabled when `SENTRY_DSN` / `NR_INSERT_KEY` are
  unset — no client is constructed and no network call is made.

## Reporting a vulnerability

Report privately through GitHub's
[security advisory](https://github.com/rmems/corinth-canal/security/advisories/new)
form rather than opening a public issue.

Please include the checkpoint or input that triggers the problem where you can,
or a description of how to construct one. A crash reachable from a crafted
`.gguf` is in scope even without a demonstrated exploit.

## What is not a vulnerability

- Missing credentials causing a run to fall back to synthetic telemetry. This
  is intended and is stamped in `run_manifest.json`.
- Absolute paths in local, gitignored configuration files. Committed absolute
  paths are a bug — report those as a normal issue.
