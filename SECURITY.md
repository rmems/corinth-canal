# Security Policy

## Reporting a vulnerability

Please report security issues **privately** through GitHub Security Advisories:

https://github.com/rmems/corinth-canal/security/advisories/new

Do not open a public GitHub issue, pull request, or discussion for a
vulnerability.

This repository parses untrusted GGUF and safetensors checkpoints, walks a
local autodiscovery directory of downloaded model files in example runners, and
exposes a CUDA FFI surface. Those paths, plus CI and dependency supply-chain
issues, are in scope.

## Supported versions

Fixes land on `main`. There is no separately maintained release branch.

## What to include

- Affected surface (checkpoint parsing, CUDA FFI, example autodiscovery, CI,
  dependencies)
- A reproducer that does not require private checkpoints if possible
- Impact if known
