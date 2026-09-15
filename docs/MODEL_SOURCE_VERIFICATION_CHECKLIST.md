# Model Source Verification Checklist

Pre-onboarding verification gate. Every candidate model must pass this
checklist before its metadata is added to runtime configs, docs, or smoke-test
plans.

---

*Documentation authored by Goose agent (deepseek-v4-pro model) for the
LLM-models-onboarding branch.*

## 1. Identity and provenance

- [ ] Official provider name verified (not a mirror or third-party redistribution).
- [ ] Canonical model ID matches provider documentation.
- [ ] Model card URL points to the official source (Hugging Face, provider docs, etc.).
- [ ] The checkpoint file's embedded metadata (`general.architecture`,
      `general.name`) matches the claimed identity.

**Evidence required:** Link to model card or provider documentation.

## 2. Architecture and packaging

- [ ] Dense or MoE classification confirmed from provider documentation.
- [ ] Active parameter count verified against official specification.
- [ ] Total parameter count verified against official specification.
- [ ] Checkpoint format (GGUF, safetensors, other) confirmed.
- [ ] Quantization format confirmed from file header metadata, not filename suffix.
- [ ] For GGUF: `general.file_type` and tensor `ggml_type` values inspected.
- [ ] For safetensors: header inspected for tensor dtypes and shapes.

**Evidence required:** GGUF probe output or safetensors manifest summary.

## 3. Legal and access constraints

- [ ] License identified and compatible with repository usage terms.
- [ ] Access restrictions documented (gated, open, research-only, etc.).
- [ ] No redistribution restrictions that would conflict with the repository
      license.
- [ ] Attribution requirements noted if applicable.

**Evidence required:** License field from model card or provider documentation.

## 4. Runtime fit

- [ ] If local GGUF: model family recognized by `infer_family()` in
      `src/moe/adapter.rs`, or a new family variant has been added.
- [ ] If local GGUF: routing tensor exists and is accessible via the checkpoint
      reader.
- [ ] If local safetensors: manifest inspection succeeds without errors.
- [ ] If cloud: provider format is a recognized value (`nvcf-nim`,
      `openai-compat`, `vertex-ai`, `watsonx-saas`, `fp8-safetensors`).
- [ ] If cloud: required env var names are documented (no values stored).
- [ ] VRAM / storage assumptions documented if known.

**Evidence required:** `cargo check` clean, manifest or probe output.

## 5. Documentation

- [ ] Model added to the appropriate lineup config:
  - Local GGUF → `configs/local_gguf_lineup.toml` (gitignored; copy from
    `configs/local_gguf_lineup.template.toml`)
  - Local safetensors → `configs/local_safetensors_lineup.template.toml` copied locally to `configs/safetensors_lineup.toml`
  - Cloud → `configs/saaq_cloud_lineup.toml`
- [ ] Slug follows directory-safe naming convention.
- [ ] Family slug matches the GGUF architecture or the closest known family.
- [ ] `docs/model_lineup.md` updated with the new entry.
- [ ] `docs/CLOUD_MODELS.md` updated if cloud candidate.
- [ ] `docs/RUN_PROFILES.md` updated if a new run profile is added.

## 6. Lightweight validation

- [ ] `cargo check --no-default-features` passes.
- [ ] `cargo test --no-default-features` passes.
- [ ] For local GGUF: `just synapse-diag` succeeds (needs a CUDA toolchain;
      `synapse_diagnostic` is `required-features = ["cuda"]` and takes no
      positional argument -- point it at a checkpoint with `CHECKPOINT_PATH`
      or `LINEUP_CONFIG`).
- [ ] For local safetensors: `cargo run --example safetensors_manifest --no-default-features -- <path> artifacts/safetensors_manifest.json` succeeds.
- [ ] For cloud: `cargo test --no-default-features cloud_lineup` passes
      (`cloud_lineup_parses_valid_toml`,
      `cloud_lineup_unknown_family_reported_on_stderr`). The shipped
      `configs/saaq_cloud_lineup.toml` declares no `required_env_vars`, so
      there is currently no fail-fast path to observe -- see GH#172 follow-up.

## Quick reference

```bash
# GGUF architecture probe
python3 -c "
import struct
with open('<path>', 'rb') as f:
    h = f.read(262144)
    off = 4 + 4 + 8 + 8  # magic + version + n_tensors + n_metadata
    for _ in range(struct.unpack_from('<Q', h, 4+4+8)[0]):
        kl = struct.unpack_from('<Q', h, off)[0]; off += 8
        k = h[off:off+kl].decode(); off += kl
        vt = struct.unpack_from('<I', h, off)[0]; off += 4
        if vt == 8:
            vl = struct.unpack_from('<Q', h, off)[0]; off += 8
            v = h[off:off+vl].decode(); off += vl
        else: off += 4
        if 'arch' in k.lower(): print(f'{k} = {v}')
"

# Safetensors manifest
cargo run --example safetensors_manifest --no-default-features -- \
  /path/to/checkpoint.safetensors artifacts/safetensors_manifest.json

# Cloud lineup parse + diagnostics check (CPU, no CUDA toolchain needed).
# saaq_latent_calibration is required-features = ["cuda"], so it cannot be
# run under --no-default-features; the lineup logic is covered by tests.
cargo test --no-default-features cloud_lineup
```