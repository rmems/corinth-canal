# Cloud Model Lineup

Cloud model execution is delegated to **Dioscuri-Cloud**. corinth-canal is
responsible for model selection, experiment metadata stamping, and fail-fast
validation — not for infrastructure provisioning, credential management, or
resource lifecycle.

---

*Documentation authored by Goose agent (deepseek-v4-pro model) for the
LLM-models-onboarding branch.*

## Configuration

Cloud model metadata lives in `configs/saaq_cloud_lineup.toml`. Helper
parsing/validation lives in `examples/support/mod.rs` and can be referenced via:

```bash
CLOUD_LINEUP_CONFIG=configs/saaq_cloud_lineup.toml
```

## Cloud model entries

| # | Slug | Model ID | Arch | Active | Provider |
|---|------|----------|------|--------|----------|
| MET-55 | `nemotron_3_nano_4b_cloud` | nvidia/nvidia-nemotron-3-nano-4B-BF16 | dense | 4B | nvcf-nim |
| MET-56 | `granite_3_1_3b_a800m_cloud` | ibm-granite/granite-3.1-3b-a800m-base | moe | 800M | watsonx-saas |
| MET-57 | `skywork_moe_base_fp8_cloud` | Skywork/Skywork-MoE-Base-FP8 | moe | 3.7B | fp8-safetensors |
| MET-58 | `trinity_nano_base_cloud` | arcee-ai/Trinity-Nano-Base | moe | 1B | nvcf-nim |
| MET-59 | `nemotron_3_nano_8b_cloud` | nvidia/NVIDIA-Nemotron-3-Nano-8B-4K-BF16 | dense | 8B | nvcf-nim |
| MET-60 | `glm46v_flash_cloud` | zai-org/GLM-4.6V-Flash | moe | 3.5B | openai-compat |
| MET-61 | `kimi_vl_a3b_cloud` | moonshotai/Kimi-VL-A3B-Instruct | moe | 2.8B | openai-compat |
| MET-62 | `gemma4_26b_a4b_cloud` | google/gemma-4-26B-A4B-it | moe | 4B | vertex-ai |
| MET-63 | `zaya1_8b_cloud` | Zyphra/ZAYA1-reasoning-base | moe | 1B | openai-compat |
| MET-64 | `marco_nano_base_cloud` | nvidia/Marco-Nano-Base | moe | 527M | nvcf-nim |

## Fail-fast behaviour

When any `required_env_vars` entry is unset or empty during helper parsing:

1. The parser emits a diagnostic to stderr listing the missing vars.
2. The candidate remains in the parsed lineup with `provider_available = false`.

Example diagnostic:

```
cloud_lineup: provider unavailable for slug=nemotron_3_nano_4b_cloud (nvidia/nvidia-nemotron-3-nano-4B-BF16): missing env vars: NEMOTRON_NIM_ENDPOINT, NEMOTRON_NIM_API_KEY
```

Main runner integration for cloud lineup metadata is intentionally separate.

## Required env vars by provider

Every cloud model entry declares the env var names it needs. Values are
never stored in corinth-canal configs or artifacts.

| Provider format | Required env vars (example names) |
|-----------------|-----------------------------------|
| `nvcf-nim` | `<PREFIX>_NIM_ENDPOINT`, `<PREFIX>_NIM_API_KEY` |
| `openai-compat` | `<PREFIX>_ENDPOINT`, `<PREFIX>_API_KEY` |
| `vertex-ai` | `VERTEX_AI_PROJECT_ID`, `VERTEX_AI_LOCATION`, `VERTEX_AI_ENDPOINT_ID` |
| `watsonx-saas` | `WATSONX_ENDPOINT`, `WATSONX_API_KEY`, `WATSONX_PROJECT_ID` |
| `fp8-safetensors` | `<PREFIX>_FP8_ENDPOINT`, `<PREFIX>_FP8_API_KEY` |

## Non-goals

corinth-canal does **not** handle:

- Terraform or IaC provisioning
- IBM Cloud, AWS, or GCP resource creation
- Cloud credential storage or rotation
- Cost ledger or billing integration
- Provider-specific managed ML runner code
- Artifact bucket setup
- Generalized cloud infrastructure abstraction

These responsibilities belong to Dioscuri-Cloud. corinth-canal only
selects, stamps, and guards.
