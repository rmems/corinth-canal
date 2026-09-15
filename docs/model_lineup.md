# Model Lineup — Rollout Batches

Durable planning document for model onboarding batches. Each batch tracks the
PR-sized onboarding issues and the metadata each issue must carry.

---

*Documentation authored by Goose agent (deepseek-v4-pro model) for the
LLM-models-onboarding branch.*

## Batch structure

Onboarding proceeds in batches. Each batch is a group of related models that
share a provider family, architecture, or deployment target.

### Batch A — Local GGUF routing targets (completed)

Models that can be loaded locally via the GGUF-backed `Router` for SAAQ
latent calibration runs.

| MET | Slug | Provider/Model | Family | Quant |
|-----|------|----------------|--------|-------|
| — | `olmoe_baseline` | allenai/OLMoE-1B-7B-0125-Instruct | Olmoe | F16 |
| — | `qwen3_moe_i1_iq3_m` | Qwen/Qwen3-MoE | Qwen3Moe | IQ3_M |
| — | `gemma4_26b_a4b_iq4_nl` | google/gemma-4-26B-A4B-it | Gemma4 | IQ4_NL |
| — | `deepseek_coder_v2_lite_q6_k_l` | deepseek-ai/DeepSeek-Coder-V2-Lite | DeepSeek2 | Q6_K_L |
| — | `llama_3_2_dark_champion_q5_k_m` | Dark-Champion/Llama-3.2-8X3B-MOE | LlamaMoe | Q5_K_M |
| MET-50 | `zaya1_8b_q8_0` | Abiray/ZAYA1-8B | Zaya | Q8_0 |
| MET-51 | `glm46v_flash_q8_0` | unsloth/GLM-4.6V-Flash | Glm4 | Q8_0 |
| MET-53 | `kimi_vl_a3b_q6_k` | ssweens/Kimi-VL-A3B-Instruct | Moonlight16BA3B | Q6_K |
| MET-54 | `marco_nano_base_q8_0` | mradermacher/Marco-Nano-Base | Qwen3Moe | Q8_0 |

Kimi-VL-A3B (MET-53) is Moonshot's Moonlight-16B-A3B language decoder, not
DeepSeek-Coder. llama.cpp GGUF packages set `general.architecture =
deepseek2` because there is no moonlight GGUF architecture; that packaging
tag is not the `ModelFamily`. `infer_family` maps Kimi/Moonlight paths to
`Moonlight16BA3B` and leaves a bare `deepseek2` arch as `DeepSeek2`.

### Batch B — Local safetensors manifest inspection

Models that exist locally as safetensors and are onboarded for header
inspection / manifest generation only. The router path remains GGUF-backed.

| MET | Slug | Provider/Model | Shards | Arch |
|-----|------|----------------|--------|------|
| MET-55 | `nemotron_3_nano_4b` | nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16 | 1 | dense |
| MET-56 | `granite_3_1_3b_a800m` | ibm-granite/granite-3.1-3b-a800m-base | 2 | moe |
| MET-58 | `trinity_nano_base` | arcee-ai/Trinity-Nano-Base | 7 | moe |
| — | `phi_tiny_moe_instruct` | microsoft/Phi-tiny-MoE-instruct | 2 | moe |
| — | `moonlight_16b_a3b_bnb_4bit` | slowfastai/Moonlight-16B-A3B-bnb-4bit | 2 | moe |

### Batch C — Cloud model metadata stubs

Cloud models whose execution is delegated to Dioscuri-Cloud. corinth-canal
carries only metadata stubs with fail-fast env var guards.

| MET | Slug | Provider/Model | Arch | Active | Provider |
|-----|------|----------------|------|--------|----------|
| MET-55 | `nemotron_3_nano_4b_cloud` | nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16 | dense | 4B | nvcf-nim |
| MET-56 | `granite_3_1_3b_a800m_cloud` | ibm-granite/granite-3.1-3b-a800m-base | moe | 800M | watsonx-saas |
| MET-57 | `skywork_moe_base_fp8_cloud` | Skywork/Skywork-MoE-Base-FP8 | moe | 3.7B | fp8-safetensors |
| MET-58 | `trinity_nano_base_cloud` | arcee-ai/Trinity-Nano-Base | moe | 1B | nvcf-nim |
| MET-59 | `nemotron_3_nano_8b_cloud` | nvidia/NVIDIA-Nemotron-3-Nano-8B-4K-BF16 | dense | 8B | nvcf-nim |
| MET-60 | `glm46v_flash_cloud` | zai-org/GLM-4.6V-Flash | moe | 3.5B | openai-compat |
| MET-61 | `kimi_vl_a3b_cloud` | moonshotai/Kimi-VL-A3B-Instruct | moe | 2.8B | openai-compat |
| MET-62 | `gemma4_26b_a4b_cloud` | google/gemma-4-26B-A4B-it | moe | 4B | vertex-ai |
| MET-63 | `zaya1_8b_cloud` | Zyphra/ZAYA1-reasoning-base | moe | 1B | openai-compat |
| MET-64 | `marco_nano_base_cloud` | nvidia/Marco-Nano-Base | moe | 527M | nvcf-nim |

## Required metadata per onboarding issue

Every onboarding issue must carry:

| Field | Description |
|-------|-------------|
| **Provider / model ID** | Canonical provider-qualified identifier |
| **Source URL** | Model card or official listing |
| **License / access** | Usage constraints and access requirements |
| **Architecture** | Dense or MoE classification |
| **Active parameters** | Known active parameter count |
| **Total parameters** | Total parameter count including shared weights |
| **Checkpoint format** | GGUF, safetensors, or other |
| **Quantization format** | F16, Q8_0, Q5_K, FP8, etc. |
| **Target** | Local or cloud |
| **Provider format** | For cloud: nvcf-nim, openai-compat, vertex-ai, etc. |
| **Required env vars** | Env var names needed for cloud execution (no values) |
| **SAAQ scope** | SAAQ version and telemetry source |

## Non-goals

- No Terraform or IaC provisioning
- No cloud credential storage
- No cost ledger implementation
- No provider-specific managed ML runners
- No real cloud resource creation