// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Public data types for `corinth-canal`.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// Dimensionality of the dense embedding the projector hands to Router.
pub const EMBEDDING_DIM: usize = 2048;

/// Supported GGUF model families for the router bridge.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum ModelFamily {
    #[default]
    Olmoe,
    Qwen3Moe,
    Gemma4,
    DeepSeek2,
    LlamaMoe,
    Moonlight16BA3B,
    Granite31A800M,
    Nemotron,
    #[serde(alias = "Nemotron3Nano4B")]
    NemotronLegacy,
    Lfm2Moe,
    SlimMoe,
    Zaya,
    Glm4,
    GptOss,
    Step,
    MiniMax,
    Cohere,
    Grin,
    Skyworks,
    Trinity,
    Grok,
}

impl ModelFamily {
    pub fn slug(self) -> &'static str {
        match self {
            Self::Olmoe => "olmoe",
            Self::Qwen3Moe => "qwen3_moe",
            Self::Gemma4 => "gemma4",
            Self::DeepSeek2 => "deepseek2",
            Self::LlamaMoe => "llama_moe",
            Self::Moonlight16BA3B => "moonlight_16b_a3b",
            Self::Granite31A800M => "granite_3_1_a800m",
            Self::Nemotron => "nemotron",
            Self::NemotronLegacy => "nemotron",
            Self::Lfm2Moe => "lfm2_moe",
            Self::SlimMoe => "slim_moe",
            Self::Zaya => "zaya",
            Self::Glm4 => "glm4",
            Self::GptOss => "gpt_oss",
            Self::Step => "step",
            Self::MiniMax => "minimax",
            Self::Cohere => "cohere",
            Self::Grin => "grin",
            Self::Skyworks => "skyworks",
            Self::Trinity => "trinity",
            Self::Grok => "grok",
        }
    }

    /// Canonical spelling table for human-authored `ModelFamily` values.
    ///
    /// This is the single source of truth for both the `MODEL_FAMILY`
    /// environment variable and lineup-config `family` entries, which
    /// previously had separate hand-rolled tables that drifted apart. Input
    /// is trimmed and matched case-insensitively.
    ///
    /// Returns `None` for unrecognised values so callers can treat that as a
    /// soft validation error (leave family for probe inference, or keep the
    /// configured default) rather than aborting a sweep.
    ///
    /// **`NemotronLegacy`:** not selected by any human alias. Operator and
    /// lineup strings `nemotron` / `nemotron_3_nano_4b` / `nemotron3nano4b`
    /// resolve to [`Self::Nemotron`]. `NemotronLegacy` remains a serde-only
    /// variant for historical checkpoint metadata (`Nemotron3Nano4B` alias)
    /// and still reports slug `"nemotron"` via [`Self::slug`].
    pub fn from_alias(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "olmoe" => Some(Self::Olmoe),
            "qwen3moe" | "qwen3_moe" | "qwen" => Some(Self::Qwen3Moe),
            "gemma4" | "gemma_4" | "gemma" => Some(Self::Gemma4),
            "deepseek2" | "deepseek_v2" | "deepseek" => Some(Self::DeepSeek2),
            "llama" | "llama_moe" | "llama3_moe" => Some(Self::LlamaMoe),
            "moonlight" | "moonlight_moe" | "moonlight_16b_a3b" => Some(Self::Moonlight16BA3B),
            "granite" | "granite_3_1" | "granite_3_1_a800m" | "granite31a800m" => {
                Some(Self::Granite31A800M)
            }
            "nemotron" | "nemotron_3_nano_4b" | "nemotron3nano4b" => Some(Self::Nemotron),
            "lfm2" | "lfm2_moe" | "lfm2moe" => Some(Self::Lfm2Moe),
            "slim_moe" | "slimmoe" | "phi_moe" | "phimoe" => Some(Self::SlimMoe),
            "zaya" | "zaya1" | "zaya1_8b" => Some(Self::Zaya),
            "glm4" | "glm_4" | "glm4moe" | "glm" => Some(Self::Glm4),
            "gpt_oss" | "gptoss" => Some(Self::GptOss),
            "step" | "step3" | "step_3_5" => Some(Self::Step),
            "minimax" | "minimax_m2" => Some(Self::MiniMax),
            "cohere" | "command_a" => Some(Self::Cohere),
            "grin" | "grin_moe" => Some(Self::Grin),
            "skyworks" | "skywork" => Some(Self::Skyworks),
            "trinity" | "trinity_nano" => Some(Self::Trinity),
            "grok" | "grok_1" | "grok_2" => Some(Self::Grok),
            _ => None,
        }
    }
}

/// Minimal local telemetry payload used to seed deterministic spike patterns.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TelemetrySnapshot {
    pub gpu_temp_c: f32,
    pub gpu_power_w: f32,
    pub cpu_tctl_c: f32,
    pub cpu_package_power_w: f32,
    pub timestamp_ms: u64,
}

impl TelemetrySnapshot {
    pub fn thermal_stress(&self) -> f32 {
        ((self.gpu_temp_c - 60.0) / 30.0).clamp(0.0, 1.0)
    }
}

/// Supported checkpoint formats for model loading.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum CheckpointFormat {
    #[default]
    Gguf,
    Safetensors,
}

impl CheckpointFormat {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Gguf => "gguf",
            Self::Safetensors => "safetensors",
        }
    }
}

/// Infer [`CheckpointFormat`] from a checkpoint path without touching the
/// filesystem.
///
/// This is the CPU-reachable counterpart of the GGUF vs Safetensors branch
/// in `Router::probe_and_map`. That branch uses `Path::is_dir()`, which a
/// unit test cannot exercise without a real checkpoint on disk. The
/// heuristics here match how the two lineups actually name checkpoints:
///
/// * `.gguf` (any case) → [`CheckpointFormat::Gguf`]. GGUF lineups always
///   point at a single file.
/// * `.safetensors`, `*.safetensors.index.json`, a HF shard such as
///   `model-00001-of-00003.safetensors`, or a path with no GGUF extension
///   (HF model directories) → [`CheckpointFormat::Safetensors`].
/// * Empty / whitespace-only → [`CheckpointFormat::Gguf`], matching
///   [`ModelConfig::default`] used by the synthetic stub path.
///
/// Trailing separators are stripped so a directory path still classifies
/// the same way with or without a final `/`.
pub fn checkpoint_format_for_path(path: &str) -> CheckpointFormat {
    let trimmed = path.trim().trim_end_matches(['/', '\\']);
    if trimmed.is_empty() {
        return CheckpointFormat::Gguf;
    }
    let file_name = Path::new(trimmed)
        .file_name()
        .and_then(|name| name.to_str())
        .or_else(|| trimmed.rsplit(['/', '\\']).find(|part| !part.is_empty()))
        .unwrap_or(trimmed);
    if Path::new(file_name)
        .extension()
        .is_some_and(|ext| ext.eq_ignore_ascii_case("gguf"))
    {
        CheckpointFormat::Gguf
    } else {
        CheckpointFormat::Safetensors
    }
}

/// Top-level configuration for the hybrid quantization pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub checkpoint_path: String,
    /// Stamp written to `run_manifest.json`. Derived from
    /// [`checkpoint_format_for_path`] at config construction; the loader
    /// inspects the path independently and does not read this field.
    pub checkpoint_format: CheckpointFormat,
    pub model_family: Option<ModelFamily>,
    pub gpu_synapse_tensor_name: String,
    pub num_experts: usize,
    pub top_k_experts: usize,
    pub routing_mode: RoutingMode,
    pub snn_steps: usize,
    pub projection_mode: ProjectionMode,
    /// Destination path for the GPU routing telemetry CSV written by
    /// `Model::forward_gpu_temporal` (and `Model::forward` on the GPU path).
    /// When `None`, the runtime falls back to the legacy CWD-relative
    /// filename `snn_gpu_routing_telemetry.csv`. Prefer an absolute path
    /// anchored in the caller's per-run artifact directory.
    #[serde(default)]
    pub gpu_routing_telemetry_path: Option<PathBuf>,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            checkpoint_path: String::new(),
            checkpoint_format: CheckpointFormat::Gguf,
            model_family: None,
            gpu_synapse_tensor_name: String::new(),
            num_experts: 8,
            top_k_experts: 1,
            routing_mode: RoutingMode::SpikingSim,
            snn_steps: 20,
            projection_mode: ProjectionMode::SpikingTernary,
            gpu_routing_telemetry_path: None,
        }
    }
}

/// Strategy used to convert spike activity into a Router embedding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum ProjectionMode {
    RateSum,
    TemporalHistogram,
    MembraneSnapshot,
    #[default]
    SpikingTernary,
}

impl ProjectionMode {
    /// Canonical spelling table for human-authored `ProjectionMode` values.
    ///
    /// This is the single source of truth for the `PROJECTION_MODE`
    /// environment variable. Input is trimmed and matched case-insensitively.
    /// PascalCase variant names (`RateSum`) are accepted because that is
    /// the spelling operators used when the env var was first ignored
    /// (GH#162, 2026-08-22).
    ///
    /// Returns `None` for unrecognised values so callers can fail fast
    /// instead of silently falling back to [`Self::SpikingTernary`].
    pub fn from_alias(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "ratesum" | "rate_sum" | "rate-sum" => Some(Self::RateSum),
            "temporalhistogram" | "temporal_histogram" | "temporal-histogram" => {
                Some(Self::TemporalHistogram)
            }
            "membranesnapshot" | "membrane_snapshot" | "membrane-snapshot" => {
                Some(Self::MembraneSnapshot)
            }
            "spikingternary" | "spiking_ternary" | "spiking-ternary" => Some(Self::SpikingTernary),
            _ => None,
        }
    }

    /// Stable stamp used in `run_manifest.json` / `summary.json`.
    pub fn as_label(self) -> &'static str {
        match self {
            Self::RateSum => "rate_sum",
            Self::TemporalHistogram => "temporal_histogram",
            Self::MembraneSnapshot => "membrane_snapshot",
            Self::SpikingTernary => "spiking_ternary",
        }
    }

    /// Parse an optional `PROJECTION_MODE` env value.
    ///
    /// * `None` or whitespace-only → `Ok(None)` so callers keep the
    ///   [`Self::SpikingTernary`] default.
    /// * Recognised alias → `Ok(Some(mode))`.
    /// * Anything else → `Err`, so the operator path can fail fast instead
    ///   of silently falling back (the 2026-08-22 `PROJECTION_MODE=RateSum`
    ///   miss happened because the env var was ignored entirely).
    pub fn parse_env_override(value: Option<&str>) -> Result<Option<Self>, String> {
        let Some(raw) = value else {
            return Ok(None);
        };
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            return Ok(None);
        }
        Self::from_alias(trimmed).map(Some).ok_or_else(|| {
            format!(
                "PROJECTION_MODE={raw:?} is not a recognised projection mode \
                 (expected RateSum|rate_sum|rate-sum|\
                 TemporalHistogram|temporal_histogram|temporal-histogram|\
                 MembraneSnapshot|membrane_snapshot|membrane-snapshot|\
                 SpikingTernary|spiking_ternary|spiking-ternary)"
            )
        })
    }

    /// Fail-fast wrapper over a raw [`std::env::var`] lookup.
    ///
    /// `NotPresent` keeps the unset semantics (`Ok(None)`), but `NotUnicode`
    /// is reported as an error rather than flattened into "unset". Collapsing
    /// it with `.ok()` would stamp the `SpikingTernary` default onto artifacts
    /// for a malformed value — the exact silent-fallback failure GH#162 is
    /// about.
    pub fn parse_env_result(
        value: Result<String, std::env::VarError>,
    ) -> Result<Option<Self>, String> {
        match value {
            Ok(raw) => Self::parse_env_override(Some(raw.as_str())),
            Err(std::env::VarError::NotPresent) => Ok(None),
            Err(std::env::VarError::NotUnicode(raw)) => Err(format!(
                "PROJECTION_MODE={raw:?} is not valid Unicode; refusing to fall back to \
                 the SpikingTernary default (GH#162)"
            )),
        }
    }

    /// Effective mode with operator precedence used by calibration runs:
    /// `env_override` (`PROJECTION_MODE`) > `lineup` > `default`
    /// (`ModelConfig` / `default_spiking_model_config`, which is
    /// [`Self::SpikingTernary`]).
    pub fn resolve(env_override: Option<Self>, lineup: Option<Self>, default: Self) -> Self {
        env_override.or(lineup).unwrap_or(default)
    }
}

/// Execution mode used by the Router router.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum RoutingMode {
    StubUniform,
    DenseSim,
    #[default]
    SpikingSim,
}

impl RoutingMode {
    /// Canonical spelling table for human-authored `RoutingMode` values.
    ///
    /// This is the single source of truth for both the `ROUTING_MODE`
    /// environment variable and lineup-config `routing_mode` entries, which
    /// previously had separate hand-rolled tables that drifted apart. Input
    /// is trimmed and matched case-insensitively.
    ///
    /// Returns `None` for unrecognised values so callers can treat that as a
    /// soft validation error and keep their configured default, rather than
    /// aborting a sweep.
    pub fn from_alias(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "dense" | "dense_sim" => Some(Self::DenseSim),
            "stub" | "stub_uniform" => Some(Self::StubUniform),
            "spiking" | "spiking_sim" => Some(Self::SpikingSim),
            _ => None,
        }
    }

    /// Effective mode with operator precedence used by calibration runs:
    /// `env_override` (`ROUTING_MODE`) > `lineup` (per-model TOML) > `default`
    /// (`ModelConfig` / `default_spiking_model_config`).
    pub fn resolve(env_override: Option<Self>, lineup: Option<Self>, default: Self) -> Self {
        env_override.or(lineup).unwrap_or(default)
    }
}

/// Output of one `Model::forward` pass.
#[derive(Debug, Clone)]
pub struct ModelOutput {
    pub spike_train: Vec<Vec<usize>>,
    pub firing_rates: Vec<f32>,
    pub membrane_potentials: Vec<f32>,
    pub embedding: Vec<f32>,
    pub expert_weights: Option<Vec<f32>>,
    pub selected_experts: Option<Vec<usize>>,
    pub reasoning: Option<String>,
}

// ── Cloud model metadata ──────────────────────────────────────────────────

/// Execution target for a model in the SAAQ lineup.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelTarget {
    Local,
    Cloud,
}

/// Model architecture classification for lineup metadata.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelArchitectureClass {
    Dense,
    Moe,
}

/// Metadata stub for a cloud-hosted model that cannot be executed locally.
///
/// Cloud models delegate execution to Dioscuri-Cloud. corinth-canal is
/// responsible for recording the candidate in experiment manifests and
/// fail-fast behaviour when the required cloud provider env vars are unset.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CloudModelSpec {
    /// Directory-safe identifier used in artifact paths.
    pub slug: String,
    /// Model family for routing tensor selection.
    pub family: Option<ModelFamily>,
    /// Provider / model ID (e.g. `nvidia/nemotron-nano-4b`).
    pub cloud_model_id: String,
    /// Canonical source URL (model card or provider listing).
    pub source_url: String,
    /// Execution target.
    pub target: ModelTarget,
    /// Architecture class.
    pub architecture: ModelArchitectureClass,
    /// Known active parameter count (e.g. `"2.4B"`).
    pub active_params: String,
    /// Known total parameter count (e.g. `"8B"`).
    pub total_params: String,
    /// Expected provider / runtime format on the cloud side
    /// (e.g. `"nvcf-nim"`, `"openai-compat"`, `"fp8-safetensors"`).
    pub provider_format: String,
    /// Environment variable names required for cloud execution.
    /// corinth-canal checks these at startup; if any are unset, execution
    /// fails fast with a diagnostic message. Values never appear in artifacts.
    /// Empty for download-on-GPU models that need no API credentials.
    #[serde(default)]
    pub required_env_vars: Vec<String>,
}

impl CloudModelSpec {
    /// Returns `true` when every env var in `required_env_vars` is set
    /// to a non-empty string.
    pub fn cloud_provider_available(&self) -> bool {
        self.required_env_vars
            .iter()
            .all(|var| std::env::var(var).is_ok_and(|v| !v.is_empty()))
    }
}

#[cfg(test)]
mod tests {
    use super::{
        CheckpointFormat, CloudModelSpec, ModelArchitectureClass, ModelFamily, ModelTarget,
        ProjectionMode, RoutingMode, checkpoint_format_for_path,
    };

    #[test]
    fn checkpoint_format_for_path_gguf_files_stay_gguf() {
        for path in [
            "model.gguf",
            "MODEL.GGUF",
            "dir/foo.Gguf",
            "/abs/path/OLMoE-1B-7B-0125-Instruct-F16.gguf",
            r"C:\models\ZAYA1-8B-Q8_0.gguf",
            "Kimi-VL-A3B-Instruct-Q6_K.gguf/",
        ] {
            assert_eq!(
                checkpoint_format_for_path(path),
                CheckpointFormat::Gguf,
                "expected Gguf for {path:?}"
            );
            assert_eq!(
                checkpoint_format_for_path(path).as_str(),
                "gguf",
                "manifest stamp for {path:?}"
            );
        }
        // Synthetic / unset path keeps ModelConfig::default's GGUF stamp.
        assert_eq!(checkpoint_format_for_path(""), CheckpointFormat::Gguf);
        assert_eq!(checkpoint_format_for_path("   "), CheckpointFormat::Gguf);
    }

    /// Safetensors lineups point at a single file, a shard, an HF index, or
    /// a model directory. All of those must stamp `safetensors`, not inherit
    /// `ModelConfig::default`'s GGUF (GH#196 / RM-1208).
    #[test]
    fn checkpoint_format_for_path_safetensors_lineup_shapes() {
        for path in [
            ".models/safetensors/nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16/model.safetensors",
            "model-00001-of-00003.safetensors",
            "model.safetensors.index.json",
            "MODEL.SAFETENSORS",
            ".models/safetensors/lfm2-8b-a1b",
            "/absolute/path/to/.models/safetensors/allenai/OLMoE-1B-7B-0125-Instruct",
            "ibm-granite/granite-3.1-3b-a800m-base/",
            r"D:\models\trinity-nano-base",
        ] {
            assert_eq!(
                checkpoint_format_for_path(path),
                CheckpointFormat::Safetensors,
                "expected Safetensors for {path:?}"
            );
            assert_eq!(
                checkpoint_format_for_path(path).as_str(),
                "safetensors",
                "manifest stamp for {path:?}"
            );
        }
    }

    #[test]
    fn model_family_slug_covers_new_variants() {
        assert_eq!(ModelFamily::Zaya.slug(), "zaya");
        assert_eq!(ModelFamily::Glm4.slug(), "glm4");
        assert_eq!(ModelFamily::Moonlight16BA3B.slug(), "moonlight_16b_a3b");
        assert_eq!(ModelFamily::Granite31A800M.slug(), "granite_3_1_a800m");
        assert_eq!(ModelFamily::Nemotron.slug(), "nemotron");
        assert_eq!(ModelFamily::NemotronLegacy.slug(), "nemotron");
        assert_eq!(ModelFamily::Lfm2Moe.slug(), "lfm2_moe");
        assert_eq!(ModelFamily::SlimMoe.slug(), "slim_moe");
        assert_eq!(ModelFamily::GptOss.slug(), "gpt_oss");
        assert_eq!(ModelFamily::Step.slug(), "step");
        assert_eq!(ModelFamily::MiniMax.slug(), "minimax");
        assert_eq!(ModelFamily::Cohere.slug(), "cohere");
        assert_eq!(ModelFamily::Grin.slug(), "grin");
        assert_eq!(ModelFamily::Skyworks.slug(), "skyworks");
        assert_eq!(ModelFamily::Trinity.slug(), "trinity");
        assert_eq!(ModelFamily::Grok.slug(), "grok");
    }

    #[test]
    fn nemotron3_nano_4b_deserializes_to_legacy() {
        let family: ModelFamily = serde_json::from_str("\"Nemotron3Nano4B\"")
            .expect("Nemotron3Nano4B should deserialize via alias");
        assert_eq!(family, ModelFamily::NemotronLegacy);
        assert_eq!(family.slug(), "nemotron");
    }

    #[test]
    fn cloud_model_spec_provider_availability_checks_env_vars() {
        let missing = "CORINTH_CANAL_TEST_MISSING_PROVIDER_VAR";

        let spec = CloudModelSpec {
            slug: "test-cloud".into(),
            family: Some(ModelFamily::Zaya),
            cloud_model_id: "provider/test-cloud".into(),
            source_url: "https://example.invalid/test-cloud".into(),
            target: ModelTarget::Cloud,
            architecture: ModelArchitectureClass::Moe,
            active_params: "1B".into(),
            total_params: "8B".into(),
            provider_format: "openai-compat".into(),
            required_env_vars: vec!["PATH".into(), missing.into()],
        };
        assert!(!spec.cloud_provider_available());

        let local_dense = CloudModelSpec {
            slug: "test-local".into(),
            family: Some(ModelFamily::Glm4),
            cloud_model_id: "provider/test-local".into(),
            source_url: "https://example.invalid/test-local".into(),
            target: ModelTarget::Local,
            architecture: ModelArchitectureClass::Dense,
            active_params: "3B".into(),
            total_params: "3B".into(),
            provider_format: "nvcf-nim".into(),
            required_env_vars: vec!["PATH".into()],
        };
        assert!(local_dense.cloud_provider_available());

        let nemotron = CloudModelSpec {
            slug: "nemotron-test".into(),
            family: Some(ModelFamily::Nemotron),
            cloud_model_id: "nvidia/nemotron-test".into(),
            source_url: "https://example.invalid/nemotron".into(),
            target: ModelTarget::Cloud,
            architecture: ModelArchitectureClass::Moe,
            active_params: "4B".into(),
            total_params: "4B".into(),
            provider_format: "nvcf-nim".into(),
            required_env_vars: vec!["PATH".into()],
        };
        assert!(nemotron.cloud_provider_available());
    }

    #[test]
    fn cloud_model_spec_deserialization_optional_required_env_vars() {
        let spec: CloudModelSpec = serde_json::from_str(
            r#"{
                "slug":"test-cloud",
                "family":"Glm4",
                "cloud_model_id":"provider/test-cloud",
                "source_url":"https://example.invalid/test-cloud",
                "target":"cloud",
                "architecture":"moe",
                "active_params":"1B",
                "total_params":"8B",
                "provider_format":"openai-compat"
            }"#,
        )
        .expect("required_env_vars should default to empty vec for download-on-GPU models");

        assert!(spec.required_env_vars.is_empty());
    }

    /// Contract for both the `ROUTING_MODE` env var and lineup-config
    /// `routing_mode` entries. These had separate hand-rolled tables that
    /// drifted: the env path accepted only `dense`/`stub`, so
    /// `ROUTING_MODE=dense_sim` was silently dropped and the config default
    /// won — a sweep could report a routing mode it had not actually run in.
    #[test]
    fn routing_mode_from_alias_accepts_every_documented_spelling() {
        for value in ["dense", "dense_sim", "DENSE_SIM", " dense_sim "] {
            assert_eq!(
                RoutingMode::from_alias(value),
                Some(RoutingMode::DenseSim),
                "expected DenseSim for {value:?}"
            );
        }
        for value in ["stub", "stub_uniform", "STUB_UNIFORM", " stub "] {
            assert_eq!(
                RoutingMode::from_alias(value),
                Some(RoutingMode::StubUniform),
                "expected StubUniform for {value:?}"
            );
        }
        for value in ["spiking", "spiking_sim", "Spiking_Sim", " spiking "] {
            assert_eq!(
                RoutingMode::from_alias(value),
                Some(RoutingMode::SpikingSim),
                "expected SpikingSim for {value:?}"
            );
        }
    }

    /// Unknown values stay `None` so callers keep their configured default
    /// instead of aborting a sweep.
    #[test]
    fn routing_mode_from_alias_rejects_unknown_values() {
        for value in ["", "densesim", "uniform", "spike", "dense-sim"] {
            assert_eq!(
                RoutingMode::from_alias(value),
                None,
                "expected None for {value:?}"
            );
        }
    }

    /// Contract for the `PROJECTION_MODE` env var. PascalCase variant names
    /// (`RateSum`) must parse: that is the spelling operators used when the
    /// env var was ignored (GH#162).
    #[test]
    fn projection_mode_from_alias_accepts_every_documented_spelling() {
        for value in ["RateSum", "rate_sum", "rate-sum", "RATESUM", " ratesum "] {
            assert_eq!(
                ProjectionMode::from_alias(value),
                Some(ProjectionMode::RateSum),
                "expected RateSum for {value:?}"
            );
        }
        for value in [
            "TemporalHistogram",
            "temporal_histogram",
            "temporal-histogram",
            "TEMPORALHISTOGRAM",
            " temporal_histogram ",
        ] {
            assert_eq!(
                ProjectionMode::from_alias(value),
                Some(ProjectionMode::TemporalHistogram),
                "expected TemporalHistogram for {value:?}"
            );
        }
        for value in [
            "MembraneSnapshot",
            "membrane_snapshot",
            "membrane-snapshot",
            "MEMBRANESNAPSHOT",
            " membrane_snapshot ",
        ] {
            assert_eq!(
                ProjectionMode::from_alias(value),
                Some(ProjectionMode::MembraneSnapshot),
                "expected MembraneSnapshot for {value:?}"
            );
        }
        for value in [
            "SpikingTernary",
            "spiking_ternary",
            "spiking-ternary",
            "SPIKINGTERNARY",
            " spiking_ternary ",
        ] {
            assert_eq!(
                ProjectionMode::from_alias(value),
                Some(ProjectionMode::SpikingTernary),
                "expected SpikingTernary for {value:?}"
            );
        }
    }

    #[test]
    fn projection_mode_default_is_spiking_ternary() {
        assert_eq!(ProjectionMode::default(), ProjectionMode::SpikingTernary);
        assert_eq!(ProjectionMode::parse_env_override(None).unwrap(), None);
        assert_eq!(ProjectionMode::parse_env_override(Some("")).unwrap(), None);
        assert_eq!(
            ProjectionMode::parse_env_override(Some("   ")).unwrap(),
            None
        );
        assert_eq!(
            ProjectionMode::resolve(None, None, ProjectionMode::SpikingTernary),
            ProjectionMode::SpikingTernary,
            "unset env and lineup must keep the SpikingTernary default"
        );
    }

    /// Unrecognised values stay `None` / `Err` so the operator path can
    /// fail fast instead of silently falling back to `SpikingTernary`.
    #[test]
    fn projection_mode_invalid_value_fails_fast() {
        for value in ["not_a_mode", "spike", "ternary", "Rate", "dense", "Spiking"] {
            assert_eq!(
                ProjectionMode::from_alias(value),
                None,
                "expected None for {value:?}"
            );
            let err = ProjectionMode::parse_env_override(Some(value))
                .expect_err("invalid PROJECTION_MODE must fail fast");
            assert!(
                err.contains("not a recognised projection mode"),
                "error should name the env var contract, got {err:?}"
            );
        }
        assert_eq!(
            ProjectionMode::parse_env_override(Some("RateSum")).unwrap(),
            Some(ProjectionMode::RateSum)
        );
    }

    /// `NotPresent` means unset, but a non-Unicode `PROJECTION_MODE` must not
    /// be flattened into the unset path — that would stamp `SpikingTernary`
    /// onto artifacts for a value the operator did set.
    #[test]
    fn projection_mode_env_result_separates_unset_from_malformed() {
        assert_eq!(
            ProjectionMode::parse_env_result(Err(std::env::VarError::NotPresent)).unwrap(),
            None
        );
        assert_eq!(
            ProjectionMode::parse_env_result(Ok("rate-sum".to_string())).unwrap(),
            Some(ProjectionMode::RateSum)
        );
        assert!(
            ProjectionMode::parse_env_result(Ok("not_a_mode".to_string())).is_err(),
            "unrecognised values must still fail fast"
        );
    }

    #[cfg(unix)]
    #[test]
    fn projection_mode_env_result_rejects_non_unicode() {
        use std::os::unix::ffi::OsStringExt;

        let raw = std::ffi::OsString::from_vec(vec![0x52, 0x61, 0x74, 0x65, 0x80]);
        let err = ProjectionMode::parse_env_result(Err(std::env::VarError::NotUnicode(raw)))
            .expect_err("non-Unicode PROJECTION_MODE must fail fast");
        assert!(
            err.contains("not valid Unicode"),
            "error should name the Unicode contract, got {err:?}"
        );
    }

    #[test]
    fn projection_mode_label_round_trips_through_from_alias() {
        for mode in [
            ProjectionMode::RateSum,
            ProjectionMode::TemporalHistogram,
            ProjectionMode::MembraneSnapshot,
            ProjectionMode::SpikingTernary,
        ] {
            assert_eq!(
                ProjectionMode::from_alias(mode.as_label()),
                Some(mode),
                "stamp {:?} must round-trip for {mode:?}",
                mode.as_label()
            );
        }
    }

    #[test]
    fn projection_mode_resolve_precedence_env_over_lineup_over_default() {
        assert_eq!(
            ProjectionMode::resolve(
                Some(ProjectionMode::RateSum),
                Some(ProjectionMode::MembraneSnapshot),
                ProjectionMode::SpikingTernary,
            ),
            ProjectionMode::RateSum,
            "env override must win over lineup"
        );
        assert_eq!(
            ProjectionMode::resolve(
                None,
                Some(ProjectionMode::TemporalHistogram),
                ProjectionMode::SpikingTernary
            ),
            ProjectionMode::TemporalHistogram,
            "lineup must win when env is unset"
        );
        assert_eq!(
            ProjectionMode::resolve(None, None, ProjectionMode::SpikingTernary),
            ProjectionMode::SpikingTernary,
            "default when both optional sources are absent"
        );
    }

    #[test]
    fn routing_mode_resolve_precedence_env_over_lineup_over_default() {
        assert_eq!(
            RoutingMode::resolve(
                Some(RoutingMode::DenseSim),
                Some(RoutingMode::StubUniform),
                RoutingMode::SpikingSim,
            ),
            RoutingMode::DenseSim,
            "env override must win over lineup"
        );
        assert_eq!(
            RoutingMode::resolve(
                None,
                Some(RoutingMode::StubUniform),
                RoutingMode::SpikingSim
            ),
            RoutingMode::StubUniform,
            "lineup must win when env is unset"
        );
        assert_eq!(
            RoutingMode::resolve(None, None, RoutingMode::SpikingSim),
            RoutingMode::SpikingSim,
            "default when both optional sources are absent"
        );
    }

    /// Contract for both the `MODEL_FAMILY` env var and lineup-config
    /// `family` entries. Tables lived in both `examples/support/mod.rs` and
    /// `lineup.rs` and could drift independently.
    #[test]
    fn model_family_from_alias_accepts_every_documented_spelling() {
        let cases: &[(&ModelFamily, &[&str])] = &[
            (&ModelFamily::Olmoe, &["olmoe", "OLMOE", " olmoe "]),
            (
                &ModelFamily::Qwen3Moe,
                &["qwen3moe", "qwen3_moe", "qwen", " QWEN "],
            ),
            (
                &ModelFamily::Gemma4,
                &["gemma4", "gemma_4", "gemma", " Gemma4 "],
            ),
            (
                &ModelFamily::DeepSeek2,
                &["deepseek2", "deepseek_v2", "deepseek", " DEEPSEEK "],
            ),
            (
                &ModelFamily::LlamaMoe,
                &["llama", "llama_moe", "llama3_moe", " LLAMA "],
            ),
            (
                &ModelFamily::Moonlight16BA3B,
                &[
                    "moonlight",
                    "moonlight_moe",
                    "moonlight_16b_a3b",
                    "MOONLIGHT",
                ],
            ),
            (
                &ModelFamily::Granite31A800M,
                &[
                    "granite",
                    "granite_3_1",
                    "granite_3_1_a800m",
                    "granite31a800m",
                    " GRANITE ",
                ],
            ),
            (
                &ModelFamily::Nemotron,
                &[
                    "nemotron",
                    "nemotron_3_nano_4b",
                    "nemotron3nano4b",
                    " NEMOTRON ",
                ],
            ),
            (
                &ModelFamily::Lfm2Moe,
                &["lfm2", "lfm2_moe", "lfm2moe", " LFM2 "],
            ),
            (
                &ModelFamily::SlimMoe,
                &["slim_moe", "slimmoe", "phi_moe", "phimoe", " SLIMMOE "],
            ),
            (&ModelFamily::Zaya, &["zaya", "zaya1", "zaya1_8b", " ZAYA "]),
            (
                &ModelFamily::Glm4,
                &["glm4", "glm_4", "glm4moe", "glm", " GLM "],
            ),
            (&ModelFamily::GptOss, &["gpt_oss", "gptoss", " GPTOSS "]),
            (&ModelFamily::Step, &["step", "step3", "step_3_5", " STEP "]),
            (
                &ModelFamily::MiniMax,
                &["minimax", "minimax_m2", " MINIMAX "],
            ),
            (&ModelFamily::Cohere, &["cohere", "command_a", " COHERE "]),
            (&ModelFamily::Grin, &["grin", "grin_moe", " GRIN "]),
            (
                &ModelFamily::Skyworks,
                &["skyworks", "skywork", " SKYWORK "],
            ),
            (
                &ModelFamily::Trinity,
                &["trinity", "trinity_nano", " TRINITY "],
            ),
            (&ModelFamily::Grok, &["grok", "grok_1", "grok_2", " GROK "]),
        ];
        for &(family, aliases) in cases {
            for value in aliases {
                assert_eq!(
                    ModelFamily::from_alias(value),
                    Some(*family),
                    "expected {family:?} for {value:?}"
                );
            }
        }
        // NemotronLegacy is serde-only; no human alias selects it.
        assert_eq!(
            ModelFamily::from_alias("nemotron"),
            Some(ModelFamily::Nemotron),
            "nemotron must not resolve to NemotronLegacy"
        );
    }

    #[test]
    fn model_family_from_alias_rejects_unknown_values() {
        for value in [
            "",
            "unknown_family",
            "moe",
            "olmo",
            "nemotron_legacy",
            "nemotronlegacy",
        ] {
            assert_eq!(
                ModelFamily::from_alias(value),
                None,
                "expected None for {value:?}"
            );
        }
    }

    /// Every human-selectable variant must round-trip through its slug, so a
    /// new variant cannot ship without a `from_alias` arm.
    #[test]
    fn model_family_slug_round_trips_through_from_alias() {
        for family in [
            ModelFamily::Olmoe,
            ModelFamily::Qwen3Moe,
            ModelFamily::Gemma4,
            ModelFamily::DeepSeek2,
            ModelFamily::LlamaMoe,
            ModelFamily::Moonlight16BA3B,
            ModelFamily::Granite31A800M,
            ModelFamily::Nemotron,
            // NemotronLegacy intentionally omitted: serde-only, slug collides.
            ModelFamily::Lfm2Moe,
            ModelFamily::SlimMoe,
            ModelFamily::Zaya,
            ModelFamily::Glm4,
            ModelFamily::GptOss,
            ModelFamily::Step,
            ModelFamily::MiniMax,
            ModelFamily::Cohere,
            ModelFamily::Grin,
            ModelFamily::Skyworks,
            ModelFamily::Trinity,
            ModelFamily::Grok,
        ] {
            assert_eq!(
                ModelFamily::from_alias(family.slug()),
                Some(family),
                "slug {:?} must round-trip for {family:?}",
                family.slug()
            );
        }
    }
}
