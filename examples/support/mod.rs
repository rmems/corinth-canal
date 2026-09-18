// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Shared helper functions for the example binaries.

pub mod config;
pub mod embedding;
pub mod lineup;
pub mod observability;
pub mod spikenaut;
pub mod synapse_diag;
pub mod telemetry_csv;

#[cfg(feature = "cuda")]
pub use config::RunConfig;
pub use embedding::{PromptEmbeddingProvider, resolve_prompt_embedding_provider};
pub use lineup::{
    cloud_execution_guard, cloud_lineup_path_from_env, load_cloud_lineup, load_safetensors_lineup,
    safetensors_lineup_path_from_env,
};
// Re-exports for example binaries; each binary only uses a subset, so the
// unused_imports lint would fire without this allow.
#[allow(unused_imports)]
pub use telemetry_csv::{
    ResolvedTelemetry, TELEMETRY_CSV_HEADER, TelemetrySource, load_csv_telemetry_rows,
    resolve_telemetry_from, synthetic_base_snapshot, telemetry_snapshot_for_tick,
};

#[cfg(feature = "cuda")]
use corinth_canal::model::ModelConfig;
use corinth_canal::{
    ModelFamily, SaaqUpdateRule, moe::Router, moe::RoutingMode, projector::ProjectionMode,
};
use std::io::Error;
use std::path::{Path, PathBuf};
use std::process::Command;

pub const DEFAULT_MATH_PROMPT_TEXT: &str = "The derivative of a constant is mathematically zero.";

pub const DEFAULT_RUST_SYNTAX_PROMPT_TEXT: &str =
    "fn main() { println!(\"Hello from a spiking MoE model.\"); }";

pub const DEFAULT_ENGLISH_SNN_PROMPT_TEXT: &str = "Let's teach this MoE model about SNN.";

pub const DEFAULT_ENGLISH_EXPLANATION_PROMPT_TEXT: &str =
    "Explain to me how Mixture of Experts models work";

pub const DEFAULT_PROGRAMMING_RUST_PROMPT_TEXT: &str = "Write a Rust function that parses a comma-separated list of integers into a Vec, returning a Result with a helpful error on invalid input.";

#[derive(Debug, Clone)]
pub struct ValidationModelSpec {
    pub slug: String,
    pub family: Option<ModelFamily>,
    pub path: String,
    /// Optional per-model routing mode override. Set by entries in the
    /// `LINEUP_CONFIG` file (see `configs/local_gguf_lineup.template.toml`);
    /// autodiscovered / CLI-injected specs leave this `None` and fall back to
    /// `ModelConfig::routing_mode`.
    pub routing_mode: Option<RoutingMode>,
}

#[cfg(feature = "cuda")]
pub fn default_spiking_model_config(checkpoint_path: String, snn_steps: usize) -> ModelConfig {
    let probe = if checkpoint_path.trim().is_empty() {
        None
    } else {
        Router::probe_model(&checkpoint_path, None).ok()
    };

    ModelConfig {
        checkpoint_path,
        model_family: probe.as_ref().map(|metadata| metadata.family),
        gpu_synapse_tensor_name: probe
            .as_ref()
            .and_then(|metadata| metadata.real_gpu_synapse_tensor_name.clone())
            .unwrap_or_default(),
        num_experts: probe
            .as_ref()
            .map(|metadata| metadata.num_experts)
            .unwrap_or(8),
        top_k_experts: probe
            .as_ref()
            .map(|metadata| metadata.expert_used_count.max(1))
            .unwrap_or(1),
        routing_mode: RoutingMode::SpikingSim,
        snn_steps,
        projection_mode: ProjectionMode::SpikingTernary,
        gpu_routing_telemetry_path: None,
        ..Default::default()
    }
}

pub fn prompt_profile_slug() -> String {
    std::env::var("PROMPT_PROFILE")
        .unwrap_or_else(|_| "math_logic".into())
        .to_ascii_lowercase()
}

pub fn prompt_text_for_profile(profile: &str) -> &'static str {
    match profile {
        "math_logic" | "math" => DEFAULT_MATH_PROMPT_TEXT,
        "rust_syntax" | "rust" => DEFAULT_RUST_SYNTAX_PROMPT_TEXT,
        "english_snn" | "english" | "snn" => DEFAULT_ENGLISH_SNN_PROMPT_TEXT,
        "english_explanation" | "english_moe" | "moe_explanation" => {
            DEFAULT_ENGLISH_EXPLANATION_PROMPT_TEXT
        }
        "programming_rust" | "rust_parse" | "rust_programming" => {
            DEFAULT_PROGRAMMING_RUST_PROMPT_TEXT
        }
        _ => DEFAULT_MATH_PROMPT_TEXT,
    }
}

pub fn model_family_override_from_env() -> Option<ModelFamily> {
    let value = std::env::var("MODEL_FAMILY").ok()?;
    parse_family_slug(&value)
}

/// Shared family-slug parser used by `MODEL_FAMILY` and by lineup-config
/// `family = "..."` entries. Thin wrapper over [`ModelFamily::from_alias`].
pub fn parse_family_slug(value: &str) -> Option<ModelFamily> {
    ModelFamily::from_alias(value)
}

/// Parse a lineup-config `routing_mode` entry.
///
/// Thin alias for [`RoutingMode::from_alias`], which owns the canonical
/// spelling table.
pub fn parse_routing_mode(value: &str) -> Option<RoutingMode> {
    RoutingMode::from_alias(value)
}

pub fn saaq_update_rule_from_env() -> SaaqUpdateRule {
    match std::env::var("SAAQ_RULE")
        .unwrap_or_else(|_| "saaq_v1_5".into())
        .to_ascii_lowercase()
        .as_str()
    {
        "legacy" | "legacy_v1_0" | "v1_0" | "saaq_v1_0" => SaaqUpdateRule::LegacyV1_0,
        _ => SaaqUpdateRule::SaaqV1_5SqrtRate,
    }
}

pub fn prompt_embedding_for_validation(
    prompt_text: &str,
    target_dim: usize,
) -> Result<(Vec<f32>, String), Box<dyn std::error::Error>> {
    let provider = std::env::var("EMBEDDING_PROVIDER").ok();

    if resolve_prompt_embedding_provider(provider.as_deref()) == PromptEmbeddingProvider::Ollama {
        match pooled_prompt_embedding_from_ollama(prompt_text, target_dim) {
            Ok((embedding, label)) => return Ok((embedding, label)),
            Err(error) => {
                eprintln!(
                    "Ollama prompt embedding unavailable: {}. Falling back to deterministic text hash embedding.",
                    error
                );
            }
        }
    } else {
        eprintln!(
            "Unknown embedding provider '{}'. Falling back to deterministic text hash embedding.",
            provider.as_deref().unwrap_or("<unset>")
        );
    }

    Ok((
        synthetic_text_embedding(prompt_text, target_dim),
        "text_hash_fallback".into(),
    ))
}

pub fn pooled_prompt_embedding_from_ollama(
    prompt_text: &str,
    target_dim: usize,
) -> Result<(Vec<f32>, String), Box<dyn std::error::Error>> {
    let model =
        std::env::var("OLLAMA_EMBED_MODEL").unwrap_or_else(|_| "nomic-embed-text".to_string());
    let url = std::env::var("OLLAMA_EMBED_URL")
        .unwrap_or_else(|_| "http://localhost:11434/api/embed".to_string());
    let prefix =
        std::env::var("OLLAMA_EMBED_PREFIX").unwrap_or_else(|_| "classification: ".to_string());

    let input = format!("{}{}", prefix, prompt_text);
    let payload = serde_json::json!({
        "model": model,
        "input": input,
    });
    let payload_str = serde_json::to_string(&payload)?;

    let output = Command::new("curl")
        .arg("--fail-with-body")
        .arg("--silent")
        .arg("--show-error")
        .arg("--connect-timeout")
        .arg("5")
        .arg("--max-time")
        .arg("30")
        .arg("-X")
        .arg("POST")
        .arg(&url)
        .arg("-H")
        .arg("Content-Type: application/json")
        .arg("-d")
        .arg(&payload_str)
        .output()?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let stdout = String::from_utf8_lossy(&output.stdout);
        let detail = match (stderr.trim(), stdout.trim()) {
            ("", "") => format!("exit status {}", output.status),
            (stderr, "") => stderr.to_owned(),
            ("", stdout) => stdout.to_owned(),
            (stderr, stdout) => format!("{stderr}; response: {stdout}"),
        };
        return Err(Error::other(format!("Ollama curl request failed: {detail}")).into());
    }

    let stdout = String::from_utf8(output.stdout)?;

    #[derive(serde::Deserialize)]
    struct OllamaResponse {
        embeddings: Option<Vec<Vec<f32>>>,
    }

    let response: OllamaResponse = serde_json::from_str(&stdout).map_err(|e| {
        Error::other(format!(
            "Failed to parse Ollama response: {e}\nResponse: {stdout}"
        ))
    })?;

    let mut embedding = response
        .embeddings
        .and_then(|mut embs| embs.pop())
        .ok_or_else(|| Error::other("Ollama response did not contain embeddings"))?;

    if embedding.len() != target_dim {
        embedding = resample_embedding(&embedding, target_dim);
    }
    normalize_embedding(&mut embedding);

    let label = format!("ollama:{}", model);
    Ok((embedding, label))
}

pub fn discover_validation_models() -> Vec<ValidationModelSpec> {
    if let Ok(path) = std::env::var("CHECKPOINT_PATH")
        && !path.trim().is_empty()
    {
        let family = Router::probe_model(&path, None)
            .ok()
            .map(|metadata| metadata.family);
        return vec![ValidationModelSpec {
            slug: slug_from_path(&path),
            family,
            path,
            routing_mode: None,
        }];
    }

    let Some(home) = std::env::var_os("HOME") else {
        return Vec::new();
    };
    let root = PathBuf::from(home)
        .join("Downloads")
        .join("SNN_Quantization");
    let candidates = [
        (
            "olmoe_baseline",
            Some(ModelFamily::Olmoe),
            PathBuf::from("olmoe-0125-gguf/OLMoE-1B-7B-0125-Instruct-F16.gguf"),
        ),
        (
            "qwen3_moe_i1_iq3_m",
            Some(ModelFamily::Qwen3Moe),
            PathBuf::from("models/qwen3-moe-i1-GGUF/qwen3-moe.i1-IQ3_M.gguf"),
        ),
        (
            "gemma4_26b_a4b_iq4_nl",
            Some(ModelFamily::Gemma4),
            PathBuf::from("models/gemma-4-26B-A4B-it-GGUF/gemma-4-26B-A4B-it-UD-IQ4_NL.gguf"),
        ),
        (
            "deepseek_coder_v2_lite_q6_k_l",
            Some(ModelFamily::DeepSeek2),
            PathBuf::from(
                "models/DeepSeek-Coder-V2-Lite-Instruct-GGUF/DeepSeek-Coder-V2-Lite-Instruct-Q6_K_L.gguf",
            ),
        ),
        (
            "llama_3_2_dark_champion_q5_k_m",
            Some(ModelFamily::LlamaMoe),
            PathBuf::from(
                "models/Llama-3.2-8X3B-MOE-Dark-Champion-GGUF/L3.2-8X3B-MOE-Dark-Champion-Inst-18.4B-uncen-ablit_D_AU-q5_k_m.gguf",
            ),
        ),
        (
            "zaya1_8b_q8_0",
            Some(ModelFamily::Zaya),
            PathBuf::from("models/ZAYA1-8B-GGUF/ZAYA1-8B-Q8_0.gguf"),
        ),
        (
            "glm46v_flash_q8_0",
            Some(ModelFamily::Glm4),
            PathBuf::from("models/GLM-4.6V-Flash-GGUF_Q8_0/GLM-4.6V-Flash-Q8_0.gguf"),
        ),
        (
            "kimi_vl_a3b_q6_k",
            // llama.cpp GGUF arch is deepseek2; infer_family disambiguates
            // Moonlight/Kimi from the path. Do not hardcode DeepSeek2 here —
            // run_validation would stamp that override onto artifacts.
            None,
            PathBuf::from("models/Kimi-VL-A3B-Instruct-GGUF_Q6_K/Kimi-VL-A3B-Instruct-Q6_K.gguf"),
        ),
        (
            "marco_nano_base_q8_0",
            Some(ModelFamily::Qwen3Moe),
            PathBuf::from("models/Marco-Nano-Base-GGUF_Q8_0/Marco-Nano-Base.Q8_0.gguf"),
        ),
        (
            "moonlight_16b_a3b_q4_k_m",
            // Same packaging as kimi: GGUF arch is deepseek2; leave family
            // unset so the probe can disambiguate from the path.
            None,
            PathBuf::from(
                "models/Moonlight-16B-A3B-bnb-4bit/moonlight-16b-a3b-bnb-4bit-q4_k_m.gguf",
            ),
        ),
        (
            "granite_3_1_3b_a800m_q4_k_m",
            Some(ModelFamily::Granite31A800M),
            PathBuf::from(
                "models/ibm-granite/granite-3.1-3b-a800m-base-GGUF/granite-3.1-3b-a800m-base-q4_k_m.gguf",
            ),
        ),
    ];

    candidates
        .into_iter()
        .filter_map(|(slug, family, rel)| {
            let path = root.join(rel);
            path.exists().then(|| ValidationModelSpec {
                slug: slug.into(),
                family,
                path: path.to_string_lossy().into_owned(),
                routing_mode: None,
            })
        })
        .collect()
}

pub fn ticks_from_env(default_ticks: usize) -> usize {
    env_usize("TICKS", default_ticks)
}

pub fn repeat_count_from_env() -> usize {
    env_usize("REPEAT_COUNT", 1).max(1)
}

/// Machine-local `TELEMETRY_SOURCE` resolution (config boundary).
///
/// Empty / unrecognised values fall back to [`TelemetrySource::Synthetic`] so a
/// fresh clone never depends on a missing CSV path.
pub fn telemetry_source_from_env() -> TelemetrySource {
    match std::env::var("TELEMETRY_SOURCE")
        .unwrap_or_default()
        .to_ascii_lowercase()
        .trim()
    {
        "csv" => TelemetrySource::Csv,
        _ => TelemetrySource::Synthetic,
    }
}

/// Machine-local `TELEMETRY_CSV_PATH` resolution (config boundary).
///
/// Defaults to a repo-relative `telemetry.csv` placeholder when unset.
pub fn telemetry_csv_path_from_env() -> PathBuf {
    if let Ok(value) = std::env::var("TELEMETRY_CSV_PATH") {
        let trimmed = value.trim();
        if !trimmed.is_empty() {
            return PathBuf::from(trimmed);
        }
    }
    PathBuf::from("telemetry.csv")
}

/// Process-wide telemetry source: env/path resolution here, CSV load/fallback
/// in [`resolve_telemetry_from`].
pub fn resolve_telemetry_source() -> ResolvedTelemetry {
    resolve_telemetry_from(telemetry_source_from_env(), telemetry_csv_path_from_env())
}

/// Scale factor applied to the prompt embedding before GPU temporal upload.
///
/// L2-normalised 2048-dim embeddings have per-element magnitude ≈ 0.022.
/// The GIF kernel's `GIF_DRIVE_SCALE=0.75` and `GIF_THRESHOLD_BASE=0.65`
/// require effective drive ≥ ~0.87 per tick to fire.  A gain of 32 lifts
/// per-element input from ~0.022 to ~0.7, producing dot-product drives that
/// comfortably cross threshold and yield healthy 5–15 % firing rates.
///
/// Override via `INPUT_DRIVE_GAIN` env var for per-model tuning.
#[cfg(feature = "cuda")]
pub fn input_drive_gain_from_env() -> f32 {
    env_f32("INPUT_DRIVE_GAIN", 32.0)
}

/// Parse `ROUTING_MODE` into a [`RoutingMode`].
///
/// - Unset → `None` (callers keep lineup / `ModelConfig` default).
/// - Present and recognised → `Some(mode)`.
/// - Present but unrecognised → `None` **with a stderr diagnostic**, so a typo
///   is not silent (same soft-fallback style as unknown lineup `routing_mode`).
///
/// Delegates to [`RoutingMode::from_alias`] so the env var and lineup-config
/// entries accept exactly the same spellings. They diverged previously:
/// this function accepted only `dense`/`stub`, so `ROUTING_MODE=dense_sim`
/// was silently dropped and the config default won.
pub fn routing_mode_override_from_env() -> Option<RoutingMode> {
    let Ok(raw) = std::env::var("ROUTING_MODE") else {
        return None;
    };
    match RoutingMode::from_alias(&raw) {
        Some(mode) => Some(mode),
        None => {
            eprintln!(
                "ROUTING_MODE={raw:?} is not a recognised routing mode \
                 (expected dense|dense_sim|stub|stub_uniform|spiking|spiking_sim); \
                 using lineup/default"
            );
            None
        }
    }
}

/// Parse `PROJECTION_MODE` into a [`ProjectionMode`].
///
/// - Unset / blank → `None` (callers keep the `SpikingTernary` default).
/// - Present and recognised → `Some(mode)`.
/// - Present but unrecognised → **fail-fast** (`exit 1`). A typo must not
///   silently fall back to `SpikingTernary`; that is how
///   `PROJECTION_MODE=RateSum` was ignored on 2026-08-22 (GH#162).
/// - Present but not valid Unicode → **fail-fast** for the same reason.
pub fn projection_mode_override_from_env() -> Option<ProjectionMode> {
    match ProjectionMode::parse_env_result(std::env::var("PROJECTION_MODE")) {
        Ok(mode) => mode,
        Err(err) => {
            eprintln!("{err}");
            std::process::exit(1);
        }
    }
}

fn slug_from_path(path: &str) -> String {
    Path::new(path)
        .file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or("gguf_model")
        .replace(['.', '-', ' '], "_")
        .to_ascii_lowercase()
}

fn resample_embedding(input: &[f32], target_len: usize) -> Vec<f32> {
    if target_len == 0 {
        return Vec::new();
    }
    if input.len() == target_len {
        return input.to_vec();
    }
    if input.is_empty() {
        return vec![0.0; target_len];
    }
    if target_len == 1 {
        return vec![input.iter().sum::<f32>() / input.len() as f32];
    }

    let scale = (input.len() - 1) as f32 / (target_len - 1) as f32;
    let mut out = Vec::with_capacity(target_len);

    for idx in 0..target_len {
        let source = idx as f32 * scale;
        let lo = source.floor() as usize;
        let hi = source.ceil().min((input.len() - 1) as f32) as usize;
        if lo == hi {
            out.push(input[lo]);
        } else {
            let t = source - lo as f32;
            out.push(input[lo] * (1.0 - t) + input[hi] * t);
        }
    }
    out
}

fn normalize_embedding(values: &mut [f32]) {
    let l2_norm = values.iter().map(|&v| v * v).sum::<f32>().sqrt();
    if l2_norm > 1e-8 {
        for value in values {
            *value /= l2_norm;
        }
    }
}

fn synthetic_text_embedding(prompt_text: &str, target_dim: usize) -> Vec<f32> {
    if target_dim == 0 {
        return Vec::new();
    }

    let bytes = prompt_text.as_bytes();
    if bytes.is_empty() {
        return vec![0.0; target_dim];
    }

    let mut embedding = vec![0.0f32; target_dim];
    for (idx, _) in bytes.iter().enumerate() {
        let start = idx.saturating_sub(3);
        let hash = fnv1a64(&bytes[start..=idx]);
        let slot = (hash as usize) % target_dim;
        let sign = if ((hash >> 11) & 1) == 0 { 1.0 } else { -1.0 };
        let magnitude = 1.0 + (bytes[idx] as f32 / 255.0);
        embedding[slot] += sign * magnitude;
    }

    normalize_embedding(&mut embedding);
    embedding
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    for &byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

fn env_f32(key: &str, default_value: f32) -> f32 {
    std::env::var(key)
        .ok()
        .and_then(|value| value.parse::<f32>().ok())
        .filter(|value| value.is_finite())
        .unwrap_or(default_value)
}

fn env_usize(key: &str, default_value: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(default_value)
}

pub(super) fn env_flag(key: &str, default_value: bool) -> bool {
    std::env::var(key)
        .ok()
        .map(|value| {
            matches!(
                value.to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or(default_value)
}
