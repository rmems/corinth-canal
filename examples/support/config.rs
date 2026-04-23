//! Single source of env-driven runtime config for example binaries.
//!
//! The runtime crate (`src/`) never reads environment variables for paths.
//! Every machine-local default lives here. Each example binary should start
//! with:
//!
//! ```ignore
//! let _ = dotenvy::from_filename(".env.local");
//! let cfg = support::RunConfig::from_env();
//! ```
//!
//! Fields map 1:1 to the entries documented in `.env.example`.

#![allow(dead_code)]

use std::path::PathBuf;

use corinth_canal::{HeartbeatConfig, ModelFamily, SaaqUpdateRule, moe::RoutingMode};

use super::{
    ResolvedTelemetry, ValidationModelSpec, discover_validation_models,
    heartbeat_config_from_env, heartbeat_modes_for_matrix, llama_embedding_binary_optional,
    model_family_override_from_env, prompt_profile_slug, prompt_text_for_profile,
    repeat_count_from_env, resolve_telemetry_source, routing_mode_override_from_env,
    saaq_update_rule_from_env, ticks_from_env,
};

/// Default output root for per-run artifacts when `VALIDATION_OUTPUT_ROOT`
/// is unset. Repo-relative on purpose so a fresh clone never writes into a
/// machine-specific consumer directory.
pub const DEFAULT_OUTPUT_ROOT: &str = "artifacts";

/// Default tick count for `saaq_latent_calibration` when `TICKS` is unset.
pub const DEFAULT_TICKS: usize = 512;

/// Aggregated env-driven configuration for an example binary run.
///
/// Every field is populated by `RunConfig::from_env()` in one pass. Binaries
/// should not read `std::env` directly.
#[derive(Debug, Clone)]
pub struct RunConfig {
    pub prompt_profile: String,
    pub prompt_text: &'static str,
    pub ticks: usize,
    pub repeat_count: usize,
    pub heartbeat: HeartbeatConfig,
    pub heartbeat_matrix: Vec<bool>,
    pub telemetry: ResolvedTelemetry,
    pub output_root: PathBuf,
    pub model_family_override: Option<ModelFamily>,
    pub saaq_rule: SaaqUpdateRule,
    pub llama_embedding_bin: Option<PathBuf>,
    pub validation_models: Vec<ValidationModelSpec>,
    pub gguf_checkpoint_path: String,
    pub routing_mode_override: Option<RoutingMode>,
}

impl RunConfig {
    /// Build a `RunConfig` by reading every supported environment variable.
    ///
    /// Call `dotenvy::from_filename(".env.local").ok();` before this if you
    /// want `.env.local` overrides applied.
    pub fn from_env() -> Self {
        let prompt_profile = prompt_profile_slug();
        let prompt_text = prompt_text_for_profile(&prompt_profile);
        Self {
            prompt_profile: prompt_profile.clone(),
            prompt_text,
            ticks: ticks_from_env(DEFAULT_TICKS),
            repeat_count: repeat_count_from_env(),
            heartbeat: heartbeat_config_from_env(),
            heartbeat_matrix: heartbeat_modes_for_matrix(),
            telemetry: resolve_telemetry_source(),
            output_root: output_root_from_env(),
            model_family_override: model_family_override_from_env(),
            saaq_rule: saaq_update_rule_from_env(),
            llama_embedding_bin: llama_embedding_binary_optional(),
            validation_models: discover_validation_models(),
            gguf_checkpoint_path: std::env::var("GGUF_CHECKPOINT_PATH").unwrap_or_default(),
            routing_mode_override: routing_mode_override_from_env(),
        }
    }
}

/// Resolve `VALIDATION_OUTPUT_ROOT`, falling back to the repo-relative
/// default `./artifacts`.
pub fn output_root_from_env() -> PathBuf {
    if let Ok(value) = std::env::var("VALIDATION_OUTPUT_ROOT") {
        let trimmed = value.trim();
        if !trimmed.is_empty() {
            return PathBuf::from(trimmed);
        }
    }
    PathBuf::from(DEFAULT_OUTPUT_ROOT)
}
