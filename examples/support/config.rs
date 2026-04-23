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

use std::path::{Path, PathBuf};

use corinth_canal::{HeartbeatConfig, ModelFamily, SaaqUpdateRule, moe::RoutingMode};
use serde::Deserialize;

use super::{
    ResolvedTelemetry, ValidationModelSpec, discover_validation_models, env_flag,
    heartbeat_config_from_env, heartbeat_modes_for_matrix, llama_embedding_binary_optional,
    model_family_override_from_env, parse_family_slug, parse_routing_mode, prompt_profile_slug,
    prompt_text_for_profile, repeat_count_from_env, resolve_telemetry_source,
    routing_mode_override_from_env, saaq_update_rule_from_env, ticks_from_env,
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
    /// Path to the parsed lineup config (if `LINEUP_CONFIG` was set and
    /// successfully resolved). Stamped into the run manifest for provenance.
    pub lineup_config_path: Option<PathBuf>,
    /// Free-form run tag from `RUN_TAG`. Empty / unset maps to `None` so
    /// callers can just do `if let Some(tag) = cfg.run_tag { ... }`.
    pub run_tag: Option<String>,
    /// When `true` and `repeat_count >= 2`, the calibration runner asserts
    /// byte-equality of `latent_telemetry.csv` across repeats per
    /// `(model_slug, telemetry_source, heartbeat_slug, saaq_rule)` group.
    pub strict_repeat_check: bool,
}

impl RunConfig {
    /// Build a `RunConfig` by reading every supported environment variable.
    ///
    /// Call `dotenvy::from_filename(".env.local").ok();` before this if you
    /// want `.env.local` overrides applied.
    pub fn from_env() -> Self {
        let prompt_profile = prompt_profile_slug();
        let prompt_text = prompt_text_for_profile(&prompt_profile);
        let gguf_checkpoint_path = std::env::var("GGUF_CHECKPOINT_PATH").unwrap_or_default();
        let lineup_config_path = lineup_config_path_from_env();
        let validation_models = resolve_validation_models(
            lineup_config_path.as_deref(),
            &gguf_checkpoint_path,
        );
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
            validation_models,
            gguf_checkpoint_path,
            routing_mode_override: routing_mode_override_from_env(),
            lineup_config_path,
            run_tag: run_tag_from_env(),
            strict_repeat_check: strict_repeat_check_from_env(),
        }
    }
}

/// Resolve the validation-model list with the documented precedence:
///
///   1. `LINEUP_CONFIG` file (hard error if set but unparseable).
///   2. `GGUF_CHECKPOINT_PATH` (single-model override via the legacy path).
///   3. Machine-local autodiscovery under `$HOME/Downloads/SNN_Quantization`.
fn resolve_validation_models(
    lineup_path: Option<&Path>,
    gguf_checkpoint_path: &str,
) -> Vec<ValidationModelSpec> {
    if let Some(path) = lineup_path {
        match load_lineup_file(path) {
            Ok(models) => return models,
            Err(err) => {
                // Hard-fail with a loud message so a typo / missing path is
                // never silently papered over by autodiscovery.
                panic!(
                    "LINEUP_CONFIG={} could not be loaded: {err}",
                    path.display()
                );
            }
        }
    }

    // Legacy single-model override or autodiscovery — let the existing
    // helper keep its current contract.
    let _ = gguf_checkpoint_path; // discover_validation_models reads it directly
    discover_validation_models()
}

#[derive(Debug, Deserialize)]
struct RawLineup {
    #[serde(default)]
    model: Vec<RawLineupModel>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawLineupModel {
    slug: String,
    family: String,
    path: String,
    #[serde(default)]
    routing_mode: Option<String>,
}

/// Parse the lineup TOML and convert each entry to a `ValidationModelSpec`.
/// Missing files are skipped with a warning (same non-fatal behavior as
/// `discover_validation_models`); unknown family / routing-mode slugs are
/// reported to stderr but do not abort the run.
fn load_lineup_file(path: &Path) -> Result<Vec<ValidationModelSpec>, Box<dyn std::error::Error>> {
    let raw = std::fs::read_to_string(path)
        .map_err(|e| format!("read {}: {e}", path.display()))?;
    let parsed: RawLineup = toml::from_str(&raw)
        .map_err(|e| format!("parse {}: {e}", path.display()))?;

    let mut out = Vec::with_capacity(parsed.model.len());
    for entry in parsed.model {
        let RawLineupModel { slug, family, path: gguf_path, routing_mode } = entry;
        let trimmed_path = gguf_path.trim();
        if trimmed_path.is_empty() {
            eprintln!(
                "lineup_config: skipping entry slug={slug}: empty path",
            );
            continue;
        }
        if !Path::new(trimmed_path).exists() {
            eprintln!(
                "lineup_config: skipping entry slug={slug} path={trimmed_path}: file not found",
            );
            continue;
        }
        let parsed_family = parse_family_slug(&family);
        if parsed_family.is_none() {
            eprintln!(
                "lineup_config: unknown family '{family}' for slug={slug}; leaving family inference to probe",
            );
        }
        let parsed_routing = match routing_mode.as_deref() {
            Some(value) => {
                let resolved = parse_routing_mode(value);
                if resolved.is_none() {
                    eprintln!(
                        "lineup_config: unknown routing_mode '{value}' for slug={slug}; using ModelConfig default",
                    );
                }
                resolved
            }
            None => None,
        };

        out.push(ValidationModelSpec {
            slug,
            family: parsed_family,
            path: trimmed_path.to_owned(),
            routing_mode: parsed_routing,
        });
    }

    Ok(out)
}

/// Parse `LINEUP_CONFIG`. Empty / unset => `None`.
pub fn lineup_config_path_from_env() -> Option<PathBuf> {
    std::env::var("LINEUP_CONFIG")
        .ok()
        .map(|s| s.trim().to_owned())
        .filter(|s| !s.is_empty())
        .map(PathBuf::from)
}

/// Parse `RUN_TAG`. Empty / unset => `None`. Whitespace-only values are
/// normalized to `None`.
pub fn run_tag_from_env() -> Option<String> {
    std::env::var("RUN_TAG")
        .ok()
        .map(|s| s.trim().to_owned())
        .filter(|s| !s.is_empty())
}

/// Parse `STRICT_REPEAT_CHECK`. Default `false` so existing workflows keep
/// their current behavior when the env var is unset.
pub fn strict_repeat_check_from_env() -> bool {
    env_flag("STRICT_REPEAT_CHECK", false)
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
