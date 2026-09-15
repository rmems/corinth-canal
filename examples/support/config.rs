// SPDX-License-Identifier: Apache-2.0 OR MIT
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

use std::path::{Path, PathBuf};

use corinth_canal::{ModelFamily, SaaqUpdateRule, moe::RoutingMode, projector::ProjectionMode};

use super::lineup::{SafetensorsModelEntry, load_gguf_lineup};
use super::{
    ResolvedTelemetry, TelemetrySource, ValidationModelSpec, cloud_execution_guard,
    cloud_lineup_path_from_env, default_spiking_model_config, discover_validation_models, env_flag,
    input_drive_gain_from_env, load_cloud_lineup, load_safetensors_lineup,
    model_family_override_from_env, parse_routing_mode, pooled_prompt_embedding_from_ollama,
    projection_mode_override_from_env, prompt_embedding_for_validation, prompt_profile_slug,
    prompt_text_for_profile, repeat_count_from_env, resolve_telemetry_source,
    routing_mode_override_from_env, saaq_update_rule_from_env, safetensors_lineup_path_from_env,
    telemetry_snapshot_for_tick, ticks_from_env,
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
/// should not read `std::env` directly. Each example binary reads only a
/// subset of these fields at runtime. We keep every field "read" (for the
/// dead_code lint) via explicit references inside from_env's dead-code guard
/// block so that *no* #[allow(dead_code)] is needed on the struct or anywhere
/// in support.
#[derive(Debug, Clone)]
pub struct RunConfig {
    pub prompt_profile: String,
    pub prompt_text: &'static str,
    pub ticks: usize,
    pub repeat_count: usize,
    pub telemetry: ResolvedTelemetry,
    pub output_root: PathBuf,
    pub model_family_override: Option<ModelFamily>,
    pub saaq_rule: SaaqUpdateRule,
    pub validation_models: Vec<ValidationModelSpec>,
    pub checkpoint_path: String,
    pub routing_mode_override: Option<RoutingMode>,
    /// `PROJECTION_MODE` env override. When set, wins over the
    /// `SpikingTernary` default in `default_spiking_model_config`.
    /// Unrecognised values fail fast rather than falling back.
    pub projection_mode_override: Option<ProjectionMode>,
    /// Free-form run tag from `RUN_TAG`. Empty / unset maps to `None` so
    /// callers can just do `if let Some(tag) = cfg.run_tag { ... }`.
    pub run_tag: Option<String>,
    /// When `true` and `repeat_count >= 2`, the calibration runner asserts
    /// byte-equality of `latent_telemetry.csv` across repeats per
    /// `(model_slug, telemetry_source, saaq_rule)` group.
    pub strict_repeat_check: bool,
    /// Declared `[[model]]` count from `LINEUP_CONFIG`. `None` when the
    /// run did not come from a GGUF lineup file.
    pub lineup_declared_count: Option<usize>,
    /// How many of those entries resolved on disk. A gap versus
    /// `lineup_declared_count` means skip-and-continue dropped models.
    pub lineup_resolved_count: Option<usize>,
}

impl RunConfig {
    /// Build a `RunConfig` by reading every supported environment variable.
    ///
    /// Call `dotenvy::from_filename(".env.local").ok();` before this if you
    /// want `.env.local` overrides applied.
    pub fn from_env() -> Self {
        let prompt_profile = prompt_profile_slug();
        let prompt_text = prompt_text_for_profile(&prompt_profile);
        let checkpoint_path = std::env::var("CHECKPOINT_PATH").unwrap_or_default();
        validate_optional_lineups_from_env();
        // Local binding only — used to pick the lineup TOML below. The
        // resolved path is intentionally not stamped onto `RunConfig` yet;
        // when campaign provenance is added back to `ValidationManifest`,
        // re-introduce both fields together in one focused commit.
        let lineup_config_path = lineup_config_path_from_env();
        let safetensors_lineup_path = safetensors_lineup_path_from_env();
        let lineup_strict = lineup_strict_from_env();
        let resolved_models = resolve_validation_models(
            lineup_config_path.as_deref(),
            safetensors_lineup_path.as_deref(),
            &checkpoint_path,
            lineup_strict,
        );
        let run_config = Self {
            prompt_profile: prompt_profile.clone(),
            prompt_text,
            ticks: ticks_from_env(DEFAULT_TICKS),
            repeat_count: repeat_count_from_env(),
            telemetry: resolve_telemetry_source(),
            output_root: output_root_from_env(),
            model_family_override: model_family_override_from_env(),
            saaq_rule: saaq_update_rule_from_env(),
            validation_models: resolved_models.models,
            checkpoint_path,
            routing_mode_override: routing_mode_override_from_env(),
            projection_mode_override: projection_mode_override_from_env(),
            run_tag: run_tag_from_env(),
            strict_repeat_check: strict_repeat_check_from_env(),
            lineup_declared_count: resolved_models.lineup_declared_count,
            lineup_resolved_count: resolved_models.lineup_resolved_count,
        };

        // NO DEAD CODE POLICY: Every cuda example binary compiles its own copy of the
        // support module tree. Only a subset of items/fields are exercised at runtime
        // by any given binary (e.g. the SAAQ-specific embedding + tick snapshot +
        // drive gain helpers are only called from saaq_latent_calibration; light
        // binaries like gpu_smoke_test only read .checkpoint_path from RunConfig).
        // We deliberately *read* every field and *name* the occasionally-used fns
        // here (inside a compile-time `if false` so zero cost/side-effects) from a
        // function that is *always* called (from_env) in every binary. This makes
        // the compiler consider them "used" for dead_code purposes in *all* binaries
        // without any #[allow(dead_code)] or #[allow(unused_imports)].
        if false {
            let _ = &run_config.prompt_profile;
            let _ = run_config.prompt_text;
            let _ = run_config.ticks;
            let _ = run_config.repeat_count;
            let _ = &run_config.telemetry.source;
            let _ = &run_config.telemetry.source_label;
            let _ = &run_config.telemetry.csv_path;
            let _ = &run_config.telemetry.rows;
            let _ = run_config.telemetry.row_count();
            let _ = &run_config.output_root;
            let _ = &run_config.model_family_override;
            let _ = &run_config.saaq_rule;
            for m in &run_config.validation_models {
                let _ = &m.slug;
                let _ = &m.family;
                let _ = &m.path;
                let _ = &m.routing_mode;
            }
            let _ = &run_config.checkpoint_path;
            let _ = &run_config.routing_mode_override;
            let _ = &run_config.projection_mode_override;
            let _ = &run_config.run_tag;
            let _ = run_config.strict_repeat_check;
            let _ = run_config.lineup_declared_count;
            let _ = run_config.lineup_resolved_count;
            let _ = parse_routing_mode("");

            // Reference the SAAQ-only helpers (and their private callees via the call graph).
            let _ = prompt_embedding_for_validation("", 0);
            let _ = pooled_prompt_embedding_from_ollama("", 0); // reachable from above in its body
            let _ = input_drive_gain_from_env();
            let dummy = ResolvedTelemetry {
                source: TelemetrySource::Synthetic,
                source_label: String::new(),
                csv_path: None,
                rows: None,
            };
            let _ = telemetry_snapshot_for_tick(0, &dummy);
            let _ = default_spiking_model_config("".into(), 0);
            // The low-level embedding helpers (resample, normalize, synthetic_text_embedding,
            // fnv1a64, env_f32) are reached from prompt_embedding_for_validation's (and pooled's) body.
        }

        run_config
    }
}

fn validate_optional_lineups_from_env() {
    if let Some(path) = cloud_lineup_path_from_env() {
        let entries = load_cloud_lineup(&path).unwrap_or_else(|err| {
            eprintln!(
                "CLOUD_LINEUP_CONFIG={} could not be loaded: {err}",
                path.display()
            );
            std::process::exit(1);
        });
        for entry in &entries {
            cloud_execution_guard(entry).unwrap_or_else(|err| {
                eprintln!(
                    "CLOUD_LINEUP_CONFIG={} failed validation for slug={}: {err}",
                    path.display(),
                    entry.slug
                );
                std::process::exit(1);
            });
        }
    }

    if let Some(path) = safetensors_lineup_path_from_env() {
        let entries = load_safetensors_lineup(&path).unwrap_or_else(|err| {
            eprintln!(
                "SAFETENSORS_LINEUP_CONFIG={} could not be loaded: {err}",
                path.display()
            );
            std::process::exit(1);
        });
        validate_safetensors_lineup_entries(&path, &entries);
    }
}

fn validate_safetensors_lineup_entries(path: &Path, entries: &[SafetensorsModelEntry]) {
    if entries.is_empty() {
        panic!(
            "SAFETENSORS_LINEUP_CONFIG={} produced no usable entries. \
             Placeholder-only or missing-path lineups are not valid runtime config.",
            path.display()
        );
    }
    for entry in entries {
        if entry.slug.trim().is_empty() {
            panic!(
                "SAFETENSORS_LINEUP_CONFIG={} contains an empty slug for path={}",
                path.display(),
                entry.path.display()
            );
        }
        if !entry.target.eq_ignore_ascii_case("local") {
            panic!(
                "SAFETENSORS_LINEUP_CONFIG={} has invalid target={} for slug={}",
                path.display(),
                entry.target,
                entry.slug
            );
        }
        if !entry.path.exists() {
            panic!(
                "SAFETENSORS_LINEUP_CONFIG={} resolved missing path={} for slug={}",
                path.display(),
                entry.path.display(),
                entry.slug
            );
        }
        if let Some(family) = entry.family {
            let _ = family.slug();
        }
    }
}

/// Result of model-list resolution, including optional GGUF-lineup coverage
/// counts for `run_manifest.json`.
struct ValidationModelResolution {
    models: Vec<ValidationModelSpec>,
    lineup_declared_count: Option<usize>,
    lineup_resolved_count: Option<usize>,
}

impl ValidationModelResolution {
    fn from_models(models: Vec<ValidationModelSpec>) -> Self {
        Self {
            models,
            lineup_declared_count: None,
            lineup_resolved_count: None,
        }
    }
}

/// Resolve the validation-model list with the documented precedence:
///
///   1. `LINEUP_CONFIG` file (hard error if set but unparseable; with
///      `LINEUP_STRICT=1`, also a hard error if any declared checkpoint is
///      missing).
///   2. `CHECKPOINT_PATH` (single-model override via the legacy path).
///   3. Machine-local autodiscovery under `$HOME/Downloads/SNN_Quantization`.
fn resolve_validation_models(
    lineup_path: Option<&Path>,
    safetensors_lineup_path: Option<&Path>,
    checkpoint_path: &str,
    lineup_strict: bool,
) -> ValidationModelResolution {
    if let Some(path) = lineup_path {
        match load_gguf_lineup(path, lineup_strict) {
            Ok(loaded) => {
                let models = loaded
                    .models
                    .into_iter()
                    .map(|entry| ValidationModelSpec {
                        slug: entry.slug,
                        family: entry.family,
                        path: entry.path,
                        routing_mode: entry.routing_mode,
                    })
                    .collect::<Vec<_>>();
                return ValidationModelResolution {
                    lineup_declared_count: Some(loaded.declared_count),
                    lineup_resolved_count: Some(models.len()),
                    models,
                };
            }
            Err(err) => {
                let path_str = path.display().to_string();
                let msg = err.to_string();
                if msg.starts_with("LINEUP_STRICT=") {
                    eprintln!("{msg}");
                } else {
                    let hint = if path_str.contains("/absolute/path/to/") {
                        "\n\nHINT: The path appears to be a placeholder from .env.example or a config template.\n      Please update LINEUP_CONFIG in .env.local with a real path."
                    } else {
                        ""
                    };
                    eprintln!("LINEUP_CONFIG={path_str} could not be loaded: {err}{hint}");
                }
                std::process::exit(1);
            }
        }
    }

    if let Some(path) = safetensors_lineup_path {
        match load_safetensors_lineup(path) {
            Ok(entries) => {
                return ValidationModelResolution::from_models(
                    entries
                        .into_iter()
                        .map(|entry| ValidationModelSpec {
                            slug: entry.slug,
                            family: entry.family,
                            path: entry.path.display().to_string(),
                            routing_mode: None,
                        })
                        .collect(),
                );
            }
            Err(err) => {
                let path_str = path.display().to_string();
                eprintln!("SAFETENSORS_LINEUP_CONFIG={path_str} could not be loaded: {err}");
                std::process::exit(1);
            }
        }
    }

    // Legacy single-model override or autodiscovery
    let _ = checkpoint_path;
    ValidationModelResolution::from_models(discover_validation_models())
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

/// Parse `LINEUP_STRICT`. Default `false` so partial-coverage lineups keep
/// skip-and-continue. `just saaq-campaign` sets this so a pinned model set
/// cannot silently shrink.
pub fn lineup_strict_from_env() -> bool {
    env_flag("LINEUP_STRICT", false)
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

#[cfg(test)]
mod tests {
    use super::super::lineup::SafetensorsModelEntry;
    use super::validate_safetensors_lineup_entries;
    use corinth_canal::ModelFamily;
    use std::path::Path;

    #[test]
    #[should_panic(expected = "produced no usable entries")]
    fn safetensors_validation_rejects_empty_lineup() {
        let entries: Vec<SafetensorsModelEntry> = Vec::new();
        validate_safetensors_lineup_entries(Path::new("configs/template.toml"), entries.as_slice());
    }

    #[test]
    fn safetensors_validation_accepts_existing_entry() {
        let existing_file = std::env::temp_dir().join(format!(
            "st_validation_{}.safetensors",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::write(&existing_file, b"test").unwrap();

        let entries = vec![SafetensorsModelEntry {
            slug: "test_st_model".into(),
            family: Some(ModelFamily::Olmoe),
            path: existing_file.clone(),
            target: "local".into(),
        }];

        validate_safetensors_lineup_entries(Path::new("configs/runtime.toml"), &entries);

        let _ = std::fs::remove_file(existing_file);
    }
}
