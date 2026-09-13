// SPDX-License-Identifier: Apache-2.0 OR MIT
use std::path::{Path, PathBuf};

use corinth_canal::{CloudModelSpec, ModelArchitectureClass, ModelFamily, ModelTarget};

fn parse_family_slug(value: &str) -> Option<ModelFamily> {
    ModelFamily::from_alias(value)
}

/// Parse a `CLOUD_LINEUP_CONFIG` file (see `configs/saaq_cloud_lineup.toml`)
/// and return every entry.
///
/// Unknown families are reported via stderr but accepted (family is left
/// `None` for probe-based inference downstream).
pub fn load_cloud_lineup(path: &Path) -> Result<Vec<CloudModelSpec>, Box<dyn std::error::Error>> {
    #[derive(Debug, serde::Deserialize)]
    #[serde(deny_unknown_fields)]
    struct RawCloudLineup {
        #[serde(default)]
        model: Vec<RawCloudModel>,
    }

    #[derive(Debug, serde::Deserialize)]
    #[serde(deny_unknown_fields)]
    struct RawCloudModel {
        slug: String,
        #[serde(default)]
        family: String,
        cloud_model_id: String,
        source_url: String,
        target: String,
        architecture: String,
        active_params: String,
        total_params: String,
        provider_format: String,
        #[serde(default)]
        required_env_vars: Vec<String>,
    }

    let raw = std::fs::read_to_string(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    let parsed: RawCloudLineup =
        toml::from_str(&raw).map_err(|e| format!("parse {}: {e}", path.display()))?;

    let mut out = Vec::with_capacity(parsed.model.len());
    for entry in parsed.model {
        let target = entry.target.trim().to_ascii_lowercase();
        if target != "cloud" {
            return Err(format!(
                "cloud_lineup: invalid target for slug={}: expected \"cloud\", got \"{}\"",
                entry.slug, entry.target
            )
            .into());
        }
        let architecture = match entry.architecture.trim().to_ascii_lowercase().as_str() {
            "moe" => ModelArchitectureClass::Moe,
            "dense" => ModelArchitectureClass::Dense,
            other => {
                return Err(format!(
                    "cloud_lineup: invalid architecture for slug={}: expected \"moe\" or \"dense\", got \"{other}\"",
                    entry.slug
                )
                .into());
            }
        };
        let family = parse_family_slug(&entry.family);
        if family.is_none() && !entry.family.is_empty() {
            eprintln!(
                "cloud_lineup: unknown family '{}' for slug={}; leaving family inference to probe",
                entry.family, entry.slug
            );
        }
        if entry.required_env_vars.is_empty() {
            eprintln!(
                "cloud_lineup: no required_env_vars for slug={}; cloud models are download-on-GPU, skipping guard",
                entry.slug
            );
        }
        let spec = CloudModelSpec {
            slug: entry.slug.clone(),
            family,
            cloud_model_id: entry.cloud_model_id.clone(),
            source_url: entry.source_url.clone(),
            target: ModelTarget::Cloud,
            architecture,
            active_params: entry.active_params.clone(),
            total_params: entry.total_params.clone(),
            provider_format: entry.provider_format.clone(),
            required_env_vars: entry.required_env_vars.clone(),
        };

        if !spec.cloud_provider_available() {
            let unset: Vec<_> = entry
                .required_env_vars
                .iter()
                .filter(|var| !std::env::var(var).is_ok_and(|v| !v.is_empty()))
                .collect();
            if !unset.is_empty() {
                eprintln!(
                    "cloud_lineup: provider unavailable for slug={} ({}): missing env vars: {}",
                    entry.slug,
                    entry.cloud_model_id,
                    unset
                        .iter()
                        .map(|s| s.as_str())
                        .collect::<Vec<_>>()
                        .join(", ")
                );
            }
        }

        out.push(spec);
    }
    Ok(out)
}

/// Fail-fast guard for cloud model execution.
///
/// Re-checks `required_env_vars` at call time and returns an error
/// describing any that are missing.
pub fn cloud_execution_guard(entry: &CloudModelSpec) -> Result<(), String> {
    if entry.required_env_vars.is_empty() {
        return Ok(());
    }
    let unset: Vec<_> = entry
        .required_env_vars
        .iter()
        .filter(|var| !std::env::var(var).is_ok_and(|v| !v.is_empty()))
        .collect();
    if unset.is_empty() {
        return Ok(());
    }
    Err(format!(
        "cloud model '{}' ({}) cannot execute: required env vars not set: {}. \
         Cloud execution is delegated to Dioscuri-Cloud; configure these env \
         vars in your deployment environment.",
        entry.slug,
        entry.cloud_model_id,
        unset
            .iter()
            .map(|s| s.as_str())
            .collect::<Vec<_>>()
            .join(", ")
    ))
}

/// Load the cloud lineup path from the `CLOUD_LINEUP_CONFIG` env var.
pub fn cloud_lineup_path_from_env() -> Option<PathBuf> {
    std::env::var("CLOUD_LINEUP_CONFIG")
        .ok()
        .filter(|v| !v.trim().is_empty())
        .map(PathBuf::from)
}

/// Parsed representation of a safetensors model entry from
/// `configs/safetensors_lineup.toml`.
#[derive(Debug, Clone)]
pub struct SafetensorsModelEntry {
    pub slug: String,
    pub family: Option<ModelFamily>,
    pub path: PathBuf,
    pub target: String,
}

/// Parse a `configs/safetensors_lineup.toml` file and return every entry
/// whose path exists on disk. Missing paths are skipped with a warning.
pub fn load_safetensors_lineup(
    path: &Path,
) -> Result<Vec<SafetensorsModelEntry>, Box<dyn std::error::Error>> {
    #[derive(Debug, serde::Deserialize)]
    #[serde(deny_unknown_fields)]
    struct RawSafetensorsLineup {
        #[serde(default)]
        model: Vec<RawSafetensorsModel>,
    }

    #[derive(Debug, serde::Deserialize)]
    #[serde(deny_unknown_fields)]
    struct RawSafetensorsModel {
        slug: String,
        #[serde(default)]
        family: String,
        path: String,
        target: String,
    }

    let raw = std::fs::read_to_string(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    let parsed: RawSafetensorsLineup =
        toml::from_str(&raw).map_err(|e| format!("parse {}: {e}", path.display()))?;

    let mut out = Vec::with_capacity(parsed.model.len());
    for entry in parsed.model {
        if !entry.target.trim().eq_ignore_ascii_case("local") {
            return Err(format!(
                "safetensors_lineup: invalid target for slug={}: expected \"local\", got \"{}\"",
                entry.slug, entry.target
            )
            .into());
        }
        let trimmed_path = entry.path.trim();
        let entry_path = PathBuf::from(trimmed_path);
        if !entry_path.exists() {
            eprintln!(
                "safetensors_lineup: skipping slug={}: path not found: {}",
                entry.slug, trimmed_path
            );
            continue;
        }
        let family = parse_family_slug(&entry.family);
        if family.is_none() && !entry.family.is_empty() {
            eprintln!(
                "safetensors_lineup: unknown family '{}' for slug={}",
                entry.family, entry.slug
            );
        }
        out.push(SafetensorsModelEntry {
            slug: entry.slug,
            family,
            path: entry_path,
            target: entry.target,
        });
    }
    Ok(out)
}

/// Load the safetensors lineup path from the `SAFETENSORS_LINEUP_CONFIG`
/// env var.
pub fn safetensors_lineup_path_from_env() -> Option<PathBuf> {
    std::env::var("SAFETENSORS_LINEUP_CONFIG")
        .ok()
        .filter(|v| !v.trim().is_empty())
        .map(PathBuf::from)
}
