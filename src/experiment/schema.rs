// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Standardized experiment output schema for SAAQ quantization runs.
//!
//! This module defines the canonical JSON structures emitted by
//! `saaq_latent_calibration` and consumed by downstream tools such as
//! `Surrogate_Viz.jl`.

use crate::error::{HybridError, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Top-level manifest for a single SAAQ experiment run.
///
/// Emitted as `run_manifest.json` inside every run directory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentManifest {
    pub run_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub run_tag: Option<String>,
    pub created_at: String,
    pub repo: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub commit_sha: Option<String>,

    pub model_slug: String,
    pub model_family: String,
    pub architecture: String,
    pub checkpoint_path: String,
    pub checkpoint_format: String,
    pub routing_tensor_name: String,
    pub synapse_source: String,

    pub prompt_embedding_source: String,
    pub prompt_profile: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_text: Option<String>,

    pub ticks: usize,
    pub saaq_rule: String,
    pub saaq_primary_rule: String,
    pub saaq_dual_emit: bool,

    pub telemetry_source: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub telemetry_csv_path: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub telemetry_row_count: Option<usize>,

    pub wraparound_enabled: bool,
    pub wraparound_loops: u64,
    pub ticks_effective: usize,

    pub run_dir: String,
    pub output_root: String,
    pub repeat_idx: usize,
    pub repeat_count: usize,

    pub validation_status: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub routing_mode: Option<String>,

    /// Projector strategy used for this run (`rate_sum`,
    /// `temporal_histogram`, `membrane_snapshot`, `spiking_ternary`).
    /// Absent on historical manifests written before GH#162.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub projection_mode: Option<String>,

    /// Declared `[[model]]` count from `LINEUP_CONFIG`. Absent when the run
    /// did not come from a GGUF lineup file (checkpoint / autodiscovery).
    /// GH#197 / RM-1209.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lineup_declared_count: Option<usize>,
    /// How many of those entries resolved on disk. A gap versus
    /// `lineup_declared_count` means skip-and-continue dropped models.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lineup_resolved_count: Option<usize>,

    pub generated_files: Vec<String>,
}

/// Compact, stable per-run summary consumed by downstream aggregators.
///
/// Emitted as `summary.json` inside every run directory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentSummary {
    pub run_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub run_tag: Option<String>,
    pub model_slug: String,
    pub model_family: String,
    pub telemetry_source: String,
    pub repeat_idx: usize,
    pub repeat_count: usize,
    pub saaq_rule: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub projection_mode: Option<String>,
    pub validation_status: String,
    pub run_dir: String,
    pub manifest_path: String,
    pub tick_telemetry_path: String,
    pub latent_telemetry_path: String,
    pub metrics: ExperimentMetrics,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub repeat_determinism: Option<String>,
}

/// Quantitative metrics for a single experiment run.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ExperimentMetrics {
    pub ticks_completed: usize,
    pub latent_rows: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mean_tick_elapsed_us: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub first_timestamp_ms: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_timestamp_ms: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub weight_mse: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub weight_cosine: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_abs_error: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub block_output_cosine: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub block_output_rmse: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub router_top1_agreement: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub router_top2_set_agreement: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expert_load_delta: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expert_load_js_divergence: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub compression_estimate: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub runtime_seconds: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub peak_memory_gib: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub warnings_count: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub failure_count: Option<usize>,
}

/// Standardized warning record.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentWarning {
    pub category: String,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tensor_name: Option<String>,
}

/// Output bundle emitted by every SAAQ run.
///
/// This is the top-level structure that `Surrogate_Viz.jl` expects to ingest.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentBundle {
    pub manifest: ExperimentManifest,
    pub summary: ExperimentSummary,
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub warnings: Vec<ExperimentWarning>,
    #[serde(skip_serializing_if = "BTreeMap::is_empty", default)]
    pub metadata: BTreeMap<String, String>,
}

// ── Model adapter configuration ───────────────────────────────────────────

/// Static per-model-family configuration used by the run matrix.
///
/// Emitted as `model_adapter_configs.toml` and consumed by the run matrix
/// validator and `Surrogate_Viz.jl`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelAdapterConfig {
    pub model_family: String,
    pub model_id_or_local_path: String,
    pub format: String,
    pub loader_hint: String,
    pub router_policy: String,
    pub norm_policy: String,
    pub expert_policy: String,
    pub supports_route_metrics: bool,
    pub supports_block_metrics: bool,
    #[serde(default)]
    pub preferred_quant_modes: Vec<String>,
    #[serde(default)]
    pub minimum_expected_artifacts: Vec<String>,
    #[serde(default)]
    pub known_risks: Vec<String>,
}

/// Dynamic per-run entry in the SAAQ run matrix.
///
/// A `RunEntry` is a *plan*, not an execution contract. No runner in this
/// repository loads a `RunMatrix`: `saaq_latent_calibration` is driven by
/// env / lineup config, so `max_runtime_minutes`, `max_disk_gib`,
/// `expected_artifacts` and `success_metrics` are declared budgets and
/// post-hoc review criteria for the operator (and `Surrogate_Viz.jl`), not
/// limits anything kills a run on or evaluates automatically (RM-1448).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunEntry {
    pub run_id: String,
    pub model_family: String,
    pub model_id_or_path: String,
    pub source_format: String,
    pub quant_mode: String,
    pub saaq_formula_version: String,
    pub router_policy: String,
    pub norm_policy: String,
    pub telemetry_source: String,
    pub output_root: String,
    /// Declared runtime budget in minutes. Must be non-zero for an enabled
    /// run so every plan states a budget; **not enforced** by any runner.
    pub max_runtime_minutes: u64,
    /// Declared disk budget in GiB. Must be non-zero for an enabled run;
    /// **not enforced** by any runner.
    pub max_disk_gib: u64,
    /// Files the operator expects in the run directory. Checked by review,
    /// not by code.
    #[serde(default)]
    pub expected_artifacts: Vec<String>,
    /// Free-form success criteria for post-hoc review. Not evaluated by code.
    #[serde(default)]
    pub success_metrics: Vec<String>,
    /// When set, the run is a documented skip: `validate` bypasses every
    /// check for it (budgets, policies, family, Grok-1 gate).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub skip_reason: Option<String>,
}

/// Full run matrix with validation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunMatrix {
    #[serde(rename = "run")]
    pub runs: Vec<RunEntry>,
}

impl RunMatrix {
    /// Validate the run matrix as a *plan*.
    ///
    /// Checks, for every run without a `skip_reason`:
    /// - `max_runtime_minutes` and `max_disk_gib` are declared (non-zero).
    ///   This only asserts the plan states a budget; nothing enforces it.
    /// - `router_policy` and `norm_policy` are explicit.
    /// - A run naming `grok-1` is rejected unless `GROK1_ARTIFACT_READY=1`
    ///   (readiness flag, not a path). The shipped matrix carries a
    ///   `skip_reason` on its grok-1 entry so it validates on a clean
    ///   checkout; drop the `skip_reason` *and* set the flag to enable it.
    /// - `model_family` is a known family slug.
    ///
    /// Runs with a `skip_reason` bypass all of the above.
    pub fn validate(&self) -> Result<()> {
        let grok_ready = std::env::var("GROK1_ARTIFACT_READY").is_ok_and(|v| v == "1");
        self.validate_with(grok_ready)
    }

    /// [`RunMatrix::validate`] with the Grok-1 readiness flag supplied by
    /// the caller instead of read from the environment. Tests use this so
    /// both branches of the gate run regardless of the real env.
    pub fn validate_with(&self, grok_ready: bool) -> Result<()> {
        for run in &self.runs {
            if run.is_documented_skip()? {
                continue;
            }
            run.validate_budgets()?;
            run.validate_policies()?;
            run.validate_family(grok_ready)?;
        }
        Ok(())
    }
}

const KNOWN_FAMILIES: &[&str] = &[
    "olmoe",
    "qwen3_moe",
    "gemma4",
    "deepseek2",
    "llama_moe",
    "moonlight_16b_a3b",
    "granite_3_1_a800m",
    "nemotron",
    "lfm2_moe",
    "slim_moe",
    "zaya",
    "glm4",
    "gpt_oss",
    "step",
    "minimax",
    "cohere",
    "grin",
    "skyworks",
    "trinity",
    "grok",
];

impl RunEntry {
    fn invalid(&self, message: impl std::fmt::Display) -> HybridError {
        HybridError::InvalidConfig(format!("run '{}' {message}", self.run_id))
    }

    /// `Ok(true)` when the run carries a non-blank `skip_reason` and so
    /// bypasses every other check. A `Some` but blank/whitespace reason is
    /// not a documented skip and is rejected rather than silently bypassing.
    fn is_documented_skip(&self) -> Result<bool> {
        match self.skip_reason.as_deref().map(str::trim) {
            None => Ok(false),
            Some("") => Err(self.invalid("has a blank skip_reason; it must be non-empty when set")),
            Some(_) => Ok(true),
        }
    }

    /// Declared budgets must be stated (non-zero). Nothing enforces them.
    fn validate_budgets(&self) -> Result<()> {
        const NOTE: &str = "(declared budget; must be non-zero, not enforced by any runner)";
        if self.max_runtime_minutes == 0 {
            return Err(self.invalid(format!("missing max_runtime_minutes {NOTE}")));
        }
        if self.max_disk_gib == 0 {
            return Err(self.invalid(format!("missing max_disk_gib {NOTE}")));
        }
        Ok(())
    }

    /// Router and norm policies must be explicit.
    fn validate_policies(&self) -> Result<()> {
        if self.router_policy.trim().is_empty() {
            return Err(self.invalid("missing router_policy"));
        }
        if self.norm_policy.trim().is_empty() {
            return Err(self.invalid("missing norm_policy"));
        }
        Ok(())
    }

    /// Grok-1 readiness gate (grok-1 only, not grok-10 / grok-11) and the
    /// known-family check.
    fn validate_family(&self, grok_ready: bool) -> Result<()> {
        if self.model_family == "grok" && names_grok_1(&self.model_id_or_path) && !grok_ready {
            return Err(self.invalid(
                "selects grok-1 but GROK1_ARTIFACT_READY is not set to 1 (set the flag, or give the run a skip_reason)",
            ));
        }
        if !KNOWN_FAMILIES.contains(&self.model_family.as_str()) {
            return Err(self.invalid(format!("uses unknown model_family '{}'", self.model_family)));
        }
        Ok(())
    }
}

/// `true` when any path segment of `model_id_or_path` is `grok-1` itself or
/// `grok-1` followed by a non-digit (`grok-1-int8`, `grok-1.gguf`), so
/// `grok-10` / `grok-11` are not gated while `xai-org/grok-1/ckpt-0` (the
/// real checkpoint layout) still is. Case-insensitive.
fn names_grok_1(model_id_or_path: &str) -> bool {
    model_id_or_path.split(['/', '\\']).any(|segment| {
        match segment.to_ascii_lowercase().strip_prefix("grok-1") {
            Some(rest) => rest.chars().next().is_none_or(|c| !c.is_ascii_digit()),
            None => false,
        }
    })
}

#[cfg(test)]
mod tests {
    use super::{ExperimentManifest, ExperimentMetrics, ExperimentSummary, RunEntry, RunMatrix};

    fn sample_run(run_id: &str, model_family: &str, model_id_or_path: &str) -> RunEntry {
        RunEntry {
            run_id: run_id.into(),
            model_family: model_family.into(),
            model_id_or_path: model_id_or_path.into(),
            source_format: "gguf".into(),
            quant_mode: "q8_0".into(),
            saaq_formula_version: "saaq_v1_5".into(),
            router_policy: "protect".into(),
            norm_policy: "protect".into(),
            telemetry_source: "synthetic".into(),
            output_root: "runs/x".into(),
            max_runtime_minutes: 30,
            max_disk_gib: 10,
            expected_artifacts: Vec::new(),
            success_metrics: Vec::new(),
            skip_reason: None,
        }
    }

    fn matrix(runs: Vec<RunEntry>) -> RunMatrix {
        RunMatrix { runs }
    }

    fn error_text(result: crate::error::Result<()>) -> String {
        result.expect_err("validation should fail").to_string()
    }

    #[test]
    fn run_matrix_accepts_well_formed_plan() {
        matrix(vec![sample_run("olmoe_q8", "olmoe", "allenai/OLMoE")])
            .validate()
            .expect("valid plan");
    }

    #[test]
    fn run_matrix_rejects_zero_runtime_budget() {
        let mut run = sample_run("r", "olmoe", "allenai/OLMoE");
        run.max_runtime_minutes = 0;
        let message = error_text(matrix(vec![run]).validate());
        assert!(message.contains("max_runtime_minutes"), "{message}");
        assert!(message.contains("not enforced"), "{message}");
    }

    #[test]
    fn run_matrix_rejects_zero_disk_budget() {
        let mut run = sample_run("r", "olmoe", "allenai/OLMoE");
        run.max_disk_gib = 0;
        let message = error_text(matrix(vec![run]).validate());
        assert!(message.contains("max_disk_gib"), "{message}");
    }

    #[test]
    fn run_matrix_rejects_blank_policies() {
        let mut run = sample_run("r", "olmoe", "allenai/OLMoE");
        run.router_policy = "  ".into();
        assert!(error_text(matrix(vec![run]).validate()).contains("router_policy"));
        let mut run = sample_run("r", "olmoe", "allenai/OLMoE");
        run.norm_policy = String::new();
        assert!(error_text(matrix(vec![run]).validate()).contains("norm_policy"));
    }

    #[test]
    fn run_matrix_rejects_unknown_family() {
        let message =
            error_text(matrix(vec![sample_run("r", "mixtral", "mistralai/Mixtral")]).validate());
        assert!(
            message.contains("unknown model_family 'mixtral'"),
            "{message}"
        );
    }

    #[test]
    fn run_matrix_gates_grok_1_but_not_other_grok_runs() {
        // Flag injected via `validate_with` so both branches run regardless
        // of the real GROK1_ARTIFACT_READY in this process.
        let grok_1 = matrix(vec![sample_run("grok_1", "grok", "xai-org/grok-1")]);
        let message = error_text(grok_1.validate_with(false));
        assert!(message.contains("GROK1_ARTIFACT_READY"), "{message}");
        grok_1
            .validate_with(true)
            .expect("grok-1 passes once the flag is set");

        // Same family, different model: never gated.
        for id in [
            "TevunahAi/grok-2-FP8",
            "xai-org/grok-10",
            "xai-org/grok-11-int8",
        ] {
            matrix(vec![sample_run("other", "grok", id)])
                .validate_with(false)
                .unwrap_or_else(|e| panic!("{id} must not be gated: {e}"));
        }
        // Gate also matches grok-1 with a non-digit suffix, a trailing slash,
        // or as an intermediate directory (the real xai-org/grok-1 layout
        // keeps weights under ckpt-0/).
        for id in [
            "xai-org/grok-1-int8",
            "xai-org/grok-1/",
            "/models/Grok-1.gguf",
            "/mnt/models/grok-1/ckpt-0",
            "xai-org/grok-1/ckpt-0",
        ] {
            let message =
                error_text(matrix(vec![sample_run("g1", "grok", id)]).validate_with(false));
            assert!(message.contains("GROK1_ARTIFACT_READY"), "{id}: {message}");
        }
        // The family check still applies: a non-grok family naming grok-1 is
        // not gated by this rule.
        matrix(vec![sample_run("mislabelled", "olmoe", "xai-org/grok-1")])
            .validate_with(false)
            .expect("gate is family-scoped");
    }

    #[test]
    fn names_grok_1_matches_only_grok_1() {
        for id in [
            "xai-org/grok-1",
            "grok-1",
            "xai-org/grok-1-int8",
            "xai-org/GROK-1",
            "grok-1/sub",
            "xai-org/grok-1/ckpt-0",
            "C:\\models\\grok-1\\ckpt-0",
        ] {
            assert!(super::names_grok_1(id), "{id}");
        }
        for id in [
            "xai-org/grok-10",
            "xai-org/grok-11",
            "xai-org/grok-2",
            "xai-org/grok-10/ckpt-0",
            "grok-12/grok-13",
        ] {
            assert!(!super::names_grok_1(id), "{id}");
        }
    }

    #[test]
    fn run_matrix_rejects_blank_skip_reason() {
        for blank in ["", "   "] {
            let mut run = sample_run("r", "olmoe", "allenai/OLMoE");
            run.skip_reason = Some(blank.into());
            let message = error_text(matrix(vec![run]).validate_with(false));
            assert!(message.contains("blank skip_reason"), "{message}");
        }
    }

    #[test]
    fn run_matrix_skip_reason_bypasses_every_check() {
        let mut run = sample_run("broken", "not_a_family", "xai-org/grok-1");
        run.max_runtime_minutes = 0;
        run.max_disk_gib = 0;
        run.router_policy = String::new();
        run.skip_reason = Some("documented skip".into());
        matrix(vec![run])
            .validate()
            .expect("skipped runs bypass validation");
    }

    #[test]
    fn shipped_vultr_matrix_validates_on_a_clean_checkout() {
        let text = include_str!("../../experiments/vultr-2026-05-28/matrix.toml");
        let matrix: RunMatrix = toml::from_str(text).expect("matrix parses");
        matrix
            .validate_with(false)
            .expect("shipped matrix validates without GROK1_ARTIFACT_READY");
        let grok_1 = matrix
            .runs
            .iter()
            .find(|run| run.model_id_or_path.contains("grok-1"))
            .expect("grok-1 entry present");
        assert!(
            grok_1.skip_reason.is_some(),
            "grok-1 must carry a skip_reason"
        );
    }

    fn sample_manifest(
        lineup_declared_count: Option<usize>,
        lineup_resolved_count: Option<usize>,
    ) -> ExperimentManifest {
        ExperimentManifest {
            run_id: "t".into(),
            run_tag: None,
            created_at: "now".into(),
            repo: "corinth-canal".into(),
            commit_sha: None,
            model_slug: "olmoe".into(),
            model_family: "Olmoe".into(),
            architecture: "MoE".into(),
            checkpoint_path: "model.gguf".into(),
            checkpoint_format: "gguf".into(),
            routing_tensor_name: "ffn_gate_inp".into(),
            synapse_source: "real".into(),
            prompt_embedding_source: "synthetic".into(),
            prompt_profile: "default".into(),
            prompt_text: None,
            ticks: 1,
            saaq_rule: "SaaqV1_5SqrtRate".into(),
            saaq_primary_rule: "SaaqV1_5SqrtRate".into(),
            saaq_dual_emit: true,
            telemetry_source: "synthetic".into(),
            telemetry_csv_path: None,
            telemetry_row_count: None,
            wraparound_enabled: false,
            wraparound_loops: 0,
            ticks_effective: 1,
            run_dir: "artifacts/x".into(),
            output_root: "artifacts".into(),
            repeat_idx: 0,
            repeat_count: 1,
            validation_status: "completed".into(),
            error: None,
            routing_mode: None,
            projection_mode: None,
            lineup_declared_count,
            lineup_resolved_count,
            generated_files: Vec::new(),
        }
    }

    fn sample_summary(projection_mode: Option<String>) -> ExperimentSummary {
        ExperimentSummary {
            run_id: "t".into(),
            run_tag: None,
            model_slug: "olmoe".into(),
            model_family: "Olmoe".into(),
            telemetry_source: "synthetic".into(),
            repeat_idx: 0,
            repeat_count: 1,
            saaq_rule: "SaaqV1_5SqrtRate".into(),
            projection_mode,
            validation_status: "completed".into(),
            run_dir: "artifacts/x".into(),
            manifest_path: "artifacts/x/run_manifest.json".into(),
            tick_telemetry_path: "artifacts/x/tick_telemetry.txt".into(),
            latent_telemetry_path: "artifacts/x/latent_telemetry.csv".into(),
            metrics: ExperimentMetrics::default(),
            repeat_determinism: None,
        }
    }

    #[test]
    fn experiment_summary_stamps_projection_mode() {
        let summary = sample_summary(Some("rate_sum".into()));
        let json = serde_json::to_value(&summary).expect("serialize summary");
        assert_eq!(json["projection_mode"], "rate_sum");
    }

    #[test]
    fn experiment_summary_missing_projection_mode_deserializes() {
        let summary = sample_summary(Some("spiking_ternary".into()));
        let json = serde_json::to_value(&summary).expect("serialize summary");
        let mut obj = json
            .as_object()
            .cloned()
            .expect("summary serializes as object");
        obj.remove("projection_mode");
        let parsed: ExperimentSummary = serde_json::from_value(serde_json::Value::Object(obj))
            .expect("old summary without field");
        assert_eq!(parsed.projection_mode, None);
    }

    #[test]
    fn experiment_manifest_stamps_lineup_counts() {
        let manifest = sample_manifest(Some(8), Some(1));
        let json = serde_json::to_value(&manifest).expect("serialize manifest");
        assert_eq!(json["lineup_declared_count"], 8);
        assert_eq!(json["lineup_resolved_count"], 1);
    }

    #[test]
    fn experiment_manifest_omits_lineup_counts_when_unset() {
        let manifest = sample_manifest(None, None);
        let json = serde_json::to_value(&manifest).expect("serialize manifest");
        assert!(json.get("lineup_declared_count").is_none());
        assert!(json.get("lineup_resolved_count").is_none());
    }

    #[test]
    fn experiment_manifest_missing_lineup_counts_deserializes() {
        let manifest = sample_manifest(Some(8), Some(1));
        let json = serde_json::to_value(&manifest).expect("serialize manifest");
        let mut obj = json
            .as_object()
            .cloned()
            .expect("manifest serializes as object");
        obj.remove("lineup_declared_count");
        obj.remove("lineup_resolved_count");
        let parsed: ExperimentManifest = serde_json::from_value(serde_json::Value::Object(obj))
            .expect("old manifest without lineup counts");
        assert_eq!(parsed.lineup_declared_count, None);
        assert_eq!(parsed.lineup_resolved_count, None);
    }
}
