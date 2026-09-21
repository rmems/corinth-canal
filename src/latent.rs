// SPDX-License-Identifier: Apache-2.0 OR MIT
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::error::{HybridError, Result};
use crate::funnel::{FUNNEL_HIDDEN_NEURONS, FunnelActivity};
use crate::types::{ModelOutput, TelemetrySnapshot};

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
pub enum SaaqUpdateRule {
    #[default]
    LegacyV1_0,
    SaaqV1_5SqrtRate,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct SnnLatentSnapshot {
    pub timestamp_ms: u64,
    pub avg_pop_firing_rate_hz: f32,
    pub membrane_dv_dt: f32,
    pub routing_entropy: f32,
    /// Legacy SAAQ column carrying whichever rule is designated primary.
    /// Preserved so existing SR.jl scripts that reference this name keep
    /// working after dual-rule emission was added.
    pub saaq_delta_q_prev: f32,
    /// Legacy SAAQ column carrying whichever rule is designated primary.
    pub saaq_delta_q_target: f32,
    pub gpu_temp_c: f32,
    pub gpu_power_w: f32,
    pub cpu_tctl_c: f32,
    pub cpu_package_power_w: f32,
    /// SAAQ 1.0 (`LegacyV1_0`) previous target at this tick. Emitted by
    /// [`SnnDualLatentCalibrator`] so SR.jl can fit candidate laws against
    /// either rule without rerunning the whole sweep.
    pub saaq_delta_q_legacy_prev: f32,
    pub saaq_delta_q_legacy_target: f32,
    /// SAAQ 1.5 (`SaaqV1_5SqrtRate`) previous target at this tick.
    pub saaq_delta_q_v15_prev: f32,
    pub saaq_delta_q_v15_target: f32,
}

#[derive(Debug, Clone, Default)]
pub struct SnnLatentCalibrator {
    prev_timestamp_ms: Option<u64>,
    prev_mean_membrane: Option<f32>,
    prev_delta_q: f32,
    update_rule: SaaqUpdateRule,
}

impl SnnLatentCalibrator {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_update_rule(update_rule: SaaqUpdateRule) -> Self {
        Self {
            update_rule,
            ..Self::default()
        }
    }

    pub fn set_update_rule(&mut self, update_rule: SaaqUpdateRule) {
        self.update_rule = update_rule;
    }

    pub fn update_rule(&self) -> SaaqUpdateRule {
        self.update_rule
    }

    pub fn observe(
        &mut self,
        snap: &TelemetrySnapshot,
        activity: &FunnelActivity,
        output: &ModelOutput,
    ) -> Result<SnnLatentSnapshot> {
        let dt_ms = self.window_dt_ms(snap.timestamp_ms);
        let dt_seconds = (dt_ms as f32 / 1000.0).max(1e-3);
        let total_hidden_spikes = activity.spike_train.iter().map(Vec::len).sum::<usize>() as f32;
        let avg_pop_firing_rate_hz =
            total_hidden_spikes / FUNNEL_HIDDEN_NEURONS as f32 / dt_seconds;

        let mean_membrane = mean(&activity.potentials);
        let previous_mean_membrane = self.prev_mean_membrane.unwrap_or(mean_membrane);
        let membrane_dv_dt = (mean_membrane - previous_mean_membrane) / dt_seconds;

        let expert_weights = output.expert_weights.as_deref().ok_or_else(|| {
            HybridError::RouterForward("missing expert_weights in ModelOutput".into())
        })?;
        let routing_entropy = normalized_entropy(expert_weights);

        let saaq_delta_q_prev = self.prev_delta_q;
        let saaq_delta_q_target = match self.update_rule {
            SaaqUpdateRule::LegacyV1_0 => {
                let activity_pressure = (avg_pop_firing_rate_hz / 24.0).clamp(0.0, 1.0);
                let membrane_pressure = (membrane_dv_dt / 12.0).clamp(-1.0, 1.0);
                0.52 * saaq_delta_q_prev
                    + 0.28 * activity_pressure
                    + 0.12 * membrane_pressure
                    + 0.20 * routing_entropy
                    - 0.18
            }
            SaaqUpdateRule::SaaqV1_5SqrtRate => {
                0.0573 * avg_pop_firing_rate_hz.max(0.0).sqrt() + 0.496 * saaq_delta_q_prev
            }
        };

        self.prev_timestamp_ms = Some(snap.timestamp_ms);
        self.prev_mean_membrane = Some(mean_membrane);
        self.prev_delta_q = saaq_delta_q_target;

        // Solo calibrators only populate the legacy `saaq_delta_q_*` columns.
        // The dual-rule fields are left at their defaults (0.0) and filled in
        // by `SnnDualLatentCalibrator::observe` when both rules are run
        // together.
        Ok(SnnLatentSnapshot {
            timestamp_ms: snap.timestamp_ms,
            avg_pop_firing_rate_hz,
            membrane_dv_dt,
            routing_entropy,
            saaq_delta_q_prev,
            saaq_delta_q_target,
            gpu_temp_c: snap.gpu_temp_c,
            gpu_power_w: snap.gpu_power_w,
            cpu_tctl_c: snap.cpu_tctl_c,
            cpu_package_power_w: snap.cpu_package_power_w,
            ..Default::default()
        })
    }

    fn window_dt_ms(&self, timestamp_ms: u64) -> u64 {
        match self.prev_timestamp_ms {
            Some(prev_timestamp_ms) if timestamp_ms > prev_timestamp_ms => {
                timestamp_ms - prev_timestamp_ms
            }
            _ => 1,
        }
    }
}

#[derive(Debug)]
pub struct SnnLatentCsvExporter {
    writer: BufWriter<File>,
}

impl SnnLatentCsvExporter {
    pub fn create<P: AsRef<Path>>(path: P) -> Result<Self> {
        let mut writer = BufWriter::new(File::create(path)?);
        writeln!(
            writer,
            "timestamp_ms,avg_pop_firing_rate_hz,membrane_dv_dt,routing_entropy,saaq_delta_q_prev,saaq_delta_q_target,gpu_temp_c,gpu_power_w,cpu_tctl_c,cpu_package_power_w,saaq_delta_q_legacy_prev,saaq_delta_q_legacy_target,saaq_delta_q_v15_prev,saaq_delta_q_v15_target"
        )?;
        Ok(Self { writer })
    }

    pub fn write_row(&mut self, snapshot: &SnnLatentSnapshot) -> Result<()> {
        writeln!(
            self.writer,
            "{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6}",
            snapshot.timestamp_ms,
            snapshot.avg_pop_firing_rate_hz,
            snapshot.membrane_dv_dt,
            snapshot.routing_entropy,
            snapshot.saaq_delta_q_prev,
            snapshot.saaq_delta_q_target,
            snapshot.gpu_temp_c,
            snapshot.gpu_power_w,
            snapshot.cpu_tctl_c,
            snapshot.cpu_package_power_w,
            snapshot.saaq_delta_q_legacy_prev,
            snapshot.saaq_delta_q_legacy_target,
            snapshot.saaq_delta_q_v15_prev,
            snapshot.saaq_delta_q_v15_target,
        )?;
        Ok(())
    }

    pub fn flush(&mut self) -> Result<()> {
        self.writer.flush()?;
        Ok(())
    }
}

/// Dual-rule latent calibrator that runs `LegacyV1_0` (SAAQ 1.0) and
/// `SaaqV1_5SqrtRate` (SAAQ 1.5) in parallel over the same `(snap, activity,
/// output)` stream.
///
/// Emits a single [`SnnLatentSnapshot`] per observation with *both* SAAQ
/// trajectories populated. The `primary_rule` selects which of the two rules
/// fills the legacy `saaq_delta_q_{prev,target}` columns so existing
/// SymbolicRegression.jl scripts (which read those names) remain valid.
///
/// Why dual-emit matters: the feature columns fed to SR.jl
/// (`avg_pop_firing_rate_hz`, `membrane_dv_dt`, `routing_entropy`) are
/// computed from activity/routing only and do **not** depend on which SAAQ
/// rule is being evolved. Running two solo calibrators against identical
/// inputs therefore produces paired `(X, y_legacy)` and `(X, y_v15)` data
/// with no run-to-run noise, at the cost of a few extra arithmetic ops.
#[derive(Debug, Clone)]
pub struct SnnDualLatentCalibrator {
    legacy: SnnLatentCalibrator,
    v15: SnnLatentCalibrator,
    primary_rule: SaaqUpdateRule,
}

impl SnnDualLatentCalibrator {
    pub fn new(primary_rule: SaaqUpdateRule) -> Self {
        Self {
            legacy: SnnLatentCalibrator::with_update_rule(SaaqUpdateRule::LegacyV1_0),
            v15: SnnLatentCalibrator::with_update_rule(SaaqUpdateRule::SaaqV1_5SqrtRate),
            primary_rule,
        }
    }

    pub fn primary_rule(&self) -> SaaqUpdateRule {
        self.primary_rule
    }

    /// Observe one tick against both rules and return a merged snapshot.
    ///
    /// The returned snapshot has:
    /// - `saaq_delta_q_{legacy,v15}_{prev,target}` populated from each rule,
    /// - `saaq_delta_q_{prev,target}` populated from the rule returned by
    ///   [`primary_rule`](Self::primary_rule),
    /// - all other fields taken from the legacy calibrator (identical across
    ///   rules by construction, since they are computed from activity alone).
    pub fn observe(
        &mut self,
        snap: &TelemetrySnapshot,
        activity: &FunnelActivity,
        output: &ModelOutput,
    ) -> Result<SnnLatentSnapshot> {
        let legacy_snapshot = self.legacy.observe(snap, activity, output)?;
        let v15_snapshot = self.v15.observe(snap, activity, output)?;

        let (primary_prev, primary_target) = match self.primary_rule {
            SaaqUpdateRule::LegacyV1_0 => (
                legacy_snapshot.saaq_delta_q_prev,
                legacy_snapshot.saaq_delta_q_target,
            ),
            SaaqUpdateRule::SaaqV1_5SqrtRate => (
                v15_snapshot.saaq_delta_q_prev,
                v15_snapshot.saaq_delta_q_target,
            ),
        };

        Ok(SnnLatentSnapshot {
            timestamp_ms: legacy_snapshot.timestamp_ms,
            avg_pop_firing_rate_hz: legacy_snapshot.avg_pop_firing_rate_hz,
            membrane_dv_dt: legacy_snapshot.membrane_dv_dt,
            routing_entropy: legacy_snapshot.routing_entropy,
            saaq_delta_q_prev: primary_prev,
            saaq_delta_q_target: primary_target,
            gpu_temp_c: legacy_snapshot.gpu_temp_c,
            gpu_power_w: legacy_snapshot.gpu_power_w,
            cpu_tctl_c: legacy_snapshot.cpu_tctl_c,
            cpu_package_power_w: legacy_snapshot.cpu_package_power_w,
            saaq_delta_q_legacy_prev: legacy_snapshot.saaq_delta_q_prev,
            saaq_delta_q_legacy_target: legacy_snapshot.saaq_delta_q_target,
            saaq_delta_q_v15_prev: v15_snapshot.saaq_delta_q_prev,
            saaq_delta_q_v15_target: v15_snapshot.saaq_delta_q_target,
        })
    }
}

fn mean(values: &[f32]) -> f32 {
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f32>() / values.len() as f32
    }
}

fn normalized_entropy(weights: &[f32]) -> f32 {
    if weights.len() <= 1 {
        return 0.0;
    }

    let entropy = weights
        .iter()
        .copied()
        .filter(|weight| weight.is_finite() && *weight > 0.0)
        .map(|weight| -weight * weight.ln())
        .sum::<f32>();
    let max_entropy = (weights.len() as f32).ln();

    if max_entropy > 0.0 {
        (entropy / max_entropy).clamp(0.0, 1.0)
    } else {
        0.0
    }
}

// ---------------------------------------------------------------------------
// SAAQ -> gif_threshold control mapping (`saaq-tau-map` export)
//
// `saaq_delta_q_target` is currently telemetry-only. The pieces below close the
// loop: they interpret the SAAQ target as a *desired sparsity fraction* and map
// it to a `gif_threshold` multiplier via the empirical τ->sparsity anchors
// measured on Grok-1 embedding weights
// (`grok-ozempic/reports/grok-1-tau-sweep/results.md`; for near-Gaussian
// tensors `zeros(τ) ~= erf(τ/sqrt(2))`):
//
//   sparsity:   0.25   0.50   0.75   0.90   0.95
//   gif_threshold: 0.31 0.65  1.12   1.63   1.97
//
// The mapping is piecewise-linear through those anchors, monotone in its
// input, and clamped into `SAAQ_TAU_RANGE` so a pathological target can never
// ask the quantizer for an invalid or degenerate τ. grok-ozempic consumes the
// emitted JSON as the `corinth-canal/saaq-tau-map` schema (version 1) through
// `precision::resolve_threshold` — this file only ever *proposes* τ values;
// grok-ozempic's manifest precedence still decides what is applied.
// ---------------------------------------------------------------------------

/// Empirical (target sparsity, gif_threshold) anchor pairs used by
/// [`saaq_delta_q_to_gif_threshold`]. Sorted by sparsity.
pub const SAAQ_TAU_ANCHORS: &[(f32, f32)] = &[
    (0.0, 0.05),
    (0.25, 0.31),
    (0.50, 0.65),
    (0.75, 1.12),
    (0.90, 1.63),
    (0.95, 1.97),
];

/// Inclusive `[min, max]` `gif_threshold` range the SAAQ mapping is allowed to
/// emit. Bounds are the anchor extremes: below 0.05 the gate barely fires,
/// above 1.97 sparsity already exceeds 95% on the measured distribution.
pub const SAAQ_TAU_RANGE: (f32, f32) = (0.05, 1.97);

/// Fallback τ used when `saaq_delta_q_target` is non-finite (NaN from a
/// broken telemetry tick, etc.). Chosen as the 50%-sparsity anchor so a
/// degraded signal degrades to mid-range sparsity rather than an extreme.
pub const SAAQ_TAU_FALLBACK: f32 = 0.65;

/// Map a `saaq_delta_q_target` value to a `gif_threshold` multiplier.
///
/// `delta_q` is interpreted as a desired sparsity fraction (0 = keep almost
/// everything, ~0.95 = keep only the top ~5% of weights by magnitude); it is
/// clamped to the anchor domain before interpolating. The result is monotone
/// non-decreasing in `delta_q` and always within [`SAAQ_TAU_RANGE`]; non-finite
/// input yields [`SAAQ_TAU_FALLBACK`].
pub fn saaq_delta_q_to_gif_threshold(delta_q: f32) -> f32 {
    if !delta_q.is_finite() {
        return SAAQ_TAU_FALLBACK;
    }
    let (s_min, s_max) = (
        SAAQ_TAU_ANCHORS[0].0,
        SAAQ_TAU_ANCHORS[SAAQ_TAU_ANCHORS.len() - 1].0,
    );
    let s = delta_q.clamp(s_min, s_max);
    // Locate the anchor interval containing s. The table is tiny, so a linear
    // scan is the clearest correct implementation.
    let hi = SAAQ_TAU_ANCHORS
        .iter()
        .position(|(anchor_s, _)| s <= *anchor_s)
        .unwrap_or(SAAQ_TAU_ANCHORS.len() - 1)
        .max(1);
    let (s0, t0) = SAAQ_TAU_ANCHORS[hi - 1];
    let (s1, t1) = SAAQ_TAU_ANCHORS[hi];
    let frac = if s1 > s0 { (s - s0) / (s1 - s0) } else { 0.0 };
    (t0 + frac * (t1 - t0)).clamp(SAAQ_TAU_RANGE.0, SAAQ_TAU_RANGE.1)
}

/// One `entries[]` element of a `saaq-tau-map` JSON file.
///
/// `pattern` uses grok-ozempic's segment-anchored glob vocabulary (the same
/// one as manifest `ternary_candidates[].name`); first match wins there.
/// `tier` / `saaq_delta_q_target` are provenance only — only `pattern` and
/// `gif_threshold` are load-bearing on the consumer side.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SaaqTauMapEntry {
    pub pattern: String,
    pub gif_threshold: f32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tier: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub saaq_delta_q_target: Option<f32>,
}

/// A `corinth-canal/saaq-tau-map` (version 1) document: the per-tensor/tier τ
/// source grok-ozempic loads via `--saaq-tau-map`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SaaqTauMap {
    pub schema: String,
    pub version: u32,
    pub entries: Vec<SaaqTauMapEntry>,
}

impl SaaqTauMap {
    /// Build a map with the canonical schema tag and version.
    pub fn new(entries: Vec<SaaqTauMapEntry>) -> Self {
        Self {
            schema: "corinth-canal/saaq-tau-map".to_string(),
            version: 1,
            entries,
        }
    }
}

/// Build a single τ-map entry from a SAAQ target, applying
/// [`saaq_delta_q_to_gif_threshold`] for the τ value.
pub fn saaq_tau_entry(
    pattern: impl Into<String>,
    tier: Option<String>,
    saaq_delta_q_target: f32,
) -> SaaqTauMapEntry {
    SaaqTauMapEntry {
        pattern: pattern.into(),
        gif_threshold: saaq_delta_q_to_gif_threshold(saaq_delta_q_target),
        tier,
        saaq_delta_q_target: Some(saaq_delta_q_target),
    }
}

/// Serialize a [`SaaqTauMap`] to `path` as pretty JSON.
pub fn write_saaq_tau_map<P: AsRef<Path>>(path: P, map: &SaaqTauMap) -> Result<()> {
    let file = File::create(path)?;
    serde_json::to_writer_pretty(BufWriter::new(file), map)
        .map_err(|e| HybridError::InvalidConfig(format!("saaq-tau-map serialize failed: {e}")))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_activity(hidden_spike_count: usize, mean_potential: f32) -> FunnelActivity {
        FunnelActivity {
            ternary_events: [1, 0, -1, 1],
            input_spike_train: vec![vec![0, 1]; 4],
            spike_train: vec![vec![0; hidden_spike_count]; 4],
            potentials: vec![mean_potential; FUNNEL_HIDDEN_NEURONS],
            iz_potentials: vec![0.0; 5],
        }
    }

    fn sample_output(expert_weights: Vec<f32>) -> ModelOutput {
        ModelOutput {
            spike_train: vec![vec![0, 1]; 4],
            firing_rates: vec![0.5; FUNNEL_HIDDEN_NEURONS],
            membrane_potentials: vec![0.25; FUNNEL_HIDDEN_NEURONS],
            embedding: vec![0.0; 2048],
            expert_weights: Some(expert_weights),
            selected_experts: Some(vec![0]),
            reasoning: None,
        }
    }

    #[test]
    fn calibrator_emits_expected_latent_fields() {
        let mut calibrator = SnnLatentCalibrator::new();
        let snap_a = TelemetrySnapshot {
            timestamp_ms: 1_000,
            gpu_temp_c: 60.0,
            gpu_power_w: 250.0,
            cpu_tctl_c: 70.0,
            cpu_package_power_w: 120.0,
        };
        let snap_b = TelemetrySnapshot {
            timestamp_ms: 1_100,
            ..snap_a.clone()
        };
        let output = sample_output(vec![0.7, 0.2, 0.1]);

        let first = calibrator
            .observe(&snap_a, &sample_activity(0, 0.20), &output)
            .unwrap();
        let second = calibrator
            .observe(&snap_b, &sample_activity(8, 0.35), &output)
            .unwrap();

        assert_eq!(first.timestamp_ms, 1_000);
        assert_eq!(first.saaq_delta_q_prev, 0.0);
        assert!(first.routing_entropy > 0.0);
        assert!(second.avg_pop_firing_rate_hz > 0.0);
        assert!(second.membrane_dv_dt > 0.0);
        assert!((second.saaq_delta_q_prev - first.saaq_delta_q_target).abs() < 1e-6);
    }

    #[test]
    fn exporter_writes_expected_header_and_row_count() {
        let path = std::env::temp_dir().join(format!(
            "corinth_canal_latent_{}.csv",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));

        let mut exporter = SnnLatentCsvExporter::create(&path).unwrap();
        exporter
            .write_row(&SnnLatentSnapshot {
                timestamp_ms: 42,
                avg_pop_firing_rate_hz: 1.0,
                membrane_dv_dt: -0.5,
                routing_entropy: 0.7,
                saaq_delta_q_prev: 0.1,
                saaq_delta_q_target: 0.2,
                gpu_temp_c: 60.0,
                gpu_power_w: 250.0,
                cpu_tctl_c: 70.0,
                cpu_package_power_w: 120.0,
                ..Default::default()
            })
            .unwrap();
        exporter.flush().unwrap();

        let contents = std::fs::read_to_string(&path).unwrap();
        let mut lines = contents.lines();
        assert_eq!(
            lines.next().unwrap(),
            "timestamp_ms,avg_pop_firing_rate_hz,membrane_dv_dt,routing_entropy,saaq_delta_q_prev,saaq_delta_q_target,gpu_temp_c,gpu_power_w,cpu_tctl_c,cpu_package_power_w,saaq_delta_q_legacy_prev,saaq_delta_q_legacy_target,saaq_delta_q_v15_prev,saaq_delta_q_v15_target"
        );
        assert!(lines.next().is_some());
        assert!(lines.next().is_none());

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn saaq_v1_5_uses_sqrt_rate_equation() {
        let mut calibrator =
            SnnLatentCalibrator::with_update_rule(SaaqUpdateRule::SaaqV1_5SqrtRate);
        let snap_a = TelemetrySnapshot {
            timestamp_ms: 1_000,
            gpu_temp_c: 60.0,
            gpu_power_w: 250.0,
            cpu_tctl_c: 70.0,
            cpu_package_power_w: 120.0,
        };
        let snap_b = TelemetrySnapshot {
            timestamp_ms: 1_100,
            ..snap_a.clone()
        };
        let output = sample_output(vec![0.7, 0.2, 0.1]);

        let first = calibrator
            .observe(&snap_a, &sample_activity(0, 0.20), &output)
            .unwrap();
        let second = calibrator
            .observe(&snap_b, &sample_activity(8, 0.35), &output)
            .unwrap();

        assert_eq!(first.saaq_delta_q_target, 0.0);
        let expected =
            0.0573 * second.avg_pop_firing_rate_hz.sqrt() + 0.496 * second.saaq_delta_q_prev;
        assert!((second.saaq_delta_q_target - expected).abs() < 1e-6);
    }

    #[test]
    fn calibrator_update_rule_can_be_changed() {
        let mut calibrator = SnnLatentCalibrator::new();
        assert_eq!(calibrator.update_rule(), SaaqUpdateRule::LegacyV1_0);
        calibrator.set_update_rule(SaaqUpdateRule::SaaqV1_5SqrtRate);
        assert_eq!(calibrator.update_rule(), SaaqUpdateRule::SaaqV1_5SqrtRate);
    }

    #[test]
    fn dual_calibrator_matches_solo_calibrators_bit_for_bit() {
        let mut legacy_solo = SnnLatentCalibrator::with_update_rule(SaaqUpdateRule::LegacyV1_0);
        let mut v15_solo = SnnLatentCalibrator::with_update_rule(SaaqUpdateRule::SaaqV1_5SqrtRate);
        let mut dual = SnnDualLatentCalibrator::new(SaaqUpdateRule::SaaqV1_5SqrtRate);

        let output = sample_output(vec![0.6, 0.3, 0.1]);
        let base = TelemetrySnapshot {
            timestamp_ms: 1_000,
            gpu_temp_c: 62.0,
            gpu_power_w: 245.0,
            cpu_tctl_c: 71.0,
            cpu_package_power_w: 118.0,
        };

        for step in 0..6 {
            let snap = TelemetrySnapshot {
                timestamp_ms: 1_000 + step as u64 * 50,
                ..base.clone()
            };
            let activity = sample_activity((step * 2) as usize, 0.15 + 0.05 * step as f32);

            let legacy = legacy_solo.observe(&snap, &activity, &output).unwrap();
            let v15 = v15_solo.observe(&snap, &activity, &output).unwrap();
            let merged = dual.observe(&snap, &activity, &output).unwrap();

            assert_eq!(merged.saaq_delta_q_legacy_prev, legacy.saaq_delta_q_prev);
            assert_eq!(
                merged.saaq_delta_q_legacy_target,
                legacy.saaq_delta_q_target
            );
            assert_eq!(merged.saaq_delta_q_v15_prev, v15.saaq_delta_q_prev);
            assert_eq!(merged.saaq_delta_q_v15_target, v15.saaq_delta_q_target);
            // Primary rule is v1.5, so legacy compatibility columns should
            // track the v15 trajectory.
            assert_eq!(merged.saaq_delta_q_prev, v15.saaq_delta_q_prev);
            assert_eq!(merged.saaq_delta_q_target, v15.saaq_delta_q_target);
            // Shared feature columns are rule-independent.
            assert_eq!(merged.avg_pop_firing_rate_hz, legacy.avg_pop_firing_rate_hz);
            assert_eq!(merged.avg_pop_firing_rate_hz, v15.avg_pop_firing_rate_hz);
            assert_eq!(merged.routing_entropy, legacy.routing_entropy);
        }
    }

    #[test]
    fn dual_calibrator_primary_rule_legacy_fills_legacy_columns() {
        let mut dual = SnnDualLatentCalibrator::new(SaaqUpdateRule::LegacyV1_0);
        let output = sample_output(vec![0.5, 0.3, 0.2]);
        let snap = TelemetrySnapshot {
            timestamp_ms: 2_000,
            gpu_temp_c: 65.0,
            gpu_power_w: 240.0,
            cpu_tctl_c: 70.0,
            cpu_package_power_w: 120.0,
        };
        let merged = dual
            .observe(&snap, &sample_activity(4, 0.25), &output)
            .unwrap();
        // Legacy primary -> saaq_delta_q_{prev,target} should equal the
        // *_legacy_* fields, not the *_v15_* fields.
        assert_eq!(merged.saaq_delta_q_prev, merged.saaq_delta_q_legacy_prev);
        assert_eq!(
            merged.saaq_delta_q_target,
            merged.saaq_delta_q_legacy_target
        );
    }

    #[test]
    fn saaq_tau_mapping_hits_anchors_and_is_monotone() {
        // Anchor table passes through exactly.
        for &(s, tau) in SAAQ_TAU_ANCHORS {
            assert!(
                (saaq_delta_q_to_gif_threshold(s) - tau).abs() < 1e-6,
                "anchor {s} -> {tau}"
            );
        }
        // Monotone non-decreasing over the domain.
        let mut prev = saaq_delta_q_to_gif_threshold(-0.5);
        for i in 0..=100 {
            let s = -0.5 + i as f32 * 0.02;
            let t = saaq_delta_q_to_gif_threshold(s);
            assert!(t >= prev, "monotone violated at s={s}: {prev} -> {t}");
            assert!((SAAQ_TAU_RANGE.0..=SAAQ_TAU_RANGE.1).contains(&t));
            prev = t;
        }
        // Out-of-domain input clamps to the anchor extremes.
        assert_eq!(saaq_delta_q_to_gif_threshold(-1.0), SAAQ_TAU_RANGE.0);
        assert_eq!(saaq_delta_q_to_gif_threshold(1.5), SAAQ_TAU_RANGE.1);
        // Non-finite input degrades to the mid-range fallback.
        assert_eq!(saaq_delta_q_to_gif_threshold(f32::NAN), SAAQ_TAU_FALLBACK);
        assert_eq!(
            saaq_delta_q_to_gif_threshold(f32::INFINITY),
            SAAQ_TAU_FALLBACK
        );
    }

    #[test]
    fn saaq_tau_map_round_trips_grok_ozempic_schema() {
        let entry = saaq_tau_entry(
            "block_*.slot_00.moe_expert.gate",
            Some("moe_expert".to_string()),
            0.5,
        );
        assert!((entry.gif_threshold - 0.65).abs() < 1e-6);
        let map = SaaqTauMap::new(vec![entry]);
        let json = serde_json::to_string(&map).unwrap();
        let decoded: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded["schema"], "corinth-canal/saaq-tau-map");
        assert_eq!(decoded["version"], 1);
        assert_eq!(
            decoded["entries"][0]["pattern"],
            "block_*.slot_00.moe_expert.gate"
        );
        assert!((decoded["entries"][0]["gif_threshold"].as_f64().unwrap() - 0.65).abs() < 1e-6);

        let path = std::env::temp_dir().join(format!(
            "corinth_canal_saaq_tau_map_{}.json",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        write_saaq_tau_map(&path, &map).unwrap();
        let written: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        assert_eq!(written["entries"][0]["tier"], "moe_expert");
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn exporter_header_includes_dual_saaq_columns() {
        let path = std::env::temp_dir().join(format!(
            "corinth_canal_dual_latent_{}.csv",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let mut exporter = SnnLatentCsvExporter::create(&path).unwrap();
        exporter.flush().unwrap();
        let contents = std::fs::read_to_string(&path).unwrap();
        let header = contents.lines().next().unwrap();
        for column in [
            "saaq_delta_q_legacy_prev",
            "saaq_delta_q_legacy_target",
            "saaq_delta_q_v15_prev",
            "saaq_delta_q_v15_target",
        ] {
            assert!(header.contains(column), "missing column: {column}");
        }
        let _ = std::fs::remove_file(path);
    }
}
