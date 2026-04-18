use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::error::{HybridError, Result};
use crate::funnel::{FunnelActivity, FUNNEL_HIDDEN_NEURONS};
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
    pub saaq_delta_q_prev: f32,
    pub saaq_delta_q_target: f32,
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
            HybridError::OlmoeForward("missing expert_weights in ModelOutput".into())
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

        Ok(SnnLatentSnapshot {
            timestamp_ms: snap.timestamp_ms,
            avg_pop_firing_rate_hz,
            membrane_dv_dt,
            routing_entropy,
            saaq_delta_q_prev,
            saaq_delta_q_target,
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
            "timestamp_ms,avg_pop_firing_rate_hz,membrane_dv_dt,routing_entropy,saaq_delta_q_prev,saaq_delta_q_target"
        )?;
        Ok(Self { writer })
    }

    pub fn write_row(&mut self, snapshot: &SnnLatentSnapshot) -> Result<()> {
        writeln!(
            self.writer,
            "{},{:.6},{:.6},{:.6},{:.6},{:.6}",
            snapshot.timestamp_ms,
            snapshot.avg_pop_firing_rate_hz,
            snapshot.membrane_dv_dt,
            snapshot.routing_entropy,
            snapshot.saaq_delta_q_prev,
            snapshot.saaq_delta_q_target,
        )?;
        Ok(())
    }

    pub fn flush(&mut self) -> Result<()> {
        self.writer.flush()?;
        Ok(())
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
            })
            .unwrap();
        exporter.flush().unwrap();

        let contents = std::fs::read_to_string(&path).unwrap();
        let mut lines = contents.lines();
        assert_eq!(
            lines.next().unwrap(),
            "timestamp_ms,avg_pop_firing_rate_hz,membrane_dv_dt,routing_entropy,saaq_delta_q_prev,saaq_delta_q_target"
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
}
