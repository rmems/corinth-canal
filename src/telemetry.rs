// SPDX-License-Identifier: Apache-2.0 OR MIT
pub use crate::types::TelemetrySnapshot;

#[derive(Debug, Clone)]
pub struct TelemetryEncoder {
    baseline: [f32; 4],
    thresholds: [f32; 4],
    initialized: bool,
}

impl TelemetryEncoder {
    pub fn new(thresholds: [f32; 4]) -> Self {
        Self {
            baseline: [0.0; 4],
            thresholds,
            initialized: false,
        }
    }

    pub fn encode(&mut self, snap: &TelemetrySnapshot) -> [i8; 4] {
        let values = snap.channel_values();

        if !self.initialized {
            self.baseline = values;
            self.initialized = true;
            return [0; 4];
        }

        let mut spikes = [0; 4];
        for i in 0..values.len() {
            let delta = values[i] - self.baseline[i];
            if delta >= self.thresholds[i] {
                spikes[i] = 1;
                self.baseline[i] = values[i];
            } else if delta <= -self.thresholds[i] {
                spikes[i] = -1;
                self.baseline[i] = values[i];
            }
        }

        spikes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn snapshot(
        gpu_temp_c: f32,
        gpu_power_w: f32,
        cpu_tctl_c: f32,
        cpu_package_power_w: f32,
    ) -> TelemetrySnapshot {
        TelemetrySnapshot {
            gpu_temp_c,
            gpu_power_w,
            cpu_tctl_c,
            cpu_package_power_w,
            timestamp_ms: 0,
        }
    }

    #[test]
    fn first_encode_seeds_baseline_and_emits_zeroes() {
        let mut encoder = TelemetryEncoder::new([1.0, 5.0, 1.0, 5.0]);
        let snap = snapshot(60.0, 260.0, 77.0, 153.0);

        assert_eq!(encoder.encode(&snap), [0, 0, 0, 0]);
        assert!(encoder.initialized);
        assert_eq!(encoder.baseline, [60.0, 260.0, 77.0, 153.0]);
    }

    #[test]
    fn positive_threshold_crossing_emits_positive_spike_and_updates_baseline() {
        let mut encoder = TelemetryEncoder::new([1.0, 5.0, 1.0, 5.0]);
        encoder.encode(&snapshot(60.0, 260.0, 77.0, 153.0));

        let output = encoder.encode(&snapshot(61.0, 260.0, 77.0, 153.0));

        assert_eq!(output, [1, 0, 0, 0]);
        assert_eq!(encoder.baseline, [61.0, 260.0, 77.0, 153.0]);
    }

    #[test]
    fn negative_threshold_crossing_emits_negative_spike_and_updates_baseline() {
        let mut encoder = TelemetryEncoder::new([1.0, 5.0, 1.0, 5.0]);
        encoder.encode(&snapshot(60.0, 260.0, 77.0, 153.0));

        let output = encoder.encode(&snapshot(60.0, 254.5, 77.0, 153.0));

        assert_eq!(output, [0, -1, 0, 0]);
        assert_eq!(encoder.baseline, [60.0, 254.5, 77.0, 153.0]);
    }

    #[test]
    fn subthreshold_change_emits_zero_and_preserves_baseline() {
        let mut encoder = TelemetryEncoder::new([1.0, 5.0, 1.0, 5.0]);
        encoder.encode(&snapshot(60.0, 260.0, 77.0, 153.0));

        let output = encoder.encode(&snapshot(60.5, 264.9, 77.9, 157.9));

        assert_eq!(output, [0, 0, 0, 0]);
        assert_eq!(encoder.baseline, [60.0, 260.0, 77.0, 153.0]);
    }

    #[test]
    fn channels_use_independent_thresholds_and_baselines() {
        let mut encoder = TelemetryEncoder::new([1.0, 10.0, 0.5, 7.0]);
        encoder.encode(&snapshot(60.0, 260.0, 77.0, 153.0));

        let output = encoder.encode(&snapshot(61.0, 269.0, 76.0, 160.0));

        assert_eq!(output, [1, 0, -1, 1]);
        assert_eq!(encoder.baseline, [61.0, 260.0, 76.0, 160.0]);
    }
}
