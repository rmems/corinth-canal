// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Per-domain JSON object → [`corinth_canal::TelemetrySnapshot`] mapping.

use serde_json::{Map, Value};

use super::domain::SpikenautDomain;
use super::fields::{finite_field, first_finite, flatten_telemetry_object};
use super::timestamp::timestamp_or_ordinal;

/// Map one JSON object onto a snapshot. `ordinal` is the 0-based emitted-row
/// candidate (file order, including skipped lines) used only when the source
/// has no honest timestamp.
pub fn map_record(
    object: &Map<String, Value>,
    domain: SpikenautDomain,
    ordinal: u64,
) -> Option<corinth_canal::TelemetrySnapshot> {
    let fields = flatten_telemetry_object(object);
    match domain {
        SpikenautDomain::Gpu => map_gpu(&fields, ordinal),
        SpikenautDomain::Mining => map_mining(&fields, ordinal),
        SpikenautDomain::Hft => map_hft(&fields, ordinal),
        SpikenautDomain::Qubic => map_qubic(&fields, ordinal),
        SpikenautDomain::State => map_state(&fields, ordinal),
    }
}

fn map_gpu(fields: &Map<String, Value>, ordinal: u64) -> Option<corinth_canal::TelemetrySnapshot> {
    let gpu_temp_c = finite_field(fields, "gpu_temp_c")?;
    let gpu_power_w = first_finite(fields, &["power_w", "gpu_power_w"])?;
    let cpu_tctl_c = first_finite(fields, &["vram_temp_c", "cpu_tctl_c"])?;
    let cpu_package_power_w = first_finite(fields, &["mem_util_pct", "cpu_package_power_w"])?;
    Some(corinth_canal::TelemetrySnapshot {
        timestamp_ms: timestamp_or_ordinal(fields, ordinal),
        gpu_temp_c,
        gpu_power_w,
        cpu_tctl_c,
        cpu_package_power_w,
    })
}

fn map_mining(
    fields: &Map<String, Value>,
    ordinal: u64,
) -> Option<corinth_canal::TelemetrySnapshot> {
    let gpu_temp_c = finite_field(fields, "gpu_temp_c")?;
    let gpu_power_w = first_finite(fields, &["power_w", "gpu_power_w"])?;
    let hashrate_mh = finite_field(fields, "hashrate_mh")?;
    let reward_hint = finite_field(fields, "reward_hint")?;
    Some(corinth_canal::TelemetrySnapshot {
        timestamp_ms: timestamp_or_ordinal(fields, ordinal),
        gpu_temp_c,
        gpu_power_w,
        // hashrate is normalized 0–1 on the published mining contract.
        cpu_tctl_c: hashrate_mh * 80.0,
        cpu_package_power_w: reward_hint * 200.0,
    })
}

fn map_hft(fields: &Map<String, Value>, ordinal: u64) -> Option<corinth_canal::TelemetrySnapshot> {
    let portfolio_value = finite_field(fields, "portfolio_value")?;
    let trade_value_usdt = finite_field(fields, "trade_value_usdt")?;
    let cumulative_pnl = finite_field(fields, "cumulative_pnl")?;
    let price_usd = finite_field(fields, "price_usd")?;
    Some(corinth_canal::TelemetrySnapshot {
        timestamp_ms: timestamp_or_ordinal(fields, ordinal),
        // Axon-style affine surrogates: 1-unit move ≈ encoder threshold.
        gpu_temp_c: portfolio_value,
        gpu_power_w: trade_value_usdt * 100.0,
        cpu_tctl_c: 70.0 + cumulative_pnl * 10.0,
        cpu_package_power_w: price_usd / 100.0,
    })
}

fn map_qubic(
    fields: &Map<String, Value>,
    ordinal: u64,
) -> Option<corinth_canal::TelemetrySnapshot> {
    // Independent signals only. Never consume `*_derived` as if measured.
    let tick_trace = finite_field(fields, "qubic_tick_trace")?;
    let tick_rate = finite_field(fields, "tick_rate")?;
    Some(corinth_canal::TelemetrySnapshot {
        timestamp_ms: timestamp_or_ordinal(fields, ordinal),
        gpu_temp_c: tick_trace * 100.0,
        gpu_power_w: tick_rate * 400.0,
        cpu_tctl_c: tick_trace * 80.0,
        cpu_package_power_w: tick_rate * 200.0,
    })
}

fn map_state(
    fields: &Map<String, Value>,
    ordinal: u64,
) -> Option<corinth_canal::TelemetrySnapshot> {
    let gpu_temp_c = finite_field(fields, "gpu_temp_c")?;
    let gpu_power_w = first_finite(fields, &["power_w", "gpu_power_w"])?;
    let cpu_tctl_c = first_finite(fields, &["cpu_temp_c", "vram_temp_c"])?;
    let cpu_package_power_w = first_finite(fields, &["board_power_w", "cpu_util_pct"])?;
    Some(corinth_canal::TelemetrySnapshot {
        timestamp_ms: timestamp_or_ordinal(fields, ordinal),
        gpu_temp_c,
        gpu_power_w,
        cpu_tctl_c,
        cpu_package_power_w,
    })
}

#[cfg(test)]
mod tests {
    use super::super::domain::detect_domain;
    use super::*;
    use serde_json::json;

    fn object(value: Value) -> Map<String, Value> {
        value.as_object().cloned().expect("object")
    }

    #[test]
    fn gpu_uses_row_index_as_ordinal_not_wall_clock() {
        let obj = object(json!({
            "gpu_temp_c": 31.0,
            "power_w": 22.502,
            "vram_temp_c": 39.0,
            "mem_util_pct": 26.0,
            "row_index": 12159
        }));
        let snap = map_record(&obj, SpikenautDomain::Gpu, 99).unwrap();
        assert_eq!(snap.timestamp_ms, 12159);
        assert!((snap.gpu_temp_c - 31.0).abs() < 1e-4);
        assert!((snap.gpu_power_w - 22.502).abs() < 1e-3);
        assert!((snap.cpu_tctl_c - 39.0).abs() < 1e-4);
        assert!((snap.cpu_package_power_w - 26.0).abs() < 1e-4);
    }

    #[test]
    fn gpu_drops_null_required_channel() {
        let obj = object(json!({
            "gpu_temp_c": 31.0,
            "power_w": null,
            "vram_temp_c": 39.0,
            "mem_util_pct": 26.0,
            "row_index": 1
        }));
        assert!(map_record(&obj, SpikenautDomain::Gpu, 0).is_none());
    }

    #[test]
    fn mining_parses_iso_timestamp_and_scales_hashrate() {
        let obj = object(json!({
            "hashrate_mh": 1.0,
            "power_w": 21.337,
            "gpu_temp_c": 77.866,
            "reward_hint": 0.05334,
            "timestamp": "2026-03-19T11:58:56.380000",
            "blockchain": null
        }));
        assert_eq!(detect_domain(&obj), Some(SpikenautDomain::Mining));
        let snap = map_record(&obj, SpikenautDomain::Mining, 0).unwrap();
        assert_eq!(snap.timestamp_ms, 1_773_921_536_380);
        assert!((snap.cpu_tctl_c - 80.0).abs() < 1e-3);
        assert!((snap.cpu_package_power_w - 10.668).abs() < 1e-2);
    }

    #[test]
    fn mining_null_timestamp_falls_back_to_ordinal() {
        let obj = object(json!({
            "hashrate_mh": 0.85,
            "power_w": 375.8,
            "gpu_temp_c": 80.0,
            "reward_hint": 0.94,
            "timestamp": null,
            "blockchain": "dynex"
        }));
        let snap = map_record(&obj, SpikenautDomain::Mining, 7).unwrap();
        assert_eq!(snap.timestamp_ms, 7);
    }

    #[test]
    fn hft_maps_non_physical_fields_through_affine_surrogate() {
        let obj = object(json!({
            "timestamp": "2026-03-11T18:28:53.899615136+00:00",
            "action": "sell",
            "price_usd": 70000.0,
            "trade_value_usdt": 0.9710481,
            "cumulative_pnl": -1.9088085,
            "portfolio_value": 495.81128
        }));
        assert_eq!(detect_domain(&obj), Some(SpikenautDomain::Hft));
        let snap = map_record(&obj, SpikenautDomain::Hft, 0).unwrap();
        assert!((snap.gpu_temp_c - 495.81128).abs() < 1e-3);
        assert!((snap.gpu_power_w - 97.10481).abs() < 1e-3);
        assert!((snap.cpu_tctl_c - (70.0 - 19.088085)).abs() < 1e-3);
        assert!((snap.cpu_package_power_w - 700.0).abs() < 1e-3);
        assert!(snap.timestamp_ms > 1_000_000_000_000);
    }

    #[test]
    fn qubic_ignores_derived_columns() {
        let obj = object(json!({
            "timestamp": "2026-03-20T09:03:49+00:00",
            "tick": 46539015,
            "tick_rate": 0.5333,
            "qubic_tick_trace": 0.5,
            "power_w_derived": 400.0,
            "gpu_temp_c_derived": 75.0
        }));
        assert_eq!(detect_domain(&obj), Some(SpikenautDomain::Qubic));
        let snap = map_record(&obj, SpikenautDomain::Qubic, 0).unwrap();
        assert!((snap.gpu_temp_c - 50.0).abs() < 1e-4);
        assert!((snap.gpu_power_w - 213.32).abs() < 1e-2);
        assert!((snap.cpu_tctl_c - 40.0).abs() < 1e-4);
        // Derived 75°C / 400W must not leak into the snapshot.
        assert!((snap.gpu_temp_c - 75.0).abs() > 1.0);
        assert!((snap.gpu_power_w - 400.0).abs() > 1.0);
    }

    #[test]
    fn state_prefers_real_cpu_and_board_channels() {
        let obj = object(json!({
            "episode_id": "gpu-000000",
            "step_idx": 3,
            "gpu_temp_c": 64.0,
            "power_w": 180.0,
            "cpu_temp_c": 71.5,
            "board_power_w": 95.0,
            "vram_temp_c": 50.0
        }));
        assert_eq!(detect_domain(&obj), Some(SpikenautDomain::State));
        let snap = map_record(&obj, SpikenautDomain::State, 99).unwrap();
        assert_eq!(snap.timestamp_ms, 3);
        assert!((snap.cpu_tctl_c - 71.5).abs() < 1e-4);
        assert!((snap.cpu_package_power_w - 95.0).abs() < 1e-4);
    }

    #[test]
    fn nested_telemetry_envelope_flattens() {
        let obj = object(json!({
            "telemetry": {
                "gpu_temp_c": 40.0,
                "power_w": 100.0,
                "vram_temp_c": 41.0,
                "mem_util_pct": 10.0,
                "row_index": 4
            }
        }));
        assert_eq!(detect_domain(&obj), Some(SpikenautDomain::Gpu));
        let snap = map_record(&obj, SpikenautDomain::Gpu, 0).unwrap();
        assert_eq!(snap.timestamp_ms, 4);
        assert!((snap.gpu_temp_c - 40.0).abs() < 1e-4);
    }
}
