// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Capture-domain labels and key-based detection.

use serde_json::{Map, Value};

use super::fields::{flatten_telemetry_object, has_key, has_pair};

/// Domain / capture family of a Spikenaut JSONL file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpikenautDomain {
    /// `neuromorphic_data.jsonl` / Hub `gpu_telemetry`.
    Gpu,
    /// `node_sync_harvest.jsonl` / Hub `mining`.
    Mining,
    /// `ghost_market_log.jsonl` / Hub `hft`.
    Hft,
    /// `qubic_ticks_snn.jsonl` / Hub `qubic_ticks`.
    Qubic,
    /// v3 `state_telemetry` (NVML + board sensors when present).
    State,
}

const DOMAIN_ALIASES: &[(&str, SpikenautDomain)] = &[
    ("gpu", SpikenautDomain::Gpu),
    ("gpu_telemetry", SpikenautDomain::Gpu),
    ("neuromorphic", SpikenautDomain::Gpu),
    ("neuromorphic_data", SpikenautDomain::Gpu),
    ("mining", SpikenautDomain::Mining),
    ("node_sync", SpikenautDomain::Mining),
    ("node_sync_harvest", SpikenautDomain::Mining),
    ("hft", SpikenautDomain::Hft),
    ("ghost", SpikenautDomain::Hft),
    ("ghost_market", SpikenautDomain::Hft),
    ("ghost_market_log", SpikenautDomain::Hft),
    ("qubic", SpikenautDomain::Qubic),
    ("qubic_ticks", SpikenautDomain::Qubic),
    ("qubic_ticks_snn", SpikenautDomain::Qubic),
    ("state", SpikenautDomain::State),
    ("state_telemetry", SpikenautDomain::State),
    ("v3", SpikenautDomain::State),
];

impl SpikenautDomain {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Gpu => "gpu",
            Self::Mining => "mining",
            Self::Hft => "hft",
            Self::Qubic => "qubic",
            Self::State => "state",
        }
    }

    /// Directory-safe telemetry source slug stamped into manifests.
    ///
    /// Pairs with `csv_source_label`: a file named `spikenaut_gpu.csv`
    /// becomes `csv_spikenaut_gpu`.
    pub fn source_slug(self) -> &'static str {
        match self {
            Self::Gpu => "spikenaut_gpu",
            Self::Mining => "spikenaut_mining",
            Self::Hft => "spikenaut_hft",
            Self::Qubic => "spikenaut_qubic",
            Self::State => "spikenaut_state",
        }
    }

    pub fn from_alias(value: &str) -> Option<Self> {
        let key = value.trim().to_ascii_lowercase();
        DOMAIN_ALIASES
            .iter()
            .find(|(alias, _)| *alias == key)
            .map(|(_, domain)| *domain)
    }
}

/// Detect domain from object keys. Nested `telemetry` objects are flattened
/// first so raw Vault envelopes still classify.
pub fn detect_domain(object: &Map<String, Value>) -> Option<SpikenautDomain> {
    let flat = flatten_telemetry_object(object);
    [
        (is_state(&flat), SpikenautDomain::State),
        (is_hft(&flat), SpikenautDomain::Hft),
        (is_qubic(&flat), SpikenautDomain::Qubic),
        (is_mining(&flat), SpikenautDomain::Mining),
        (is_gpu(&flat), SpikenautDomain::Gpu),
    ]
    .into_iter()
    .find_map(|(matched, domain)| matched.then_some(domain))
}

fn is_state(flat: &Map<String, Value>) -> bool {
    has_key(flat, "gpu_util_pct")
        || has_pair(flat, "cpu_temp_c", "board_power_w")
        || has_pair(flat, "episode_id", "step_idx")
}

fn is_hft(flat: &Map<String, Value>) -> bool {
    has_key(flat, "portfolio_value")
        || has_key(flat, "ch2_mode")
        || has_pair(flat, "trade_value_usdt", "action")
}

fn is_qubic(flat: &Map<String, Value>) -> bool {
    has_pair(flat, "tick", "tick_rate") && !has_key(flat, "hashrate_mh")
}

fn is_mining(flat: &Map<String, Value>) -> bool {
    has_key(flat, "hashrate_mh")
}

fn is_gpu(flat: &Map<String, Value>) -> bool {
    has_key(flat, "gpu_temp_c") && (has_key(flat, "power_w") || has_key(flat, "gpu_power_w"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn object(value: Value) -> Map<String, Value> {
        value.as_object().cloned().expect("object")
    }

    #[test]
    fn domain_aliases_cover_vault_filenames() {
        assert_eq!(
            SpikenautDomain::from_alias("neuromorphic_data"),
            Some(SpikenautDomain::Gpu)
        );
        assert_eq!(
            SpikenautDomain::from_alias("node_sync_harvest"),
            Some(SpikenautDomain::Mining)
        );
        assert_eq!(
            SpikenautDomain::from_alias("ghost_market_log"),
            Some(SpikenautDomain::Hft)
        );
        assert_eq!(
            SpikenautDomain::from_alias("qubic_ticks_snn"),
            Some(SpikenautDomain::Qubic)
        );
        assert_eq!(SpikenautDomain::from_alias("auto"), None);
    }

    #[test]
    fn detect_gpu_from_published_sample_keys() {
        let obj = object(json!({
            "gpu_temp_c": 31.0,
            "power_w": 22.5,
            "vram_temp_c": 39.0,
            "mem_util_pct": 26.0,
            "row_index": 12159
        }));
        assert_eq!(detect_domain(&obj), Some(SpikenautDomain::Gpu));
    }
}
