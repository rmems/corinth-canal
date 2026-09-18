// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Spikenaut-SNN-Telemetry JSONL → canonical corinth replay CSV.
//!
//! Maps published Vault / Hub JSONL (`gpu_telemetry`, `mining`, `hft`,
//! `qubic_ticks`, v3 `state_telemetry`) onto [`corinth_canal::TelemetrySnapshot`].
//! Field mapping is documented in `docs/SPIKENAUT_TELEMETRY.md`.
//!
//! Missing stays missing: a null or absent required channel drops the row
//! rather than filling `0.0`. GPU captures have no wall clock — `row_index`
//! is used as an ordinal `timestamp_ms`, never a fabricated ISO datetime.

use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;

use serde_json::{Map, Value};

/// Must match [`super::telemetry_csv::TELEMETRY_CSV_HEADER`] when both modules
/// are loaded from `examples/support/mod.rs`. Duplicated so this file stays
/// `#[path]`-includable as a standalone test module.
pub const SPIKENAUT_CSV_HEADER: &str =
    "timestamp_ms,gpu_temp_c,gpu_power_w,cpu_tctl_c,cpu_package_power_w";

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
        match value.trim().to_ascii_lowercase().as_str() {
            "gpu" | "gpu_telemetry" | "neuromorphic" | "neuromorphic_data" => Some(Self::Gpu),
            "mining" | "node_sync" | "node_sync_harvest" => Some(Self::Mining),
            "hft" | "ghost" | "ghost_market" | "ghost_market_log" => Some(Self::Hft),
            "qubic" | "qubic_ticks" | "qubic_ticks_snn" => Some(Self::Qubic),
            "state" | "state_telemetry" | "v3" => Some(Self::State),
            "auto" => None,
            _ => None,
        }
    }
}

/// Outcome of ingesting one JSONL file.
#[derive(Debug, Clone)]
pub struct IngestResult {
    pub domain: SpikenautDomain,
    pub rows: Vec<corinth_canal::TelemetrySnapshot>,
    pub skipped_malformed: usize,
    pub skipped_unmapped: usize,
}

/// Detect domain from object keys. Nested `telemetry` objects are flattened
/// first so raw Vault envelopes still classify.
pub fn detect_domain(object: &Map<String, Value>) -> Option<SpikenautDomain> {
    let flat = flatten_telemetry_object(object);
    if flat.contains_key("gpu_util_pct")
        || (flat.contains_key("cpu_temp_c") && flat.contains_key("board_power_w"))
        || (flat.contains_key("episode_id") && flat.contains_key("step_idx"))
    {
        return Some(SpikenautDomain::State);
    }
    if flat.contains_key("portfolio_value")
        || flat.contains_key("ch2_mode")
        || (flat.contains_key("trade_value_usdt") && flat.contains_key("action"))
    {
        return Some(SpikenautDomain::Hft);
    }
    if flat.contains_key("tick")
        && flat.contains_key("tick_rate")
        && !flat.contains_key("hashrate_mh")
    {
        return Some(SpikenautDomain::Qubic);
    }
    if flat.contains_key("hashrate_mh") {
        return Some(SpikenautDomain::Mining);
    }
    if flat.contains_key("gpu_temp_c")
        && (flat.contains_key("power_w") || flat.contains_key("gpu_power_w"))
    {
        return Some(SpikenautDomain::Gpu);
    }
    None
}

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

/// Stream a JSONL file into snapshots. Domain is taken from `forced` or
/// inferred from the first object that classifies.
pub fn ingest_jsonl(
    path: &Path,
    forced: Option<SpikenautDomain>,
    limit: Option<usize>,
) -> Result<IngestResult, Box<dyn std::error::Error>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut domain = forced;
    let mut rows = Vec::new();
    let mut skipped_malformed = 0usize;
    let mut skipped_unmapped = 0usize;
    let mut ordinal = 0u64;

    for raw in reader.lines() {
        if limit.is_some_and(|n| rows.len() >= n) {
            break;
        }
        let line = raw?;
        if line.trim().is_empty() {
            ordinal += 1;
            continue;
        }
        let value: Value = match serde_json::from_str(&line) {
            Ok(v) => v,
            Err(_) => {
                skipped_malformed += 1;
                ordinal += 1;
                continue;
            }
        };
        let Some(object) = value.as_object() else {
            skipped_malformed += 1;
            ordinal += 1;
            continue;
        };
        if domain.is_none() {
            domain = detect_domain(object);
        }
        let Some(active) = domain else {
            skipped_unmapped += 1;
            ordinal += 1;
            continue;
        };
        match map_record(object, active, ordinal) {
            Some(snap) => rows.push(snap),
            None => skipped_unmapped += 1,
        }
        ordinal += 1;
    }

    let domain = domain.ok_or_else(|| {
        std::io::Error::other(format!(
            "could not detect Spikenaut domain from '{}'; pass --domain gpu|mining|hft|qubic|state",
            path.display()
        ))
    })?;
    Ok(IngestResult {
        domain,
        rows,
        skipped_malformed,
        skipped_unmapped,
    })
}

/// Write snapshots using the canonical five-column replay header.
pub fn write_canonical_csv(
    path: &Path,
    rows: &[corinth_canal::TelemetrySnapshot],
) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)?;
    }
    let mut writer = BufWriter::new(File::create(path)?);
    writeln!(writer, "{SPIKENAUT_CSV_HEADER}")?;
    for row in rows {
        writeln!(
            writer,
            "{},{:.6},{:.6},{:.6},{:.6}",
            row.timestamp_ms,
            row.gpu_temp_c,
            row.gpu_power_w,
            row.cpu_tctl_c,
            row.cpu_package_power_w
        )?;
    }
    writer.flush()?;
    Ok(())
}

/// CPU dual-SAAQ (1.0 + 1.5) smoke over already-mapped snapshots.
///
/// Uses the funnel + projector + stub router — no CUDA, no checkpoint.
/// Writes `run_manifest.json`, `summary.json`, `latent_telemetry.csv`, and
/// `tick_telemetry.txt` under `run_dir`.
pub fn run_dual_saaq_cpu_smoke(
    rows: &[corinth_canal::TelemetrySnapshot],
    run_dir: &Path,
    domain: SpikenautDomain,
    csv_path: Option<&Path>,
    output_root: &Path,
) -> Result<corinth_canal::ExperimentManifest, Box<dyn std::error::Error>> {
    use corinth_canal::moe::{Router, RoutingMode};
    use corinth_canal::projector::{ProjectionMode, Projector};
    use corinth_canal::types::ModelOutput;
    use corinth_canal::{
        ExperimentManifest, ExperimentMetrics, ExperimentSummary, FUNNEL_HIDDEN_NEURONS,
        SaaqUpdateRule, SnnDualLatentCalibrator, SnnLatentCsvExporter, TelemetryFunnel,
    };

    if rows.is_empty() {
        return Err(std::io::Error::other("dual-SAAQ smoke needs at least one mapped row").into());
    }
    std::fs::create_dir_all(run_dir)?;

    const THRESHOLDS: [f32; 4] = [1.0, 5.0, 1.0, 5.0];
    const SNN_STEPS: usize = 8;
    const NUM_EXPERTS: usize = 8;
    const TOP_K: usize = 2;

    let mut funnel = TelemetryFunnel::new(THRESHOLDS, SNN_STEPS);
    let mut projector = Projector::new(ProjectionMode::SpikingTernary);
    let mut router = Router::load_with_mode("", NUM_EXPERTS, TOP_K, RoutingMode::StubUniform)?;
    let mut calibrator = SnnDualLatentCalibrator::new(SaaqUpdateRule::SaaqV1_5SqrtRate);

    let latent_path = run_dir.join("latent_telemetry.csv");
    let tick_path = run_dir.join("tick_telemetry.txt");
    let manifest_path = run_dir.join("run_manifest.json");
    let summary_path = run_dir.join("summary.json");

    let mut exporter = SnnLatentCsvExporter::create(&latent_path)?;
    let mut tick_writer = BufWriter::new(File::create(&tick_path)?);
    writeln!(
        tick_writer,
        "tick,timestamp_ms,gpu_temp_c,gpu_power_w,cpu_tctl_c,cpu_package_power_w,ternary"
    )?;

    let mut metrics = ExperimentMetrics {
        ticks_completed: 0,
        latent_rows: 0,
        ..ExperimentMetrics::default()
    };

    for (tick, snap) in rows.iter().enumerate() {
        let activity = funnel.encode_snapshot(snap);
        let embedding = projector.project(
            &activity.spike_train,
            &activity.potentials,
            &activity.iz_potentials,
        )?;
        let routed = router.forward(&embedding)?;
        let output = ModelOutput {
            spike_train: activity.spike_train.clone(),
            firing_rates: vec![0.0; FUNNEL_HIDDEN_NEURONS],
            membrane_potentials: activity.potentials.clone(),
            embedding,
            expert_weights: Some(routed.expert_weights),
            selected_experts: Some(routed.selected_experts),
            reasoning: None,
        };
        let latent = calibrator.observe(snap, &activity, &output)?;
        exporter.write_row(&latent)?;
        writeln!(
            tick_writer,
            "{},{},{:.6},{:.6},{:.6},{:.6},{:?}",
            tick,
            snap.timestamp_ms,
            snap.gpu_temp_c,
            snap.gpu_power_w,
            snap.cpu_tctl_c,
            snap.cpu_package_power_w,
            activity.ternary_events
        )?;
        if metrics.first_timestamp_ms.is_none() {
            metrics.first_timestamp_ms = Some(snap.timestamp_ms);
        }
        metrics.last_timestamp_ms = Some(snap.timestamp_ms);
        metrics.ticks_completed += 1;
        metrics.latent_rows += 1;
    }
    exporter.flush()?;
    tick_writer.flush()?;

    let source_label = format!("csv_{}", domain.source_slug());
    let run_id = run_dir
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("spikenaut_dual_saaq_smoke")
        .to_owned();
    let generated_files = vec![
        "run_manifest.json".to_owned(),
        "summary.json".to_owned(),
        "tick_telemetry.txt".to_owned(),
        "latent_telemetry.csv".to_owned(),
    ];

    let manifest = ExperimentManifest {
        run_id: run_id.clone(),
        run_tag: Some(format!("spikenaut_{}", domain.as_str())),
        created_at: format_system_time_rfc3339_utc(std::time::SystemTime::now()),
        repo: "corinth-canal".to_owned(),
        commit_sha: None,
        model_slug: "stub_olmoe".to_owned(),
        model_family: "Olmoe".to_owned(),
        architecture: "stub".to_owned(),
        checkpoint_path: String::new(),
        checkpoint_format: "gguf".to_owned(),
        routing_tensor_name: "synthetic".to_owned(),
        synapse_source: "synthetic-fallback".to_owned(),
        prompt_embedding_source: "none".to_owned(),
        prompt_profile: "spikenaut_ingest".to_owned(),
        prompt_text: None,
        ticks: rows.len(),
        saaq_rule: "SaaqV1_5SqrtRate".to_owned(),
        saaq_primary_rule: "SaaqV1_5SqrtRate".to_owned(),
        saaq_dual_emit: true,
        telemetry_source: source_label.clone(),
        telemetry_csv_path: csv_path.map(|p| p.to_string_lossy().into_owned()),
        telemetry_row_count: Some(rows.len()),
        wraparound_enabled: false,
        wraparound_loops: 0,
        ticks_effective: rows.len(),
        run_dir: run_dir.to_string_lossy().into_owned(),
        output_root: output_root.to_string_lossy().into_owned(),
        repeat_idx: 0,
        repeat_count: 1,
        validation_status: "completed".to_owned(),
        error: None,
        routing_mode: Some("stub_uniform".to_owned()),
        projection_mode: Some("spiking_ternary".to_owned()),
        lineup_declared_count: None,
        lineup_resolved_count: None,
        generated_files,
    };
    std::fs::write(&manifest_path, serde_json::to_string_pretty(&manifest)?)?;

    let summary = ExperimentSummary {
        run_id,
        run_tag: manifest.run_tag.clone(),
        model_slug: manifest.model_slug.clone(),
        model_family: manifest.model_family.clone(),
        telemetry_source: source_label,
        repeat_idx: 0,
        repeat_count: 1,
        saaq_rule: manifest.saaq_rule.clone(),
        projection_mode: manifest.projection_mode.clone(),
        validation_status: "completed".to_owned(),
        run_dir: run_dir.to_string_lossy().into_owned(),
        manifest_path: manifest_path.to_string_lossy().into_owned(),
        tick_telemetry_path: tick_path.to_string_lossy().into_owned(),
        latent_telemetry_path: latent_path.to_string_lossy().into_owned(),
        metrics,
        repeat_determinism: None,
    };
    std::fs::write(&summary_path, serde_json::to_string_pretty(&summary)?)?;
    Ok(manifest)
}

fn flatten_telemetry_object(object: &Map<String, Value>) -> Map<String, Value> {
    let mut flat = object.clone();
    if let Some(Value::Object(inner)) = object.get("telemetry") {
        for (key, value) in inner {
            flat.entry(key.clone()).or_insert_with(|| value.clone());
        }
    }
    flat
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

fn timestamp_or_ordinal(fields: &Map<String, Value>, ordinal: u64) -> u64 {
    if let Some(ms) = parse_timestamp_field(fields.get("timestamp")) {
        return ms;
    }
    if let Some(ms) = parse_timestamp_field(fields.get("ts_utc")) {
        return ms;
    }
    if let Some(ms) = integer_field(fields, "row_index") {
        return ms;
    }
    if let Some(ms) = integer_field(fields, "step_idx") {
        return ms;
    }
    ordinal
}

fn parse_timestamp_field(value: Option<&Value>) -> Option<u64> {
    match value {
        Some(Value::Number(n)) => numeric_timestamp_ms(n.as_f64()?),
        Some(Value::String(s)) => parse_timestamp_string(s),
        _ => None,
    }
}

fn numeric_timestamp_ms(value: f64) -> Option<u64> {
    if !value.is_finite() || value < 0.0 {
        return None;
    }
    // ns since epoch (~1e18 in 2026), ms (~1e12), seconds (~1e9).
    let ms = if value >= 1.0e16 {
        value / 1.0e6
    } else if value >= 1.0e12 {
        value
    } else if value >= 1.0e9 {
        value * 1_000.0
    } else {
        // Small integers (row_index, step_idx) are ordinals, not unix time.
        value
    };
    if ms.is_finite() && ms >= 0.0 && ms <= u64::MAX as f64 {
        Some(ms as u64)
    } else {
        None
    }
}

/// Parse ISO-8601 / space-separated datetimes. Returns `None` rather than
/// inventing a clock when the string is not a datetime.
pub fn parse_timestamp_string(raw: &str) -> Option<u64> {
    let s = raw.trim();
    if s.is_empty() || s.eq_ignore_ascii_case("null") {
        return None;
    }
    // Refuse chain-tagged mining stamps such as `dynex:919876`.
    if s.contains(':') && !s.as_bytes().first().is_some_and(|b| b.is_ascii_digit()) {
        return None;
    }
    let bytes = s.as_bytes();
    if bytes.len() < 19 {
        return None;
    }
    let year: i32 = s.get(0..4)?.parse().ok()?;
    let month: u32 = s.get(5..7)?.parse().ok()?;
    let day: u32 = s.get(8..10)?.parse().ok()?;
    let sep = *s.as_bytes().get(10)?;
    if sep != b'T' && sep != b' ' {
        return None;
    }
    let hour: u32 = s.get(11..13)?.parse().ok()?;
    let minute: u32 = s.get(14..16)?.parse().ok()?;
    let second: u32 = s.get(17..19)?.parse().ok()?;
    if !(1..=12).contains(&month)
        || day == 0
        || day > days_in_month(year, month)
        || hour > 23
        || minute > 59
        || second > 60
    {
        return None;
    }

    let rest = s.get(19..).unwrap_or("");
    let (frac, tz) = split_frac_and_tz(rest);
    let mut millis: u32 = 0;
    if let Some(frac) = frac.filter(|f| !f.is_empty()) {
        let mut digits = frac.chars().take(3).collect::<String>();
        while digits.len() < 3 {
            digits.push('0');
        }
        millis = digits.parse().ok()?;
    }

    let mut unix_ms = i64::from(days_from_civil(year, month, day))
        .checked_mul(86_400_000)?
        .checked_add(i64::from(hour) * 3_600_000)?
        .checked_add(i64::from(minute) * 60_000)?
        .checked_add(i64::from(second) * 1_000)?
        .checked_add(i64::from(millis))?;
    unix_ms = apply_tz_offset(unix_ms, tz)?;
    u64::try_from(unix_ms).ok()
}

fn split_frac_and_tz(rest: &str) -> (Option<&str>, &str) {
    if rest.is_empty() {
        return (None, "");
    }
    if let Some(body) = rest.strip_prefix('.') {
        let tz_at = body.find(['Z', 'z', '+', '-']).unwrap_or(body.len());
        (Some(&body[..tz_at]), &body[tz_at..])
    } else {
        (None, rest)
    }
}

fn apply_tz_offset(unix_ms: i64, tz: &str) -> Option<i64> {
    let tz = tz.trim();
    if tz.is_empty() || tz.eq_ignore_ascii_case("Z") {
        return Some(unix_ms);
    }
    let sign = match tz.as_bytes().first()? {
        b'+' => 1i64,
        b'-' => -1i64,
        _ => return None,
    };
    let body = &tz[1..];
    let (hh, mm) = if body.len() >= 5 && body.as_bytes().get(2) == Some(&b':') {
        (
            body.get(0..2)?.parse::<i64>().ok()?,
            body.get(3..5)?.parse::<i64>().ok()?,
        )
    } else if body.len() >= 4 {
        (
            body.get(0..2)?.parse::<i64>().ok()?,
            body.get(2..4)?.parse::<i64>().ok()?,
        )
    } else if body.len() >= 2 {
        (body.get(0..2)?.parse::<i64>().ok()?, 0)
    } else {
        return None;
    };
    if !(0..=23).contains(&hh) || !(0..=59).contains(&mm) {
        return None;
    }
    unix_ms.checked_sub(sign * (hh * 3_600_000 + mm * 60_000))
}

fn days_in_month(year: i32, month: u32) -> u32 {
    match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 if is_leap_year(year) => 29,
        2 => 28,
        _ => 0,
    }
}

fn is_leap_year(year: i32) -> bool {
    year % 4 == 0 && (year % 100 != 0 || year % 400 == 0)
}

fn format_system_time_rfc3339_utc(now: std::time::SystemTime) -> String {
    let dur = now
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    let secs = dur.as_secs();
    let millis = dur.subsec_millis();
    let days = i32::try_from(secs / 86_400).unwrap_or(0);
    let tod = secs % 86_400;
    let (year, month, day) = civil_from_days(days);
    format!(
        "{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}.{millis:03}Z",
        tod / 3600,
        (tod % 3600) / 60,
        tod % 60,
    )
}

/// Inverse of [`days_from_civil`] (Howard Hinnant).
fn civil_from_days(z: i32) -> (i32, u32, u32) {
    let z = z + 719_468;
    let era = z.div_euclid(146_097);
    let doe = u32::try_from(z - era * 146_097).unwrap_or(0);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = i32::try_from(yoe).unwrap_or(0) + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = if month <= 2 { y + 1 } else { y };
    (year, month, day)
}

/// Howard Hinnant's public-domain `days_from_civil`.
fn days_from_civil(year: i32, month: u32, day: u32) -> i32 {
    let y = if month <= 2 { year - 1 } else { year };
    let era = y.div_euclid(400);
    let yoe = (y - era * 400) as u32;
    let mp = if month > 2 { month - 3 } else { month + 9 };
    let doy = (153 * mp + 2) / 5 + day - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe as i32 - 719_468
}

fn finite_field(fields: &Map<String, Value>, key: &str) -> Option<f32> {
    finite_value(fields.get(key)?)
}

fn first_finite(fields: &Map<String, Value>, keys: &[&str]) -> Option<f32> {
    keys.iter().find_map(|key| finite_field(fields, key))
}

fn integer_field(fields: &Map<String, Value>, key: &str) -> Option<u64> {
    match fields.get(key)? {
        Value::Number(n) => {
            if let Some(u) = n.as_u64() {
                return Some(u);
            }
            let f = n.as_f64()?;
            if f.is_finite() && f >= 0.0 && f == f.trunc() && f <= u64::MAX as f64 {
                Some(f as u64)
            } else {
                None
            }
        }
        Value::String(s) => s.parse().ok(),
        _ => None,
    }
}

fn finite_value(value: &Value) -> Option<f32> {
    match value {
        Value::Number(n) => {
            let parsed = n.as_f64()? as f32;
            parsed.is_finite().then_some(parsed)
        }
        Value::String(s) => {
            let parsed: f32 = s.parse().ok()?;
            parsed.is_finite().then_some(parsed)
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn object(value: Value) -> Map<String, Value> {
        value.as_object().cloned().expect("object")
    }

    #[test]
    fn csv_header_matches_canonical_replay_contract() {
        assert_eq!(
            SPIKENAUT_CSV_HEADER,
            "timestamp_ms,gpu_temp_c,gpu_power_w,cpu_tctl_c,cpu_package_power_w"
        );
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

    #[test]
    fn parse_space_separated_mining_timestamp() {
        let ms = parse_timestamp_string("2026-03-19 11:55:13.132").unwrap();
        assert_eq!(ms, 1_773_921_313_132);
    }

    #[test]
    fn parse_timestamp_refuses_chain_tag() {
        assert!(parse_timestamp_string("dynex:919876").is_none());
    }

    #[test]
    fn parse_timestamp_rejects_nonexistent_calendar_dates() {
        assert!(parse_timestamp_string("2026-02-30T12:00:00Z").is_none());
        assert!(parse_timestamp_string("2026-04-31T12:00:00Z").is_none());
        assert!(parse_timestamp_string("2026-02-29T12:00:00Z").is_none());
        assert!(parse_timestamp_string("2024-02-29T12:00:00Z").is_some());
    }

    #[test]
    fn parse_timestamp_rejects_out_of_range_offsets() {
        assert!(parse_timestamp_string("2026-03-19T12:00:00+99:99").is_none());
        assert!(parse_timestamp_string("2026-03-19T12:00:00+24:00").is_none());
        assert!(parse_timestamp_string("2026-03-19T12:00:00-00:60").is_none());
        assert!(parse_timestamp_string("2026-03-19T12:00:00+00:00").is_some());
    }

    #[test]
    fn format_system_time_rfc3339_utc_epoch() {
        assert_eq!(
            format_system_time_rfc3339_utc(std::time::UNIX_EPOCH),
            "1970-01-01T00:00:00.000Z"
        );
        assert_eq!(parse_timestamp_string("1970-01-01T00:00:00.000Z"), Some(0));
    }

    #[test]
    fn ingest_advances_ordinal_past_blank_and_malformed_lines() {
        let path = {
            let dir = std::env::var_os("CARGO_TARGET_TMPDIR")
                .map(std::path::PathBuf::from)
                .unwrap_or_else(|| std::path::PathBuf::from("target").join("tmp-tests"));
            let _ = std::fs::create_dir_all(&dir);
            let path = dir.join(format!(
                "spikenaut_ordinal_{}.jsonl",
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos()
            ));
            std::fs::write(
                &path,
                concat!(
                    "\n",
                    "{not json}\n",
                    "{\"hashrate_mh\":0.85,\"power_w\":375.8,\"gpu_temp_c\":80.0,\"reward_hint\":0.94,\"timestamp\":null}\n",
                ),
            )
            .unwrap();
            path
        };
        let ingested = ingest_jsonl(&path, Some(SpikenautDomain::Mining), None).unwrap();
        assert_eq!(ingested.skipped_malformed, 1);
        assert_eq!(ingested.rows.len(), 1);
        assert_eq!(
            ingested.rows[0].timestamp_ms, 2,
            "blank + malformed must consume ordinals 0 and 1"
        );
        let _ = std::fs::remove_file(path);
    }
}
