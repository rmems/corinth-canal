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
//!
//! Sibling modules live in this directory. `mod.rs` stays `#[path]`-includable
//! from the CPU example and integration test.

pub mod cli;
mod csv;
mod domain;
mod fields;
mod ingest;
mod map;
mod smoke;
mod smoke_artifacts;
mod timestamp;

// Re-exports are consumed by the #[path] integration test and by callers that
// only need a subset of the facade. The ingest example uses a narrower set.
#[allow(unused_imports)]
pub use csv::{SPIKENAUT_CSV_HEADER, write_canonical_csv};
#[allow(unused_imports)]
pub use domain::{SpikenautDomain, detect_domain};
#[allow(unused_imports)]
pub use ingest::{IngestResult, ingest_jsonl};
#[allow(unused_imports)]
pub use map::map_record;
#[allow(unused_imports)]
pub use smoke::run_dual_saaq_cpu_smoke;
#[allow(unused_imports)]
pub use timestamp::parse_timestamp_string;
