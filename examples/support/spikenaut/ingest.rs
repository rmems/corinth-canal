// SPDX-License-Identifier: Apache-2.0 OR MIT
//! JSONL streaming ingest with file-order ordinals.

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use serde_json::{Map, Value};

use super::domain::{SpikenautDomain, detect_domain};
use super::map::map_record;

/// Outcome of ingesting one JSONL file.
#[derive(Debug, Clone)]
pub struct IngestResult {
    pub domain: SpikenautDomain,
    pub rows: Vec<corinth_canal::TelemetrySnapshot>,
    pub skipped_malformed: usize,
    pub skipped_unmapped: usize,
}

/// Stream a JSONL file into snapshots. Domain is taken from `forced` or
/// inferred from the first object that classifies.
pub fn ingest_jsonl(
    path: &Path,
    forced: Option<SpikenautDomain>,
    limit: Option<usize>,
) -> Result<IngestResult, Box<dyn std::error::Error>> {
    let reader = BufReader::new(File::open(path)?);
    let mut acc = IngestAccumulator::new(forced);
    for raw in reader.lines() {
        if acc.reached_limit(limit) {
            break;
        }
        acc.consume_line(&raw?)?;
    }
    acc.finish(path)
}

struct IngestAccumulator {
    domain: Option<SpikenautDomain>,
    rows: Vec<corinth_canal::TelemetrySnapshot>,
    skipped_malformed: usize,
    skipped_unmapped: usize,
    ordinal: u64,
}

impl IngestAccumulator {
    fn new(forced: Option<SpikenautDomain>) -> Self {
        Self {
            domain: forced,
            rows: Vec::new(),
            skipped_malformed: 0,
            skipped_unmapped: 0,
            ordinal: 0,
        }
    }

    fn reached_limit(&self, limit: Option<usize>) -> bool {
        limit.is_some_and(|n| self.rows.len() >= n)
    }

    fn consume_line(&mut self, line: &str) -> Result<(), Box<dyn std::error::Error>> {
        if line.trim().is_empty() {
            self.ordinal += 1;
            return Ok(());
        }
        match parse_object_line(line) {
            LineParse::Object(object) => self.consume_object(&object),
            LineParse::Malformed => {
                self.skipped_malformed += 1;
                self.ordinal += 1;
            }
        }
        Ok(())
    }

    fn consume_object(&mut self, object: &Map<String, Value>) {
        if self.domain.is_none() {
            self.domain = detect_domain(object);
        }
        let Some(active) = self.domain else {
            self.skipped_unmapped += 1;
            self.ordinal += 1;
            return;
        };
        match map_record(object, active, self.ordinal) {
            Some(snap) => self.rows.push(snap),
            None => self.skipped_unmapped += 1,
        }
        self.ordinal += 1;
    }

    fn finish(self, path: &Path) -> Result<IngestResult, Box<dyn std::error::Error>> {
        let domain = self.domain.ok_or_else(|| {
            std::io::Error::other(format!(
                "could not detect Spikenaut domain from '{}'; pass --domain gpu|mining|hft|qubic|state",
                path.display()
            ))
        })?;
        Ok(IngestResult {
            domain,
            rows: self.rows,
            skipped_malformed: self.skipped_malformed,
            skipped_unmapped: self.skipped_unmapped,
        })
    }
}

enum LineParse {
    Object(Map<String, Value>),
    Malformed,
}

fn parse_object_line(line: &str) -> LineParse {
    match serde_json::from_str::<Value>(line) {
        Ok(Value::Object(object)) => LineParse::Object(object),
        _ => LineParse::Malformed,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
