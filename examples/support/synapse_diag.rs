// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Exit-status aggregation for `synapse_diagnostic`.
//!
//! `probe_one` records a failed load on the JSON row rather than propagating
//! it, so the example would otherwise write the report and return `Ok(())`
//! even when every checkpoint failed — or when nothing resolved at all.
//! This module is the single place that folds those rows into a process-level
//! outcome. It is self-contained (`std` only) so integration tests can
//! `#[path]`-include it under `--no-default-features`.

/// Parse a `SYNAPSE_DIAG_STRICT` value. Default `false` so exploratory probing
/// stays non-fatal when the var is unset. Accepts the same truthy tokens as
/// the other example flags (`1` / `true` / `yes` / `on`).
pub fn parse_synapse_diag_strict(value: Option<&str>) -> bool {
    value
        .map(|raw| {
            matches!(
                raw.to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or(false)
}

/// Read `SYNAPSE_DIAG_STRICT` from the environment.
pub fn synapse_diag_strict_from_env() -> bool {
    parse_synapse_diag_strict(std::env::var("SYNAPSE_DIAG_STRICT").ok().as_deref())
}

/// Aggregated outcome of a synapse-diagnostic run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SynapseDiagOutcome {
    Success,
    NoModelsResolved,
    ProbeFailures { failed: usize, total: usize },
}

impl SynapseDiagOutcome {
    /// Fold the resolved-model count and the number of rows with `error: Some`
    /// into an outcome. An empty list takes precedence over any failure count.
    pub fn from_counts(resolved_models: usize, failed_probes: usize) -> Self {
        if resolved_models == 0 {
            Self::NoModelsResolved
        } else if failed_probes > 0 {
            Self::ProbeFailures {
                failed: failed_probes,
                total: resolved_models,
            }
        } else {
            Self::Success
        }
    }

    /// Empty resolution is always fatal (a vacuously empty report must not
    /// look like success). Probe failures are fatal only in strict mode.
    pub fn is_fatal(&self, strict: bool) -> bool {
        match self {
            Self::Success => false,
            Self::NoModelsResolved => true,
            Self::ProbeFailures { .. } => strict,
        }
    }

    /// Human-readable warning or error line. `None` on a clean run.
    pub fn message(&self, strict: bool) -> Option<String> {
        match self {
            Self::Success => None,
            Self::NoModelsResolved => Some(
                "synapse_diagnostic: no validation models resolved (LINEUP_CONFIG / \
                 CHECKPOINT_PATH / autodiscovery all returned empty)"
                    .to_owned(),
            ),
            Self::ProbeFailures { failed, total } if strict => Some(format!(
                "synapse_diagnostic: {failed}/{total} probe(s) failed"
            )),
            Self::ProbeFailures { failed, total } => Some(format!(
                "synapse_diagnostic: {failed}/{total} probe(s) failed \
                 (non-fatal; SYNAPSE_DIAG_STRICT=1 to fail the process)"
            )),
        }
    }
}

/// Fold probe counts into a process-level result.
///
/// * `Ok(None)` — every probe succeeded.
/// * `Ok(Some(warning))` — at least one probe failed, but strict mode is off.
/// * `Err(message)` — empty resolution, or a probe failed in strict mode.
pub fn synapse_diag_result(
    resolved_models: usize,
    failed_probes: usize,
    strict: bool,
) -> Result<Option<String>, String> {
    let outcome = SynapseDiagOutcome::from_counts(resolved_models, failed_probes);
    let message = outcome.message(strict);
    if outcome.is_fatal(strict) {
        Err(message.unwrap_or_else(|| "synapse_diagnostic failed".to_owned()))
    } else {
        Ok(message)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_strict_accepts_truthy_tokens() {
        for value in ["1", "true", "TRUE", "yes", "On"] {
            assert!(
                parse_synapse_diag_strict(Some(value)),
                "expected {value:?} to be truthy"
            );
        }
    }

    #[test]
    fn parse_strict_rejects_falsey_and_unset() {
        for value in [None, Some(""), Some("0"), Some("false"), Some("off")] {
            assert!(
                !parse_synapse_diag_strict(value),
                "expected {value:?} to be falsey"
            );
        }
        // Thin env wrapper: call it so the #[path] harness does not warn
        // unused. Do not assert the value — parallel tests must not mutate
        // process-wide environment.
        let _ = synapse_diag_strict_from_env();
    }

    #[test]
    fn empty_resolution_is_always_fatal() {
        assert_eq!(
            SynapseDiagOutcome::from_counts(0, 0),
            SynapseDiagOutcome::NoModelsResolved
        );
        assert!(SynapseDiagOutcome::NoModelsResolved.is_fatal(false));
        assert!(SynapseDiagOutcome::NoModelsResolved.is_fatal(true));
        let err = synapse_diag_result(0, 0, false).expect_err("empty list must fail");
        assert!(err.contains("no validation models resolved"));
        assert!(synapse_diag_result(0, 0, true).is_err());
    }

    #[test]
    fn clean_run_is_success_regardless_of_strict() {
        assert_eq!(
            SynapseDiagOutcome::from_counts(3, 0),
            SynapseDiagOutcome::Success
        );
        assert!(!SynapseDiagOutcome::Success.is_fatal(false));
        assert!(!SynapseDiagOutcome::Success.is_fatal(true));
        assert_eq!(synapse_diag_result(3, 0, false), Ok(None));
        assert_eq!(synapse_diag_result(3, 0, true), Ok(None));
    }

    #[test]
    fn probe_failures_are_fatal_only_when_strict() {
        let outcome = SynapseDiagOutcome::from_counts(3, 1);
        assert_eq!(
            outcome,
            SynapseDiagOutcome::ProbeFailures {
                failed: 1,
                total: 3
            }
        );
        assert!(!outcome.is_fatal(false));
        assert!(outcome.is_fatal(true));

        let warning = synapse_diag_result(3, 1, false).expect("non-strict stays Ok");
        let warning = warning.expect("non-strict failures emit a warning");
        assert!(warning.contains("1/3"));
        assert!(warning.contains("SYNAPSE_DIAG_STRICT=1"));

        let err = synapse_diag_result(3, 1, true).expect_err("strict must fail");
        assert!(err.contains("1/3"));
        assert!(!err.contains("non-fatal"));
    }

    #[test]
    fn every_probe_failing_still_folds_as_probe_failures() {
        let outcome = SynapseDiagOutcome::from_counts(2, 2);
        assert_eq!(
            outcome,
            SynapseDiagOutcome::ProbeFailures {
                failed: 2,
                total: 2
            }
        );
        assert!(synapse_diag_result(2, 2, true).is_err());
        assert!(synapse_diag_result(2, 2, false).is_ok());
    }
}
