// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Experiment orchestration and output schema.
//!
//! This module holds the canonical data structures consumed by downstream
//! tools (e.g. `Surrogate_Viz.jl`) and any shared experiment logic that is
//! not tied to a specific example binary.

pub mod schema;

pub use schema::{
    ExperimentBundle, ExperimentManifest, ExperimentMetrics, ExperimentSummary, ExperimentWarning,
    ModelAdapterConfig, RunEntry, RunMatrix, WraparoundStats,
};
