// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Integration harness for `examples/support/synapse_diag.rs`.
//!
//! Example targets do not run unit-test harnesses, so this `#[path]` include
//! is what makes the exit-status aggregation tests execute under
//! `cargo test --no-default-features` (self-hosted / local all-targets;
//! hosted PR CI is `--lib` only and does not run this file).

#[path = "../examples/support/synapse_diag.rs"]
mod synapse_diag;
