// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Integration harness for `examples/support/telemetry_csv.rs`.
//!
//! Plain `cargo test` builds example targets without running their tests
//! (`--all-targets` does run them), so this `#[path]` include is what makes
//! the CSV/snapshot tests execute under
//! `cargo test --no-default-features` (self-hosted / local all-targets;
//! hosted PR CI is `--lib` only and does not run this file).

#[path = "../examples/support/telemetry_csv.rs"]
mod telemetry_csv;
