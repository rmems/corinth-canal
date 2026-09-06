// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Integration harness for `examples/support/observability.rs`.
//!
//! Example targets do not run unit-test harnesses, and all five examples that
//! use this module are `required-features = ["cuda"]`, so nothing executed its
//! tests under the CPU CI job. This `#[path]` include is what makes the
//! Sentry path-redaction tests — and the existing "stays disabled when
//! SENTRY_DSN is blank" guards — run under
//! `cargo test --no-default-features`.

#[path = "../examples/support/observability.rs"]
mod observability;
