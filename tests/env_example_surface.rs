// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Operator-surface regression for GH#174 / RM-951.
//!
//! `.env.example` and the documented `just` invocations are the first five
//! minutes of the repo. Hosted PR CI is `--lib` only, so this file runs under
//! self-hosted / local `--all-targets` (and plain `cargo test --no-default-features`).

fn assignment_lines(text: &str) -> Vec<&str> {
    text.lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with('#') && line.contains('='))
        .collect()
}

fn has_assignment(text: &str, key: &str) -> bool {
    let prefix = format!("{key}=");
    assignment_lines(text)
        .iter()
        .any(|line| line.starts_with(&prefix))
}

#[test]
fn env_example_omits_unread_keys() {
    let text = include_str!("../.env.example");
    for key in ["MODEL_ROOT", "LLAMA_EMBEDDING_BIN", "EMBEDDING_BACKEND"] {
        assert!(
            !has_assignment(text, key),
            "{key} is documented in .env.example but nothing in src/ or examples/ reads it"
        );
    }
}

#[test]
fn env_example_documents_keys_the_code_reads() {
    let text = include_str!("../.env.example");
    for key in [
        "EMBEDDING_PROVIDER",
        "OLLAMA_EMBED_MODEL",
        "OLLAMA_EMBED_PREFIX",
        "OLLAMA_EMBED_URL",
        "CLOUD_LINEUP_CONFIG",
        "GPU_SMOKE_TICKS",
        "GROK1_ARTIFACT_READY",
        "OBSERVABILITY_PROBE_MODE",
        "INPUT_DRIVE_GAIN",
    ] {
        assert!(
            has_assignment(text, key),
            "{key} is read by the examples / src but missing from .env.example"
        );
    }
}

#[test]
fn justfile_documents_working_replay_and_saaq_csv() {
    let text = include_str!("../justfile");
    assert!(
        text.contains("just replay /path/to/telemetry.csv"),
        "replay recipe comment must show the positional form"
    );
    assert!(
        !text.contains("just replay PATH="),
        "just replay PATH=... passes the literal assignment as the CSV path"
    );
    assert!(
        text.contains("TELEMETRY_CSV_PATH=/path/to/telemetry.csv just saaq-csv"),
        "saaq-csv comment must show a shell assignment, not a fake recipe argument"
    );
}

#[test]
fn claude_md_documents_working_replay() {
    let text = include_str!("../CLAUDE.md");
    assert!(
        text.contains("just replay /path/telemetry.csv"),
        "CLAUDE.md must show the positional replay form"
    );
    assert!(
        !text.contains("just replay PATH="),
        "CLAUDE.md must not repeat the assignment form that just treats as the path"
    );
}
