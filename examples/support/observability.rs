#![allow(dead_code)]

use std::process::Command;
use std::sync::Once;
use std::time::{SystemTime, UNIX_EPOCH};

use tracing_subscriber::EnvFilter;

static INIT: Once = Once::new();

pub fn init_tracing() {
    INIT.call_once(|| {
        let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
        let builder = tracing_subscriber::fmt().with_env_filter(filter);

        if std::env::var("AGENTOS_JSON_TRACING").as_deref() == Ok("1") {
            builder.json().init();
        } else {
            builder.init();
        }
    });
}

pub fn run_id() -> String {
    std::env::var("AGENTOS_RUN_ID")
        .ok()
        .filter(|value| !value.trim().is_empty())
        .unwrap_or_else(|| {
            let millis = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|duration| duration.as_millis())
                .unwrap_or(0);
            format!("corinth-canal-{millis}")
        })
}

/// Resolve the git SHA stamped into observability events.
///
/// Precedence:
///   1. `AGENTOS_GIT_SHA` (set by managed CI / AgentOS).
///   2. `git rev-parse --short HEAD` invoked from the current working
///      directory. This matches the fallback already used by
///      `scripts/observability/newrelic_event.sh` and
///      `scripts/observability/sentry_release.sh`, so Rust traces and
///      shell-emitted releases / events agree on the SHA for the same
///      checkout (PR #30 review consistency requirement).
///   3. Literal `"unknown"` when neither source resolves (e.g. running
///      from outside a git checkout with the env var unset).
pub fn git_sha() -> String {
    if let Some(value) = std::env::var("AGENTOS_GIT_SHA")
        .ok()
        .filter(|value| !value.trim().is_empty())
    {
        return value;
    }
    if let Ok(output) = Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
    {
        if output.status.success() {
            let sha = String::from_utf8_lossy(&output.stdout).trim().to_owned();
            if !sha.is_empty() {
                return sha;
            }
        }
    }
    "unknown".to_owned()
}

pub fn error_category(status: Option<&str>, error: Option<&str>) -> &'static str {
    match status.unwrap_or_default() {
        "completed" => "none",
        "prompt_embedding_failed" => "config_error",
        // Model::new() and router-load failures both stem from checkpoint
        // / runtime configuration problems (missing GGUF metadata, bad
        // family override, unreachable path). Map them deterministically
        // here so observability dashboards never see them fall through to
        // the substring heuristic below.
        "model_setup_failed" => "config_error",
        "router_load_failed" => "config_error",
        "gpu_setup_failed" => "gpu_error",
        "tick_failed" => "experiment_error",
        _ => {
            let message = error.unwrap_or_default().to_ascii_lowercase();
            if message.contains("strict_repeat_check") {
                "experiment_error"
            } else if message.contains("gpu") || message.contains("cuda") {
                "gpu_error"
            } else if message.contains("checkpoint")
                || message.contains("config")
                || message.contains("no gguf")
            {
                "config_error"
            } else if error.is_some() {
                "unknown_error"
            } else {
                "none"
            }
        }
    }
}
