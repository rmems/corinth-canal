// SPDX-License-Identifier: Apache-2.0 OR MIT
#![allow(dead_code)]

use std::borrow::Cow;
use std::cell::RefCell;
#[cfg(test)]
use std::ffi::OsString;
use std::path::Path;
use std::process::Command;
use std::sync::{Once, OnceLock};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use opentelemetry::trace::{Span as _, Tracer as _, TracerProvider as _};
use opentelemetry_otlp::{WithExportConfig, WithHttpConfig};
use opentelemetry_sdk::trace::{SdkTracerProvider, Span as SdkSpan};
use sentry::ClientInitGuard;
use serde_json::json;
use tracing_subscriber::EnvFilter;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;

static INIT: Once = Once::new();
const REPO_NAME: &str = "corinth-canal";

/// OpenTelemetry tracer provider guard. Kept alive for the process lifetime
/// so the global tracer is not dropped prematurely.
static OTEL_PROVIDER: OnceLock<SdkTracerProvider> = OnceLock::new();

#[derive(Debug, Clone, Copy, Default)]
pub struct SafeDiagnosticData<'a> {
    pub model_slug: Option<&'a str>,
    pub telemetry_source: Option<&'a str>,
    pub validation_status: Option<&'a str>,
    pub error_category: Option<&'a str>,
    pub prompt_profile: Option<&'a str>,
    pub saaq_rule: Option<&'a str>,
}

impl<'a> SafeDiagnosticData<'a> {
    pub fn with_model_slug(mut self, model_slug: &'a str) -> Self {
        self.model_slug = Some(model_slug);
        self
    }

    pub fn with_telemetry_source(mut self, telemetry_source: &'a str) -> Self {
        self.telemetry_source = Some(telemetry_source);
        self
    }

    pub fn with_validation_status(mut self, validation_status: &'a str) -> Self {
        self.validation_status = Some(validation_status);
        self
    }

    pub fn with_error_category(mut self, error_category: &'a str) -> Self {
        self.error_category = Some(error_category);
        self
    }

    pub fn with_prompt_profile(mut self, prompt_profile: &'a str) -> Self {
        self.prompt_profile = Some(prompt_profile);
        self
    }

    pub fn with_saaq_rule(mut self, saaq_rule: &'a str) -> Self {
        self.saaq_rule = Some(saaq_rule);
        self
    }
}

pub struct CommandObserver {
    command: &'static str,
    run_id: String,
    git_sha: String,
    started: Instant,
    safe_data: RefCell<OwnedDiagnosticData>,
    span: RefCell<Option<SdkSpan>>,
}

pub trait ErrorReport {
    fn as_dyn_error(&self) -> &(dyn std::error::Error + 'static);
}

#[cfg(test)]
const SUBPROCESS_PROBE_ARG: &str = "__observability_probe";

#[derive(Debug, Clone, Default)]
struct OwnedDiagnosticData {
    model_slug: Option<String>,
    telemetry_source: Option<String>,
    validation_status: Option<String>,
    error_category: Option<String>,
    prompt_profile: Option<String>,
    saaq_rule: Option<String>,
}

impl OwnedDiagnosticData {
    fn merge(&mut self, data: SafeDiagnosticData<'_>) {
        if let Some(model_slug) = data.model_slug {
            self.model_slug = Some(model_slug.to_owned());
        }
        if let Some(telemetry_source) = data.telemetry_source {
            self.telemetry_source = Some(telemetry_source.to_owned());
        }
        if let Some(validation_status) = data.validation_status {
            self.validation_status = Some(validation_status.to_owned());
        }
        if let Some(error_category) = data.error_category {
            self.error_category = Some(error_category.to_owned());
        }
        if let Some(prompt_profile) = data.prompt_profile {
            self.prompt_profile = Some(prompt_profile.to_owned());
        }
        if let Some(saaq_rule) = data.saaq_rule {
            self.saaq_rule = Some(saaq_rule.to_owned());
        }
    }

    fn with_status(mut self, validation_status: &str, error_category: &str) -> Self {
        self.validation_status = Some(validation_status.to_owned());
        self.error_category = Some(error_category.to_owned());
        self
    }

    fn as_safe(&self) -> SafeDiagnosticData<'_> {
        SafeDiagnosticData {
            model_slug: self.model_slug.as_deref(),
            telemetry_source: self.telemetry_source.as_deref(),
            validation_status: self.validation_status.as_deref(),
            error_category: self.error_category.as_deref(),
            prompt_profile: self.prompt_profile.as_deref(),
            saaq_rule: self.saaq_rule.as_deref(),
        }
    }
}

pub fn init_tracing() {
    INIT.call_once(|| {
        let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
        let json_tracing = std::env::var("AGENTOS_JSON_TRACING").as_deref() == Ok("1");
        init_opentelemetry_provider();

        if json_tracing {
            tracing_subscriber::registry()
                .with(filter)
                .with(tracing_subscriber::fmt::layer().json())
                .init();
        } else {
            tracing_subscriber::registry()
                .with(filter)
                .with(tracing_subscriber::fmt::layer())
                .init();
        }
    });
}

/// Attempt to build an OpenTelemetry provider backed by a New Relic OTLP
/// exporter. Missing or invalid credentials leave tracing local-only.
fn init_opentelemetry_provider() {
    let api_key = std::env::var("NR_INSERT_KEY")
        .ok()
        .filter(|v| !v.trim().is_empty());
    let Some(api_key) = api_key else {
        return;
    };
    let service_name = std::env::var("OTEL_SERVICE_NAME")
        .ok()
        .filter(|v| !v.trim().is_empty())
        .unwrap_or_else(|| REPO_NAME.to_owned());

    let has_generic_endpoint = std::env::var("OTEL_EXPORTER_OTLP_ENDPOINT")
        .ok()
        .filter(|v| !v.trim().is_empty())
        .is_some();
    let has_traces_endpoint = std::env::var("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT")
        .ok()
        .filter(|v| !v.trim().is_empty())
        .is_some();

    let mut exporter_builder = opentelemetry_otlp::SpanExporter::builder()
        .with_http()
        .with_headers(std::collections::HashMap::from([(
            "api-key".to_owned(),
            api_key,
        )]));

    if !has_generic_endpoint && !has_traces_endpoint {
        exporter_builder = exporter_builder.with_endpoint("https://otlp.nr-data.net/v1/traces");
    }

    let exporter = match exporter_builder.build() {
        Ok(exporter) => exporter,
        Err(error) => {
            eprintln!("OpenTelemetry disabled: failed to build New Relic OTLP exporter ({error})");
            return;
        }
    };

    let resource = opentelemetry_sdk::Resource::builder()
        .with_attribute(opentelemetry::KeyValue::new("service.name", service_name))
        .with_attribute(opentelemetry::KeyValue::new(
            "service.repository",
            REPO_NAME,
        ))
        .build();

    let provider = SdkTracerProvider::builder()
        .with_simple_exporter(exporter)
        .with_resource(resource)
        .build();

    opentelemetry::global::set_tracer_provider(provider.clone());
    let _ = OTEL_PROVIDER.set(provider);
}

/// Flush any pending OpenTelemetry spans without shutting down the global
/// provider. Safe to call when OTel was never initialized; it becomes a no-op.
pub fn flush_opentelemetry() {
    if let Some(provider) = OTEL_PROVIDER.get() {
        let _ = provider.force_flush();
    }
}

pub fn start_command(command: &'static str) -> CommandObserver {
    init_tracing();
    let run_id = run_id();
    let git_sha = git_sha();
    let span = start_otel_command_span(command, &run_id, &git_sha);
    let observer = CommandObserver {
        command,
        run_id,
        git_sha,
        started: Instant::now(),
        safe_data: RefCell::new(OwnedDiagnosticData::default()),
        span: RefCell::new(span),
    };
    annotate_scope(
        observer.command,
        &observer.run_id,
        &observer.git_sha,
        SafeDiagnosticData::default(),
    );
    tracing::info!("command_start");
    observer
}

fn start_otel_command_span(command: &'static str, run_id: &str, git_sha: &str) -> Option<SdkSpan> {
    let provider = OTEL_PROVIDER.get()?;
    let tracer = provider.tracer(REPO_NAME);
    let mut span = tracer.start("command_execution");
    span.set_attribute(opentelemetry::KeyValue::new("repo", REPO_NAME));
    span.set_attribute(opentelemetry::KeyValue::new("command", command));
    span.set_attribute(opentelemetry::KeyValue::new("run_id", run_id.to_owned()));
    span.set_attribute(opentelemetry::KeyValue::new("git_sha", git_sha.to_owned()));
    Some(span)
}

pub fn init_sentry(command: &'static str) -> Option<ClientInitGuard> {
    let dsn = std::env::var("SENTRY_DSN")
        .ok()
        .map(|value| value.trim().to_owned())
        .filter(|value| !value.is_empty())?;
    let git_sha = git_sha();
    let release = resolve_sentry_release(&git_sha);
    let environment = std::env::var("SENTRY_ENVIRONMENT")
        .ok()
        .map(|value| value.trim().to_owned())
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| "local".to_owned());
    let parsed_dsn = match dsn.parse() {
        Ok(dsn) => dsn,
        Err(error) => {
            eprintln!("Sentry disabled: invalid SENTRY_DSN ({error})");
            return None;
        }
    };

    let guard = sentry::init(sentry::ClientOptions {
        dsn: Some(parsed_dsn),
        release: Some(Cow::Owned(release)),
        environment: Some(Cow::Owned(environment)),
        sample_rate: 1.0,
        traces_sample_rate: 0.0,
        default_integrations: true,
        ..Default::default()
    });

    annotate_scope(command, &run_id(), &git_sha, SafeDiagnosticData::default());

    Some(guard)
}

/// Replace absolute filesystem paths with a non-identifying stand-in.
///
/// The Sentry tag allowlist only covers *tags*. Exception values and messages
/// go to Sentry verbatim, and several of them carry absolute checkpoint paths:
/// `HybridError::ModelLoad` and `MissingTensor` bake `{path}` into their
/// `Display`, and the SAAQ runner formats `ctx.spec.path` into a message.
///
/// A path is replaced by `<path:stem>`, which keeps the event correlatable to
/// a checkpoint without publishing the directory layout of the machine.
pub fn redact_absolute_paths(text: &str) -> String {
    const DELIMITERS: [char; 9] = [' ', '\t', '\n', '"', '\'', '(', ')', '[', ']'];

    let mut out = String::with_capacity(text.len());
    let mut chars = text.char_indices().peekable();
    let mut at_token_start = true;

    while let Some((idx, ch)) = chars.next() {
        if ch == '/' && at_token_start {
            let end = text[idx..]
                .find(|c| DELIMITERS.contains(&c))
                .map(|offset| idx + offset)
                .unwrap_or(text.len());
            let candidate = &text[idx..end];

            // Require at least two separators so a bare "/" or a lone "/tmp"
            // is left alone; a real checkpoint path always has more.
            if candidate.matches('/').count() >= 2 {
                let stem = Path::new(candidate)
                    .file_stem()
                    .and_then(|stem| stem.to_str())
                    .unwrap_or("file");
                out.push_str(&format!("<path:{stem}>"));
                while let Some(&(next_idx, _)) = chars.peek() {
                    if next_idx >= end {
                        break;
                    }
                    chars.next();
                }
                at_token_start = false;
                continue;
            }
        }

        at_token_start = DELIMITERS.contains(&ch) || ch == '=' || ch == ':' || ch == ',';
        out.push(ch);
    }

    out
}

/// Capture an error with absolute paths stripped from every exception value.
fn capture_error_redacted(error: &(dyn std::error::Error + 'static)) {
    let mut event = sentry::event_from_error(error);
    for exception in &mut event.exception.values {
        if let Some(value) = exception.value.as_mut() {
            *value = redact_absolute_paths(value);
        }
    }
    if let Some(message) = event.message.as_mut() {
        *message = redact_absolute_paths(message);
    }
    sentry::capture_event(event);
}

pub fn capture_top_level_error(_command: &'static str, error: &(dyn std::error::Error + 'static)) {
    capture_error_redacted(error);
}

pub fn annotate_scope(
    command: &'static str,
    run_id: &str,
    git_sha: &str,
    data: SafeDiagnosticData<'_>,
) {
    if sentry::Hub::with_active(|hub| hub.client().is_none()) {
        return;
    }

    sentry::configure_scope(|scope| {
        apply_scope(scope, command, run_id, git_sha, data);
    });
}

pub fn capture_scoped_error(
    command: &'static str,
    run_id: &str,
    data: SafeDiagnosticData<'_>,
    error: &(dyn std::error::Error + 'static),
) {
    if sentry::Hub::with_active(|hub| hub.client().is_none()) {
        return;
    }

    let git_sha = git_sha();
    sentry::with_scope(
        |scope| {
            apply_scope(scope, command, run_id, &git_sha, data);
        },
        || {
            capture_error_redacted(error);
        },
    );
}

pub fn checkpoint_slug(path: &str) -> Option<String> {
    let trimmed = path.trim();
    if trimmed.is_empty() {
        return None;
    }

    Some(
        Path::new(trimmed)
            .file_stem()
            .and_then(|stem| stem.to_str())
            .unwrap_or("gguf_model")
            .replace(['.', '-', ' '], "_")
            .to_ascii_lowercase(),
    )
}

pub fn run_id() -> String {
    static RUN_ID: OnceLock<String> = OnceLock::new();

    RUN_ID
        .get_or_init(|| {
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
        })
        .clone()
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
        && output.status.success()
    {
        let sha = String::from_utf8_lossy(&output.stdout).trim().to_owned();
        if !sha.is_empty() {
            return sha;
        }
    }
    "unknown".to_owned()
}

fn resolve_sentry_release(git_sha: &str) -> String {
    if let Some(release) = std::env::var("SENTRY_RELEASE")
        .ok()
        .map(|value| value.trim().to_owned())
        .filter(|value| !value.is_empty())
    {
        return release;
    }

    if !git_sha.trim().is_empty() && git_sha != "unknown" {
        return format!("corinth-canal@{git_sha}");
    }

    if let Some(release) = sentry::release_name!() {
        let release = release.into_owned();
        if !release.trim().is_empty() {
            return release;
        }
    }

    "corinth-canal@unknown".to_owned()
}

impl CommandObserver {
    pub fn run_id(&self) -> &str {
        &self.run_id
    }

    pub fn git_sha(&self) -> &str {
        &self.git_sha
    }

    pub fn annotate(&self, data: SafeDiagnosticData<'_>) {
        let snapshot = {
            let mut safe_data = self.safe_data.borrow_mut();
            safe_data.merge(data);
            safe_data.clone()
        };
        annotate_scope(
            self.command,
            &self.run_id,
            &self.git_sha,
            snapshot.as_safe(),
        );
    }

    pub fn finish<T, E>(&self, result: &Result<T, E>, data: SafeDiagnosticData<'_>)
    where
        E: ErrorReport + ToString,
    {
        let error_message = result.as_ref().err().map(|error| error.to_string());
        let resolved_status = data.validation_status.unwrap_or(if result.is_ok() {
            "completed"
        } else {
            "failed"
        });
        let resolved_category = data.error_category.unwrap_or(error_category(
            Some(resolved_status),
            error_message.as_deref(),
        ));
        let enriched = {
            let mut safe_data = self.safe_data.borrow_mut();
            safe_data.merge(data);
            safe_data
                .clone()
                .with_status(resolved_status, resolved_category)
        };

        annotate_scope(
            self.command,
            &self.run_id,
            &self.git_sha,
            enriched.as_safe(),
        );
        let span = self.span.borrow_mut().take();
        if let Some(mut span) = span {
            span.set_attribute(opentelemetry::KeyValue::new("success", result.is_ok()));
            span.set_attribute(opentelemetry::KeyValue::new(
                "error_category",
                resolved_category.to_owned(),
            ));
            span.set_attribute(opentelemetry::KeyValue::new(
                "validation_status",
                resolved_status.to_owned(),
            ));
            span.end();
        }
        tracing::info!("command_finish");

        if let Err(error) = result.as_ref() {
            capture_scoped_error(
                self.command,
                &self.run_id,
                enriched.as_safe(),
                error.as_dyn_error(),
            );
        }

        flush_opentelemetry();
    }
}

impl ErrorReport for Box<dyn std::error::Error> {
    fn as_dyn_error(&self) -> &(dyn std::error::Error + 'static) {
        self.as_ref()
    }
}

impl ErrorReport for corinth_canal::HybridError {
    fn as_dyn_error(&self) -> &(dyn std::error::Error + 'static) {
        self
    }
}

fn apply_scope(
    scope: &mut sentry::Scope,
    command: &'static str,
    run_id: &str,
    git_sha: &str,
    data: SafeDiagnosticData<'_>,
) {
    scope.set_tag("repo", REPO_NAME);
    scope.set_tag("command", command);
    scope.set_tag("git_sha", git_sha.to_owned());
    if let Some(model_slug) = data.model_slug {
        scope.set_tag("model_slug", model_slug);
    } else {
        scope.remove_tag("model_slug");
    }
    if let Some(telemetry_source) = data.telemetry_source {
        scope.set_tag("telemetry_source", telemetry_source);
    } else {
        scope.remove_tag("telemetry_source");
    }
    if let Some(validation_status) = data.validation_status {
        scope.set_tag("validation_status", validation_status);
    } else {
        scope.remove_tag("validation_status");
    }
    if let Some(error_category) = data.error_category {
        scope.set_tag("error_category", error_category);
    } else {
        scope.remove_tag("error_category");
    }
    if let Some(prompt_profile) = data.prompt_profile {
        scope.set_tag("prompt_profile", prompt_profile);
    } else {
        scope.remove_tag("prompt_profile");
    }
    if let Some(saaq_rule) = data.saaq_rule {
        scope.set_tag("saaq_rule", saaq_rule);
    } else {
        scope.remove_tag("saaq_rule");
    }
    scope.set_extra("run_id", json!(run_id));
}

// ── New Relic telemetry verification helpers ─────────────────────────────────
//
// These functions check whether New Relic environment variables are set and
// provide dry-run verification so SAAQ experiment runs can document telemetry
// health without requiring a live New Relic connection. All functions are
// safe to call when New Relic env vars are unset — they simply report the
// missing state.

/// New Relic environment variables used by the SAAQ observability pipeline.
pub const NR_ENV_VARS: [&str; 4] = [
    "NR_INSERT_KEY",
    "NR_ACCOUNT_ID",
    "NR_QUERY_KEY",
    "NEW_RELIC_APP_NAME",
];

/// Returns a summary of which New Relic env vars are set, for telemetry
/// verification. Never fails — missing env vars are reported as `None`.
pub fn new_relic_env_status() -> Vec<(&'static str, Option<String>)> {
    NR_ENV_VARS
        .iter()
        .map(|&var| {
            let value = std::env::var(var).ok().filter(|v| !v.trim().is_empty());
            (var, value.map(|v| format!("{}chars", v.len())))
        })
        .collect()
}

/// Minimal New Relic env vars required for ingest telemetry.
const NR_REQUIRED: [&str; 2] = ["NR_INSERT_KEY", "NR_ACCOUNT_ID"];

/// Returns `true` only if the minimal New Relic ingest credentials are
/// present (both `NR_INSERT_KEY` and `NR_ACCOUNT_ID` are set and non-empty).
/// This avoids the false-positive "configured" signal when only optional
/// vars (e.g. `NEW_RELIC_APP_NAME`) are present.
pub fn new_relic_is_configured() -> bool {
    NR_REQUIRED.iter().all(|&var| {
        std::env::var(var)
            .ok()
            .filter(|v| !v.trim().is_empty())
            .is_some()
    })
}

/// Returns a human-readable summary of New Relic verification status, suitable
/// for writing into experiment logs and run manifests.
pub fn new_relic_verification_summary() -> String {
    let status = new_relic_env_status();
    let configured = status
        .iter()
        .any(|(var, value)| NR_REQUIRED.contains(var) && value.is_some());
    let mut lines = Vec::new();
    lines.push(format!("new_relic_configured: {configured}"));
    for (var, value) in &status {
        match value {
            Some(masked) => lines.push(format!("  {var}: set ({masked})")),
            None => lines.push(format!("  {var}: unset")),
        }
    }
    if configured {
        lines.push("telemetry_status: new_relic_available".to_owned());
    } else {
        lines.push("telemetry_status: dry_run_no_new_relic".to_owned());
    }
    lines.join("\n")
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
            } else if message.contains("invalid configuration")
                || message.contains("missing telemetry csv path")
            {
                "config_error"
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::process::Command as ProcessCommand;

    #[test]
    fn subprocess_probe() {
        if !std::env::args().any(|arg| arg == SUBPROCESS_PROBE_ARG) {
            return;
        }

        let value = match std::env::var("OBSERVABILITY_PROBE_MODE").ok().as_deref() {
            Some("run_id") => run_id(),
            Some("git_sha") => git_sha(),
            Some("sentry_enabled") => init_sentry("test_command").is_some().to_string(),
            other => panic!("unknown probe mode: {other:?}"),
        };
        println!("OBSERVABILITY_PROBE_RESULT={value}");
    }

    /// libtest path of `subprocess_probe`, derived rather than hardcoded.
    ///
    /// This file is `#[path]`-included from several targets, so its module
    /// path differs per target (`support::observability::tests` inside an
    /// example, `observability::tests` inside tests/). A hardcoded filter
    /// matched nothing outside the example targets, and the probe then
    /// produced no output at all — surfacing as "missing probe result"
    /// rather than as a wrong path. `module_path!()` minus the crate segment
    /// is exactly what libtest's `--exact` expects.
    fn probe_test_path() -> String {
        let module = module_path!();
        let without_crate = module
            .split_once("::")
            .map(|(_, rest)| rest)
            .unwrap_or(module);
        format!("{without_crate}::subprocess_probe")
    }

    fn run_probe(probe_mode: &str, envs: &[(&str, &str)]) -> String {
        let output = ProcessCommand::new(std::env::current_exe().unwrap())
            .arg("--exact")
            .arg(probe_test_path())
            .arg("--nocapture")
            .arg("--")
            .arg(SUBPROCESS_PROBE_ARG)
            .env_remove("AGENTOS_RUN_ID")
            .env_remove("AGENTOS_GIT_SHA")
            .env_remove("SENTRY_DSN")
            .env_remove("SENTRY_RELEASE")
            .env_remove("SENTRY_ENVIRONMENT")
            .env_remove("OBSERVABILITY_PROBE_MODE")
            .env("OBSERVABILITY_PROBE_MODE", probe_mode)
            .envs(
                envs.iter()
                    .map(|(k, v)| (OsString::from(k), OsString::from(v))),
            )
            .output()
            .unwrap();

        assert!(output.status.success(), "probe failed: {output:?}");
        let stdout = String::from_utf8(output.stdout).unwrap();
        stdout
            .lines()
            .find_map(|line| line.strip_prefix("OBSERVABILITY_PROBE_RESULT="))
            .map(str::to_owned)
            .expect("missing probe result")
    }

    #[test]
    fn run_id_uses_agentos_run_id_when_set() {
        assert_eq!(
            run_probe("run_id", &[("AGENTOS_RUN_ID", "run-123")]),
            "run-123"
        );
    }

    #[test]
    fn run_id_falls_back_to_corinth_canal_millis() {
        let envs: Vec<(&str, &str)> = Vec::new();
        let value = run_probe("run_id", envs.as_slice());
        assert!(value.starts_with("corinth-canal-"));
        assert!(value["corinth-canal-".len()..].parse::<u128>().is_ok());
    }

    #[test]
    fn run_id_is_stable_within_a_process() {
        let first = run_id();
        let second = run_id();
        assert_eq!(first, second);
    }

    #[test]
    fn git_sha_uses_agentos_git_sha_when_set() {
        assert_eq!(
            run_probe("git_sha", &[("AGENTOS_GIT_SHA", "deadbee")]),
            "deadbee"
        );
    }

    #[test]
    fn error_category_maps_known_statuses() {
        assert_eq!(error_category(Some("completed"), None), "none");
        assert_eq!(
            error_category(Some("model_setup_failed"), Some("boom")),
            "config_error"
        );
        assert_eq!(
            error_category(Some("router_load_failed"), Some("boom")),
            "config_error"
        );
        assert_eq!(
            error_category(Some("gpu_setup_failed"), Some("boom")),
            "gpu_error"
        );
        assert_eq!(
            error_category(Some("tick_failed"), Some("boom")),
            "experiment_error"
        );
    }

    #[test]
    fn error_category_uses_substring_fallbacks() {
        assert_eq!(
            error_category(None, Some("strict_repeat_check mismatch")),
            "experiment_error"
        );
        assert_eq!(error_category(None, Some("GPU device failed")), "gpu_error");
        assert_eq!(
            error_category(None, Some("cuda launch failed")),
            "gpu_error"
        );
        assert_eq!(
            error_category(None, Some("checkpoint metadata missing")),
            "config_error"
        );
        assert_eq!(
            error_category(None, Some("config parse failed")),
            "config_error"
        );
        assert_eq!(
            error_category(
                None,
                Some(
                    "invalid configuration: missing telemetry CSV path. Usage: cargo run --example csv_replay <telemetry.csv>; CSV format: timestamp_ms,gpu_temp_c,gpu_power_w,cpu_tctl_c,cpu_package_power_w"
                )
            ),
            "config_error"
        );
        assert_eq!(
            error_category(None, Some("cuda config mismatch during device init")),
            "gpu_error"
        );
        assert_eq!(
            error_category(None, Some("mystery failure")),
            "unknown_error"
        );
    }

    #[test]
    fn owned_diagnostic_data_preserves_prior_context_when_new_data_is_sparse() {
        let mut data = OwnedDiagnosticData::default();
        data.merge(
            SafeDiagnosticData::default()
                .with_model_slug("model_a")
                .with_telemetry_source("csv"),
        );

        let merged_data = data.clone().with_status("tick_failed", "experiment_error");
        let merged = merged_data.as_safe();

        assert_eq!(merged.model_slug, Some("model_a"));
        assert_eq!(merged.telemetry_source, Some("csv"));
        assert_eq!(merged.validation_status, Some("tick_failed"));
        assert_eq!(merged.error_category, Some("experiment_error"));
    }

    #[test]
    fn unset_sentry_dsn_disables_sentry_cleanly() {
        let envs: Vec<(&str, &str)> = Vec::new();
        assert_eq!(run_probe("sentry_enabled", envs.as_slice()), "false");
    }

    #[test]
    fn empty_sentry_dsn_disables_sentry_cleanly() {
        assert_eq!(
            run_probe("sentry_enabled", &[("SENTRY_DSN", "   ")]),
            "false"
        );
    }
}

#[cfg(test)]
mod redaction_tests {
    use super::redact_absolute_paths;

    #[test]
    fn strips_absolute_checkpoint_paths_but_keeps_the_stem() {
        let message =
            "model load failed for '/home/alice/.models/gguf/Foo/Bar-Q8_0.gguf': bad magic";
        let redacted = redact_absolute_paths(message);
        assert!(!redacted.contains("/home/alice"), "leaked: {redacted}");
        assert!(!redacted.contains(".models"), "leaked: {redacted}");
        assert!(
            redacted.contains("<path:Bar-Q8_0>"),
            "lost the stem: {redacted}"
        );
        assert!(
            redacted.contains("bad magic"),
            "lost the reason: {redacted}"
        );
    }

    #[test]
    fn redacts_every_path_in_a_multi_path_message() {
        let message = "copy /home/bob/a/model.gguf -> /var/lib/out/result.json failed";
        let redacted = redact_absolute_paths(message);
        assert!(!redacted.contains("/home/bob"), "leaked: {redacted}");
        assert!(!redacted.contains("/var/lib"), "leaked: {redacted}");
        assert!(redacted.contains("<path:model>"));
        assert!(redacted.contains("<path:result>"));
    }

    #[test]
    fn leaves_non_path_text_untouched() {
        for text in [
            "missing tensor 'blk.0.attn_q.weight'",
            "ratio 3/4 exceeded",
            "input length mismatch: expected 2048, got 12",
        ] {
            assert_eq!(redact_absolute_paths(text), text, "mangled: {text}");
        }
    }

    #[test]
    fn survives_a_message_with_no_paths_and_no_delimiters() {
        assert_eq!(redact_absolute_paths(""), "");
        assert_eq!(redact_absolute_paths("/"), "/");
    }
}
