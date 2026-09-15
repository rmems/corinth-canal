// SPDX-License-Identifier: Apache-2.0 OR MIT
#[path = "../examples/support/lineup.rs"]
mod lineup;

use corinth_canal::{ModelArchitectureClass, ModelFamily, ModelTarget};
use lineup::{
    UnresolvedLineupEntry, cloud_execution_guard, cloud_lineup_path_from_env,
    format_unresolved_lineup_error, load_cloud_lineup, load_gguf_lineup, load_safetensors_lineup,
    safetensors_lineup_path_from_env,
};
use std::path::PathBuf;

#[test]
fn cloud_lineup_parses_valid_toml() {
    let tmp = std::env::temp_dir().join(format!(
        "cloud_test_{}.toml",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::write(
        &tmp,
        r#"
[[model]]
slug = "test_cloud_model"
family = "olmoe"
cloud_model_id = "test/example-model"
source_url = "https://example.com/model"
target = "cloud"
architecture = "MoE"
active_params = "1B"
total_params = "7B"
provider_format = "nvcf-nim"
required_env_vars = ["TEST_ENDPOINT", "TEST_API_KEY"]
"#,
    )
    .unwrap();

    let entries = load_cloud_lineup(&tmp).unwrap();
    let _ = std::fs::remove_file(&tmp);

    assert_eq!(entries.len(), 1);
    let e = &entries[0];
    assert_eq!(e.slug, "test_cloud_model");
    assert_eq!(e.family, Some(ModelFamily::Olmoe));
    assert_eq!(e.cloud_model_id, "test/example-model");
    assert_eq!(e.target, ModelTarget::Cloud);
    assert_eq!(e.architecture, ModelArchitectureClass::Moe);
    assert_eq!(e.active_params, "1B");
    assert_eq!(e.total_params, "7B");
    assert_eq!(e.provider_format, "nvcf-nim");
    assert_eq!(e.required_env_vars, vec!["TEST_ENDPOINT", "TEST_API_KEY"]);
    assert!(!e.cloud_provider_available());
}

#[test]
fn cloud_execution_guard_fails_when_provider_unavailable() {
    let entry = corinth_canal::CloudModelSpec {
        slug: "test_model".into(),
        family: Some(ModelFamily::Olmoe),
        cloud_model_id: "test/model".into(),
        source_url: "https://example.com".into(),
        target: ModelTarget::Cloud,
        architecture: ModelArchitectureClass::Moe,
        active_params: "1B".into(),
        total_params: "7B".into(),
        provider_format: "nvcf-nim".into(),
        required_env_vars: vec!["UNSET_VAR_XYZ".into()],
    };
    let err = cloud_execution_guard(&entry).unwrap_err();
    assert!(err.contains("UNSET_VAR_XYZ"));
    assert!(err.contains("Dioscuri-Cloud"));
}

#[test]
fn cloud_execution_guard_passes_when_provider_available() {
    let entry = corinth_canal::CloudModelSpec {
        slug: "test_model".into(),
        family: Some(ModelFamily::Olmoe),
        cloud_model_id: "test/model".into(),
        source_url: "https://example.com".into(),
        target: ModelTarget::Cloud,
        architecture: ModelArchitectureClass::Moe,
        active_params: "1B".into(),
        total_params: "7B".into(),
        provider_format: "nvcf-nim".into(),
        required_env_vars: vec!["PATH".into()],
    };
    assert!(cloud_execution_guard(&entry).is_ok());
}

#[test]
fn cloud_lineup_unknown_family_reported_on_stderr() {
    let tmp = std::env::temp_dir().join(format!(
        "cloud_unknown_{}.toml",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::write(
        &tmp,
        r#"
[[model]]
slug = "unknown_family_model"
family = "totally_fake_family"
cloud_model_id = "fake/model"
source_url = "https://example.com"
target = "cloud"
architecture = "dense"
active_params = "100M"
total_params = "100M"
provider_format = "rest"
required_env_vars = ["TEST_ENDPOINT"]
"#,
    )
    .unwrap();

    let entries = load_cloud_lineup(&tmp).unwrap();
    let _ = std::fs::remove_file(&tmp);
    assert_eq!(entries.len(), 1);
    assert!(entries[0].family.is_none());
}

#[test]
fn lineup_env_path_helpers_are_reachable() {
    let _cloud: fn() -> Option<PathBuf> = cloud_lineup_path_from_env;
    let _safetensors: fn() -> Option<PathBuf> = safetensors_lineup_path_from_env;
}

#[test]
fn safetensors_lineup_parses_valid_toml() {
    let tmp = std::env::temp_dir().join(format!(
        "safetensors_test_{}.toml",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let existing_file = std::env::temp_dir().join(format!(
        "st_dummy_{}.safetensors",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::write(&existing_file, b"test").unwrap();

    std::fs::write(
        &tmp,
        format!(
            r#"
[[model]]
slug = "test_st_model"
family = "olmoe"
path = "{}"
target = "local"
"#,
            existing_file.display()
        ),
    )
    .unwrap();

    let entries = load_safetensors_lineup(&tmp).unwrap();

    assert_eq!(entries.len(), 1);
    let e = &entries[0];
    assert_eq!(e.slug, "test_st_model");
    assert_eq!(e.family, Some(ModelFamily::Olmoe));
    assert_eq!(e.path, existing_file);
    assert_eq!(e.target, "local");

    let _ = std::fs::remove_file(&tmp);
    let _ = std::fs::remove_file(&existing_file);
}

#[test]
fn safetensors_lineup_skips_missing_paths() {
    let tmp = std::env::temp_dir().join(format!(
        "st_missing_{}.toml",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::write(
        &tmp,
        r#"
[[model]]
slug = "missing_model"
family = "olmoe"
path = "/nonexistent/path/model.safetensors"
target = "local"
"#,
    )
    .unwrap();

    let entries = load_safetensors_lineup(&tmp).unwrap();
    let _ = std::fs::remove_file(&tmp);
    assert!(entries.is_empty());
}

/// Every `provider_format` the checklist and `docs/RUN_PROFILES.md` recognize.
///
/// Two kinds of value live here and the docs list both: an API protocol the
/// provider speaks (`nvcf-nim`, `openai-compat`, `vertex-ai`, `watsonx-saas`)
/// and a weights format downloaded and run on our own GPU (`safetensors`,
/// `fp8-safetensors`). Nothing in `src/` validates this field, so this list
/// and the docs are the only thing keeping the vocabulary from drifting.
const RECOGNIZED_PROVIDER_FORMATS: &[&str] = &[
    "nvcf-nim",
    "openai-compat",
    "vertex-ai",
    "watsonx-saas",
    "safetensors",
    "fp8-safetensors",
];

/// Raw view of the shipped lineup, used only to recover a distinction
/// `CloudModelSpec` erases.
///
/// It stores `family` as `Option<ModelFamily>`, so a deliberately blank
/// `family` and a misspelled one both arrive as `None`. Re-reading the file
/// keeps the two apart: blank is documented and fine, misspelled is a typo.
#[derive(serde::Deserialize)]
struct RawInventory {
    #[serde(default)]
    model: Vec<RawInventoryEntry>,
}

#[derive(serde::Deserialize)]
struct RawInventoryEntry {
    slug: String,
    #[serde(default)]
    family: String,
}

/// Parse the checked-in cloud inventory itself, not a synthetic fixture.
///
/// Every other test in this file writes its own TOML, so a malformed or
/// mis-typed entry landing in `configs/saaq_cloud_lineup.toml` would ship
/// unnoticed: `load_cloud_lineup` hard-errors on unknown fields, a `target`
/// other than `cloud`, and an `architecture` other than `moe`/`dense`, but
/// nothing was calling it against the shipped file.
///
/// `CARGO_MANIFEST_DIR` is expanded at compile time, so this is not the
/// env-based path discovery that CLAUDE.md forbids in `src/` — and this is a
/// test target, not library code.
#[test]
fn cloud_lineup_shipped_inventory_parses() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("configs")
        .join("saaq_cloud_lineup.toml");
    assert!(
        path.is_file(),
        "{} is missing; the cloud onboarding checklist references it",
        path.display()
    );

    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("{} could not be read: {e}", path.display()));
    let raw: RawInventory = toml::from_str(&text)
        .unwrap_or_else(|e| panic!("{} failed to parse as TOML: {e}", path.display()));
    let declared_family: std::collections::HashMap<&str, &str> = raw
        .model
        .iter()
        .map(|m| (m.slug.as_str(), m.family.trim()))
        .collect();

    let entries = load_cloud_lineup(&path)
        .unwrap_or_else(|e| panic!("{} failed to parse: {e}", path.display()));
    assert!(
        !entries.is_empty(),
        "{} declares no [[model]] entries",
        path.display()
    );

    let mut seen: Vec<&str> = Vec::with_capacity(entries.len());
    for entry in &entries {
        assert!(
            !entry.slug.trim().is_empty(),
            "cloud lineup entry has an empty slug"
        );
        // Slugs become artifact directory names; a duplicate silently
        // overwrites a sibling model's run output.
        assert!(
            !seen.contains(&entry.slug.as_str()),
            "duplicate slug '{}' in {}",
            entry.slug,
            path.display()
        );
        seen.push(&entry.slug);

        // A blank `family` is a documented way to say "no corinth-canal family
        // matches yet" (see the field list at the top of the lineup file), and
        // `load_cloud_lineup` only warns on stderr for one it cannot resolve.
        // So `None` is a failure only when the file actually spelled
        // something — which is why the raw string is read back above.
        let declared = declared_family
            .get(entry.slug.as_str())
            .copied()
            .unwrap_or_default();
        if !declared.is_empty() {
            assert!(
                entry.family.is_some(),
                "slug '{}' declares family '{declared}', which no ModelFamily \
                 alias resolves; fix the spelling or leave it blank",
                entry.slug
            );
        }

        assert_eq!(
            entry.target,
            ModelTarget::Cloud,
            "slug '{}' is not targeted at cloud",
            entry.slug
        );
        assert!(
            !entry.cloud_model_id.trim().is_empty(),
            "slug '{}' has an empty cloud_model_id",
            entry.slug
        );
        assert!(
            entry.source_url.starts_with("https://"),
            "slug '{}' source_url is not an https URL: '{}'",
            entry.slug,
            entry.source_url
        );
        // Nothing in `src/` checks this field, so an unrecognized value would
        // otherwise reach an operator ticking the checklist's "provider format
        // is a recognized value" box with nothing to check it against.
        assert!(
            RECOGNIZED_PROVIDER_FORMATS.contains(&entry.provider_format.as_str()),
            "slug '{}' has provider_format '{}', which is not one of {:?}; \
             add it to the documented set in \
             docs/MODEL_SOURCE_VERIFICATION_CHECKLIST.md and \
             docs/RUN_PROFILES.md if it is legitimate",
            entry.slug,
            entry.provider_format,
            RECOGNIZED_PROVIDER_FORMATS
        );
    }
}

fn unique_temp(prefix: &str, suffix: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "{prefix}_{}_{suffix}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ))
}

fn write_gguf_lineup(toml: &str) -> (PathBuf, PathBuf) {
    let checkpoint = unique_temp("gguf_lineup", "present.gguf");
    std::fs::write(&checkpoint, b"not-a-real-gguf").unwrap();
    let lineup = unique_temp("gguf_lineup", "lineup.toml");
    let rendered = toml.replace("{PRESENT}", &checkpoint.display().to_string());
    std::fs::write(&lineup, rendered).unwrap();
    (lineup, checkpoint)
}

#[test]
fn gguf_lineup_skips_missing_checkpoints_without_strict() {
    let (lineup, checkpoint) = write_gguf_lineup(
        r#"
[[model]]
slug = "present_model"
family = "olmoe"
path = "{PRESENT}"

[[model]]
slug = "missing_model"
family = "olmoe"
path = "/this/checkpoint/does/not/exist.gguf"

[[model]]
slug = "empty_path_model"
family = "olmoe"
path = ""
"#,
    );

    let loaded = load_gguf_lineup(&lineup, false).expect("non-strict load");
    let _ = std::fs::remove_file(&lineup);
    let _ = std::fs::remove_file(&checkpoint);

    assert_eq!(loaded.declared_count, 3);
    assert_eq!(loaded.models.len(), 1);
    assert_eq!(loaded.models[0].slug, "present_model");
    assert_eq!(loaded.models[0].family, Some(ModelFamily::Olmoe));
}

#[test]
fn gguf_lineup_strict_names_unresolved_slugs() {
    let (lineup, checkpoint) = write_gguf_lineup(
        r#"
[[model]]
slug = "present_model"
family = "olmoe"
path = "{PRESENT}"

[[model]]
slug = "missing_alpha"
family = "qwen3_moe"
path = "/missing/alpha.gguf"

[[model]]
slug = "missing_beta"
family = "gemma4"
path = "/missing/beta.gguf"
"#,
    );

    let err = load_gguf_lineup(&lineup, true).expect_err("strict load must fail");
    let _ = std::fs::remove_file(&lineup);
    let _ = std::fs::remove_file(&checkpoint);

    let msg = err.to_string();
    assert!(
        msg.starts_with("LINEUP_STRICT=1:"),
        "strict error should lead with LINEUP_STRICT=1, got {msg}"
    );
    assert!(msg.contains("declared 3 models"), "got {msg}");
    assert!(msg.contains("2 unresolved"), "got {msg}");
    assert!(msg.contains("slug=missing_alpha"), "got {msg}");
    assert!(msg.contains("slug=missing_beta"), "got {msg}");
    assert!(msg.contains("/missing/alpha.gguf"), "got {msg}");
    assert!(msg.contains("/missing/beta.gguf"), "got {msg}");
    assert!(
        !msg.contains("slug=present_model"),
        "resolved slug should not be listed as unresolved: {msg}"
    );
}

#[test]
fn gguf_lineup_strict_succeeds_when_every_entry_resolves() {
    let (lineup, checkpoint) = write_gguf_lineup(
        r#"
[[model]]
slug = "only_model"
family = "olmoe"
path = "{PRESENT}"
"#,
    );

    let loaded = load_gguf_lineup(&lineup, true).expect("strict load of a complete lineup");
    let _ = std::fs::remove_file(&lineup);
    let _ = std::fs::remove_file(&checkpoint);

    assert_eq!(loaded.declared_count, 1);
    assert_eq!(loaded.models.len(), 1);
    assert_eq!(loaded.models[0].slug, "only_model");
}

#[test]
fn format_unresolved_lineup_error_names_empty_path() {
    let msg = format_unresolved_lineup_error(
        2,
        &[
            UnresolvedLineupEntry {
                slug: "empty".into(),
                path: String::new(),
                reason: "empty path",
            },
            UnresolvedLineupEntry {
                slug: "gone".into(),
                path: "/nope.gguf".into(),
                reason: "file not found",
            },
        ],
    );
    assert!(msg.contains("slug=empty path=(empty): empty path"));
    assert!(msg.contains("slug=gone path=/nope.gguf: file not found"));
}
