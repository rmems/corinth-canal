// SPDX-License-Identifier: Apache-2.0 OR MIT
#[path = "../examples/support/lineup.rs"]
mod lineup;

use corinth_canal::{ModelArchitectureClass, ModelFamily, ModelTarget};
use lineup::{
    cloud_execution_guard, cloud_lineup_path_from_env, load_cloud_lineup, load_safetensors_lineup,
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

        // `load_cloud_lineup` tolerates an unresolvable family (it only warns
        // on stderr) because a blank family is a documented way to say "no
        // corinth-canal family matches yet". A *non-blank* family that does
        // not resolve is a typo, and this is the only place it would surface.
        assert!(
            entry.family.is_some(),
            "slug '{}' declares a family that no ModelFamily alias resolves; \
             leave it blank if no family matches",
            entry.slug
        );

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
    }
}
