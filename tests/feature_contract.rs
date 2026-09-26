// SPDX-License-Identifier: Apache-2.0 OR MIT

use std::fs;
use std::path::Path;

#[test]
fn cargo_features_match_the_documented_build_contract() {
    let manifest = fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("Cargo.toml"))
        .expect("read Cargo.toml");
    let manifest: toml::Value = toml::from_str(&manifest).expect("parse Cargo.toml");
    let features = manifest["features"]
        .as_table()
        .expect("Cargo.toml [features] table");

    assert_eq!(
        features.keys().map(String::as_str).collect::<Vec<_>>(),
        ["cuda", "default", "gpu-stub"]
    );
    assert_eq!(features["default"][0].as_str(), Some("cuda"));
    assert_eq!(features["cuda"][0].as_str(), Some("dep:cust"));
    assert_eq!(features["gpu-stub"][0].as_str(), Some("cuda"));
}

#[test]
fn gpu_stub_build_script_forces_stub_before_nvcc_compilation() {
    let build_rs = fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("build.rs"))
        .expect("read build.rs");

    let stub_branch = build_rs
        .find("if stub_enabled")
        .expect("build.rs must branch on gpu-stub before nvcc compilation");
    let nvcc_kernel_loop = build_rs
        .find("for &(cu_name, fatbin_name) in kernels")
        .expect("build.rs must compile kernels with nvcc on the non-stub path");

    assert!(
        stub_branch < nvcc_kernel_loop,
        "gpu-stub must take the stub path before any nvcc fatbin compilation"
    );
    assert!(
        build_rs.contains("cargo:rustc-cfg=gpu_stub"),
        "gpu-stub builds must emit cfg(gpu_stub) for compile-time contract checks"
    );
    assert!(
        build_rs.contains("fn build_cuda_stub"),
        "stub fatbin/shim wiring should live in build_cuda_stub"
    );
    let build_contract = compact_ws(&build_rs);
    assert!(
        build_contract.contains("Could not find a cuda installation"),
        "build.rs must record that cust/find_cuda_helper still requires a CUDA library layout"
    );
    assert!(
        build_contract.contains("not a no-CUDA build"),
        "gpu-stub must not be described as a build that runs without CUDA libraries"
    );
}

fn compact_ws(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

#[test]
fn gpu_stub_documents_cust_cuda_library_prerequisite() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let cargo_toml = fs::read_to_string(root.join("Cargo.toml")).expect("read Cargo.toml");
    let claude = fs::read_to_string(root.join("CLAUDE.md")).expect("read CLAUDE.md");
    let module_status =
        fs::read_to_string(root.join("docs/MODULE_STATUS.md")).expect("read MODULE_STATUS.md");

    for (name, text) in [
        ("Cargo.toml", cargo_toml.as_str()),
        ("CLAUDE.md", claude.as_str()),
        ("docs/MODULE_STATUS.md", module_status.as_str()),
    ] {
        assert!(
            text.contains("find_cuda_helper::include_cuda()"),
            "{name} must name cust's CUDA library probe"
        );
        let compact = compact_ws(text);
        assert!(
            compact.contains("Could not find a cuda installation"),
            "{name} must record the panic from a host with no CUDA library layout"
        );
        assert!(
            compact.contains("not a no-CUDA build"),
            "{name} must state that gpu-stub is not a no-CUDA build"
        );
    }
}

#[test]
fn gpu_stub_disables_accelerator_readiness_in_source() {
    let accelerator = fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("src/gpu/wrappers/accelerator.rs"),
    )
    .expect("read accelerator.rs");
    let ready = accelerator
        .find("pub fn is_ready")
        .expect("GpuAccelerator::is_ready");
    let body = &accelerator[ready..];
    let next_fn = body
        .find("pub fn kernels")
        .expect("is_ready body ends before kernels");
    let ready_body = &body[..next_fn];
    assert!(
        ready_body.contains("cfg!(gpu_stub)"),
        "is_ready must consult cfg(gpu_stub)"
    );
    assert!(
        ready_body.contains("return false"),
        "is_ready must return false under the stub cfg"
    );
}
