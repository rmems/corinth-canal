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
}
