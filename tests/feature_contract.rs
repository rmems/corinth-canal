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
    assert_eq!(
        features["default"].as_array(),
        features["gpu-stub"].as_array()
    );
    assert_eq!(features["cuda"][0].as_str(), Some("dep:cust"));
    assert_eq!(features["gpu-stub"][0].as_str(), Some("cuda"));
}
