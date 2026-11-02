// SPDX-License-Identifier: Apache-2.0 OR MIT
use super::config::HfConfig;
use super::discovery::{classify_tensor, discover_candidates};
use super::manifest::{
    INDEX_UNREFERENCED_SHARDS_KEY, build_manifest, merge_shard_metadata,
    shard_metadata_namespaced_key,
};
use super::paths::{canonical_existing_or_parent, parent_or_current};
use super::test_support::{temp_dir, write_hf_config, write_safetensors, write_safetensors_bytes};
use super::*;
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

#[test]
fn parses_single_file_manifest_and_labels_candidates() {
    let dir = temp_dir("single");
    let path = dir.join("model.safetensors");
    write_safetensors(
        &path,
        r#"{
                "__metadata__": {"format": "pt"},
                "model.layers.0.mlp.gate.weight": {"dtype": "F32", "shape": [64, 2048], "data_offsets": [0, 524288]},
                "model.layers.0.mlp.experts.0.w1.weight": {"dtype": "F16", "shape": [2048, 4096], "data_offsets": [524288, 17301504]}
            }"#,
        17_301_504,
    );

    let manifest = inspect_safetensors_checkpoint(&path).unwrap();
    assert_eq!(manifest.checkpoint.input_kind, "single_file");
    assert_eq!(manifest.checkpoint.shard_count, 1);
    assert_eq!(manifest.checkpoint.tensor_count, 2);
    assert_eq!(manifest.checkpoint.metadata.get("format").unwrap(), "pt");
    assert_eq!(
        manifest.tensors[0].name,
        "model.layers.0.mlp.experts.0.w1.weight"
    );
    assert_eq!(manifest.tensors[0].source_shard, "model.safetensors");
    assert_eq!(manifest.candidates.router_tensors.len(), 1);
    assert_eq!(manifest.candidates.expert_tensors.len(), 1);
    assert_eq!(
        manifest.candidates.detected_layout_family,
        Some("generic_moe")
    );
    assert_eq!(manifest.candidates.router_candidates[0].layer_hint, Some(0));
    assert_eq!(manifest.candidates.expert_groups.len(), 1);
    assert_eq!(manifest.candidates.expert_groups[0].layer_hint, Some(0));
    assert_eq!(manifest.candidates.expert_groups[0].expert_indices, vec![0]);
}

#[test]
fn reads_hugging_face_index_and_orders_deterministically() {
    let dir = temp_dir("index");
    let shard_a = dir.join("model-00001-of-00002.safetensors");
    let shard_b = dir.join("model-00002-of-00002.safetensors");
    write_safetensors(
        &shard_b,
        r#"{"z.weight": {"dtype": "F16", "shape": [2, 2], "data_offsets": [0, 8]}}"#,
        8,
    );
    write_safetensors(
        &shard_a,
        r#"{"a.router.weight": {"dtype": "F32", "shape": [8, 2048], "data_offsets": [0, 65536]}}"#,
        65_536,
    );
    fs::write(
        dir.join("model.safetensors.index.json"),
        r#"{
                "metadata": {"total_size": 65544},
                "weight_map": {
                    "z.weight": "model-00002-of-00002.safetensors",
                    "a.router.weight": "model-00001-of-00002.safetensors"
                }
            }"#,
    )
    .unwrap();

    let manifest = inspect_safetensors_checkpoint(&dir).unwrap();
    assert_eq!(manifest.checkpoint.input_kind, "hf_index");
    assert_eq!(
        manifest.checkpoint.index_file.as_deref(),
        Some("model.safetensors.index.json")
    );
    assert_eq!(manifest.checkpoint.shard_count, 2);
    assert_eq!(
        manifest
            .checkpoint
            .metadata
            .get("index:total_size")
            .unwrap(),
        "65544"
    );
    assert_eq!(manifest.tensors[0].name, "a.router.weight");
    assert_eq!(manifest.tensors[1].name, "z.weight");
    assert_eq!(manifest.candidates.router_tensors, vec!["a.router.weight"]);
}

#[test]
fn user_index_metadata_cannot_spoof_unreferenced_shards() {
    let dir = temp_dir("reserved-index-metadata");
    write_safetensors(
        &dir.join("model.safetensors"),
        r#"{"a.weight": {"dtype": "F16", "shape": [1], "data_offsets": [0, 2]}}"#,
        2,
    );
    fs::write(
        dir.join("model.safetensors.index.json"),
        r#"{
                "metadata": {"unreferenced_shards": "spoof.safetensors"},
                "weight_map": {"a.weight": "model.safetensors"}
            }"#,
    )
    .unwrap();

    let manifest = inspect_safetensors_checkpoint(&dir).unwrap();
    assert!(
        !manifest
            .checkpoint
            .metadata
            .contains_key(INDEX_UNREFERENCED_SHARDS_KEY)
    );
}

#[test]
fn index_shard_paths_are_normalized_before_deduplication() {
    let dir = temp_dir("normalized-index-paths");
    write_safetensors(
        &dir.join("model.safetensors"),
        r#"{
                "a.weight": {"dtype": "F16", "shape": [1], "data_offsets": [0, 2]},
                "b.weight": {"dtype": "F16", "shape": [1], "data_offsets": [2, 4]}
            }"#,
        4,
    );
    fs::write(
        dir.join("model.safetensors.index.json"),
        r#"{
                "weight_map": {
                    "a.weight": "model.safetensors",
                    "b.weight": "./model.safetensors"
                }
            }"#,
    )
    .unwrap();

    let manifest = inspect_safetensors_checkpoint(&dir).unwrap();
    assert_eq!(manifest.checkpoint.shard_count, 1);
    assert_eq!(manifest.checkpoint.tensor_count, 2);
    assert_eq!(manifest.tensors[0].source_shard, "model.safetensors");
    assert_eq!(manifest.tensors[1].source_shard, "model.safetensors");
}

#[test]
fn rejects_tensor_offsets_beyond_data_section() {
    let dir = temp_dir("bounds");
    let path = dir.join("bad.safetensors");
    write_safetensors(
        &path,
        r#"{"bad.weight": {"dtype": "F16", "shape": [4], "data_offsets": [0, 16]}}"#,
        8,
    );

    let err = inspect_safetensors_checkpoint(&path).unwrap_err();
    assert!(
        err.to_string()
            .contains("extends beyond Safetensors data section")
    );
}

#[test]
fn rejects_index_shard_paths_that_escape_checkpoint_directory() {
    let dir = temp_dir("escape");
    fs::write(
        dir.join("model.safetensors.index.json"),
        r#"{
                "weight_map": {
                    "a.weight": "../outside.safetensors"
                }
            }"#,
    )
    .unwrap();

    let err = inspect_safetensors_checkpoint(&dir).unwrap_err();
    assert!(
        err.to_string()
            .contains("must stay within the checkpoint directory")
    );
}

#[cfg(unix)]
#[test]
fn rejects_index_shard_paths_that_escape_via_symlink() {
    let dir = temp_dir("symlink");
    let outside = temp_dir("outside");
    let outside_shard = outside.join("outside.safetensors");
    write_safetensors(
        &outside_shard,
        r#"{"a.weight": {"dtype": "F16", "shape": [1], "data_offsets": [0, 2]}}"#,
        2,
    );
    std::os::unix::fs::symlink(&outside_shard, dir.join("link.safetensors")).unwrap();
    fs::write(
        dir.join("model.safetensors.index.json"),
        r#"{
                "weight_map": {
                    "a.weight": "link.safetensors"
                }
            }"#,
    )
    .unwrap();

    let err = inspect_safetensors_checkpoint(&dir).unwrap_err();
    assert!(
        err.to_string()
            .contains("must stay within the checkpoint directory")
    );
}

#[cfg(unix)]
#[test]
fn rejects_directory_index_that_escapes_via_symlink() {
    let dir = temp_dir("index-symlink");
    let outside = temp_dir("outside-index");
    fs::write(
        outside.join("external.safetensors.index.json"),
        r#"{"weight_map": {}}"#,
    )
    .unwrap();
    std::os::unix::fs::symlink(
        outside.join("external.safetensors.index.json"),
        dir.join("model.safetensors.index.json"),
    )
    .unwrap();

    let err = inspect_safetensors_checkpoint(&dir).unwrap_err();
    assert!(
        err.to_string()
            .contains("must stay within the checkpoint directory")
    );
}

#[test]
fn rejects_multiple_directory_indexes() {
    let dir = temp_dir("multiple-indexes");
    fs::write(
        dir.join("a.safetensors.index.json"),
        r#"{"weight_map": {}}"#,
    )
    .unwrap();
    fs::write(
        dir.join("b.safetensors.index.json"),
        r#"{"weight_map": {}}"#,
    )
    .unwrap();

    let err = inspect_safetensors_checkpoint(&dir).unwrap_err();
    assert!(err.to_string().contains("multiple Safetensors index files"));
}

#[test]
fn hf_index_filters_extra_tensors_and_requires_mapped_tensors() {
    let dir = temp_dir("index-filter");
    let shard = dir.join("model-00001-of-00001.safetensors");
    write_safetensors(
        &shard,
        r#"{
                "a.router.weight": {"dtype": "F32", "shape": [8, 2048], "data_offsets": [0, 65536]},
                "stale.router.weight": {"dtype": "F32", "shape": [8, 2048], "data_offsets": [65536, 131072]}
            }"#,
        131_072,
    );
    write_safetensors(
        &dir.join("unused.safetensors"),
        r#"{"unused.weight": {"dtype": "F16", "shape": [1], "data_offsets": [0, 2]}}"#,
        2,
    );
    let index = dir.join("model.safetensors.index.json");
    fs::write(
        &index,
        r#"{"weight_map": {"a.router.weight": "model-00001-of-00001.safetensors"}}"#,
    )
    .unwrap();

    let manifest = inspect_safetensors_checkpoint(&dir).unwrap();
    assert_eq!(manifest.checkpoint.tensor_count, 1);
    assert_eq!(manifest.tensors[0].name, "a.router.weight");
    assert_eq!(
        manifest
            .checkpoint
            .metadata
            .get(INDEX_UNREFERENCED_SHARDS_KEY)
            .map(String::as_str),
        Some(r#"["unused.safetensors"]"#)
    );

    fs::write(
        &index,
        r#"{"weight_map": {"missing.weight": "model-00001-of-00001.safetensors"}}"#,
    )
    .unwrap();
    let err = inspect_safetensors_checkpoint(&dir).unwrap_err();
    assert!(err.to_string().contains("does not contain it"));
}

#[test]
fn rejects_shape_dtype_byte_size_mismatch() {
    let dir = temp_dir("byte-size");
    let path = dir.join("bad.safetensors");
    write_safetensors(
        &path,
        r#"{"bad.weight": {"dtype": "F16", "shape": [4], "data_offsets": [0, 4]}}"#,
        4,
    );

    let err = inspect_safetensors_checkpoint(&path).unwrap_err();
    assert!(err.to_string().contains("byte size mismatch"));
}

#[test]
fn rejects_unknown_safetensors_dtype() {
    let dir = temp_dir("unknown-dtype");
    let path = dir.join("bad.safetensors");
    write_safetensors(
        &path,
        r#"{"bad.weight": {"dtype": "F128", "shape": [1], "data_offsets": [0, 2]}}"#,
        2,
    );

    let err = inspect_safetensors_checkpoint(&path).unwrap_err();
    assert!(err.to_string().contains("unsupported Safetensors dtype"));
}

#[test]
fn rejects_duplicate_header_keys() {
    let dir = temp_dir("duplicate-keys");
    let path = dir.join("bad.safetensors");
    write_safetensors(
        &path,
        r#"{
                "dup.weight": {"dtype": "F16", "shape": [1], "data_offsets": [0, 2]},
                "dup.weight": {"dtype": "F16", "shape": [1], "data_offsets": [2, 4]}
            }"#,
        4,
    );

    let err = inspect_safetensors_checkpoint(&path).unwrap_err();
    assert!(err.to_string().contains("duplicate JSON key"));
}

#[test]
fn rejects_non_string_metadata_values() {
    let dir = temp_dir("metadata-values");
    let path = dir.join("bad.safetensors");
    write_safetensors(
        &path,
        r#"{
                "__metadata__": {"format": 1},
                "a.weight": {"dtype": "F16", "shape": [1], "data_offsets": [0, 2]}
            }"#,
        2,
    );

    let err = inspect_safetensors_checkpoint(&path).unwrap_err();
    assert!(err.to_string().contains("must be a string"));
}

#[test]
fn rejects_overlapping_tensor_ranges() {
    let dir = temp_dir("overlap");
    let path = dir.join("bad.safetensors");
    write_safetensors(
        &path,
        r#"{
                "a.weight": {"dtype": "F16", "shape": [4], "data_offsets": [0, 8]},
                "b.weight": {"dtype": "F16", "shape": [4], "data_offsets": [4, 12]}
            }"#,
        12,
    );

    let err = inspect_safetensors_checkpoint(&path).unwrap_err();
    assert!(err.to_string().contains("data range overlaps"));
}

#[test]
fn rejects_tensor_data_gaps_and_trailing_bytes() {
    let dir = temp_dir("data-gaps");
    let gap_path = dir.join("gap.safetensors");
    write_safetensors(
        &gap_path,
        r#"{
                "a.weight": {"dtype": "F16", "shape": [1], "data_offsets": [0, 2]},
                "b.weight": {"dtype": "F16", "shape": [1], "data_offsets": [4, 6]}
            }"#,
        6,
    );
    let err = inspect_safetensors_checkpoint(&gap_path).unwrap_err();
    assert!(err.to_string().contains("leaves a gap"));

    let trailing_path = dir.join("trailing.safetensors");
    write_safetensors(
        &trailing_path,
        r#"{"a.weight": {"dtype": "F16", "shape": [1], "data_offsets": [0, 2]}}"#,
        4,
    );
    let err = inspect_safetensors_checkpoint(&trailing_path).unwrap_err();
    assert!(err.to_string().contains("data section is 4 bytes"));
}

#[test]
fn conflicting_shard_metadata_is_only_namespaced() {
    let dir = temp_dir("metadata-conflict");
    write_safetensors(
        &dir.join("a.safetensors"),
        r#"{
                "__metadata__": {"format": "pt"},
                "a.weight": {"dtype": "F16", "shape": [1], "data_offsets": [0, 2]}
            }"#,
        2,
    );
    write_safetensors(
        &dir.join("b.safetensors"),
        r#"{
                "__metadata__": {"format": "np"},
                "b.weight": {"dtype": "F16", "shape": [1], "data_offsets": [0, 2]}
            }"#,
        2,
    );

    let manifest = inspect_safetensors_checkpoint(&dir).unwrap();
    assert!(!manifest.checkpoint.metadata.contains_key("format"));
    assert_eq!(
        manifest
            .checkpoint
            .metadata
            .get(&shard_metadata_namespaced_key("a.safetensors", "format"))
            .map(String::as_str),
        Some("pt")
    );
    assert_eq!(
        manifest
            .checkpoint
            .metadata
            .get(&shard_metadata_namespaced_key("b.safetensors", "format"))
            .map(String::as_str),
        Some("np")
    );
}

#[test]
fn colon_in_metadata_logical_key_conflict_does_not_restore_base_key() {
    let mut metadata = BTreeMap::new();
    let mut first = BTreeMap::new();
    first.insert("my:key".to_string(), "a".to_string());
    merge_shard_metadata(&mut metadata, "x.safetensors", first);

    let mut second = BTreeMap::new();
    second.insert("my:key".to_string(), "b".to_string());
    merge_shard_metadata(&mut metadata, "y.safetensors", second);

    assert!(!metadata.contains_key("my:key"));

    let mut third = BTreeMap::new();
    third.insert("my:key".to_string(), "c".to_string());
    merge_shard_metadata(&mut metadata, "z.safetensors", third);

    assert!(!metadata.contains_key("my:key"));
    assert_eq!(
        metadata
            .get(&shard_metadata_namespaced_key("z.safetensors", "my:key"))
            .map(String::as_str),
        Some("c")
    );
}

#[test]
fn rejects_manifest_output_that_overwrites_unreferenced_index_shard_with_comma_in_name() {
    let dir = temp_dir("output-unreferenced-comma");
    let main_shard = dir.join("model-00001-of-00001.safetensors");
    write_safetensors(
        &main_shard,
        r#"{"a.weight": {"dtype": "F16", "shape": [1], "data_offsets": [0, 2]}}"#,
        2,
    );
    let unreferenced = dir.join("a,b.safetensors");
    write_safetensors(
        &unreferenced,
        r#"{"b.weight": {"dtype": "F16", "shape": [1], "data_offsets": [0, 2]}}"#,
        2,
    );
    fs::write(
        dir.join("model.safetensors.index.json"),
        r#"{"weight_map": {"a.weight": "model-00001-of-00001.safetensors"}}"#,
    )
    .unwrap();

    let err = write_safetensors_manifest(&dir, &unreferenced).unwrap_err();
    assert!(err.to_string().contains("must not overwrite"));
}

#[test]
fn shard_metadata_key_detection_uses_exact_key_match() {
    let mut metadata = BTreeMap::new();
    let mut first = BTreeMap::new();
    first.insert("xfoo".to_string(), "1".to_string());
    merge_shard_metadata(&mut metadata, "a.safetensors", first);

    let mut second = BTreeMap::new();
    second.insert("foo".to_string(), "2".to_string());
    merge_shard_metadata(&mut metadata, "b.safetensors", second);

    assert_eq!(metadata.get("foo").map(String::as_str), Some("2"));
}

#[test]
fn directory_scan_ignores_non_file_safetensors_entries() {
    let dir = temp_dir("non-file-entry");
    fs::create_dir(dir.join("scratch.safetensors")).unwrap();
    write_safetensors(
        &dir.join("model.safetensors"),
        r#"{"a.weight": {"dtype": "F16", "shape": [1], "data_offsets": [0, 2]}}"#,
        2,
    );

    let manifest = inspect_safetensors_checkpoint(&dir).unwrap();
    assert_eq!(manifest.checkpoint.shard_count, 1);
    assert_eq!(manifest.checkpoint.tensor_count, 1);
    assert_eq!(manifest.tensors[0].source_shard, "model.safetensors");
}

#[test]
fn bare_paths_resolve_parent_as_current_directory() {
    assert_eq!(
        parent_or_current(Path::new("model.safetensors")),
        Path::new(".")
    );
    assert!(canonical_existing_or_parent(Path::new("manifest.json")).is_ok());
}

#[test]
fn rejects_manifest_output_that_overwrites_checkpoint_file() {
    let dir = temp_dir("output-conflict");
    let path = dir.join("model.safetensors");
    write_safetensors(
        &path,
        r#"{"a.weight": {"dtype": "F16", "shape": [1], "data_offsets": [0, 2]}}"#,
        2,
    );

    let err = write_safetensors_manifest(&path, &path).unwrap_err();
    assert!(err.to_string().contains("must not overwrite"));
}

#[test]
fn rejects_manifest_output_that_overwrites_unreferenced_index_shard() {
    let dir = temp_dir("output-unreferenced");
    let main_shard = dir.join("model-00001-of-00001.safetensors");
    write_safetensors(
        &main_shard,
        r#"{"a.weight": {"dtype": "F16", "shape": [1], "data_offsets": [0, 2]}}"#,
        2,
    );
    let unreferenced = dir.join("unused.safetensors");
    write_safetensors(
        &unreferenced,
        r#"{"b.weight": {"dtype": "F16", "shape": [1], "data_offsets": [0, 2]}}"#,
        2,
    );
    fs::write(
        dir.join("model.safetensors.index.json"),
        r#"{"weight_map": {"a.weight": "model-00001-of-00001.safetensors"}}"#,
    )
    .unwrap();

    let err = write_safetensors_manifest(&dir, &unreferenced).unwrap_err();
    assert!(err.to_string().contains("must not overwrite"));
}

#[test]
fn accepts_current_additional_safetensors_dtypes() {
    let dir = temp_dir("extra-dtypes");
    let path = dir.join("model.safetensors");
    write_safetensors(
        &path,
        r#"{
                "a.weight": {"dtype": "F8_E8M0", "shape": [4], "data_offsets": [0, 4]},
                "b.weight": {"dtype": "C64", "shape": [2], "data_offsets": [4, 20]}
            }"#,
        20,
    );

    let manifest = inspect_safetensors_checkpoint(&path).unwrap();
    assert_eq!(manifest.checkpoint.tensor_count, 2);
}

#[test]
fn generic_dense_ffn_names_are_not_moe_expert_candidates() {
    let labels = classify_tensor("model.layers.0.mlp.gate_proj.weight", &[4096, 2048]);
    assert!(!labels.contains(&"moe_expert_candidate"));

    let labels = classify_tensor("model.layers.0.block_sparse_moe.gate.weight", &[8, 2048]);
    assert!(labels.contains(&"moe_router_candidate"));
    assert!(!labels.contains(&"moe_expert_candidate"));
}

#[test]
fn detects_deepseek_v3_family_block_sparse_moe_layout() {
    let dir = temp_dir("deepseek-v3-family");
    let path = dir.join("model.safetensors");
    write_safetensors(
        &path,
        r#"{
                "model.layers.0.block_sparse_moe.gate.weight": {"dtype": "F32", "shape": [8, 2048], "data_offsets": [0, 65536]},
                "model.layers.0.block_sparse_moe.experts.0.w1.weight": {"dtype": "F16", "shape": [2, 2], "data_offsets": [65536, 65544]},
                "model.layers.0.block_sparse_moe.experts.0.w2.weight": {"dtype": "F16", "shape": [2, 2], "data_offsets": [65544, 65552]},
                "model.layers.0.block_sparse_moe.experts.0.w3.weight": {"dtype": "F16", "shape": [2, 2], "data_offsets": [65552, 65560]}
            }"#,
        65_560,
    );

    let manifest = inspect_safetensors_checkpoint(&path).unwrap();
    assert_eq!(
        manifest.candidates.detected_layout_family,
        Some("deepseek_v3_family")
    );
    assert_eq!(
        manifest.candidates.router_tensors,
        vec!["model.layers.0.block_sparse_moe.gate.weight"]
    );
    assert_eq!(manifest.candidates.expert_groups.len(), 1);
    let group = &manifest.candidates.expert_groups[0];
    assert_eq!(group.group_key, "model.layers.0.block_sparse_moe");
    assert_eq!(group.layer_hint, Some(0));
    assert_eq!(group.expert_indices, vec![0]);
    assert_eq!(group.weight_kinds, vec!["w1", "w2", "w3"]);
    assert_eq!(group.source_shards, vec!["model.safetensors"]);
}

#[test]
fn detects_requested_named_moe_families() {
    for (family, router_name, expert_name) in [
        (
            "phimoe",
            "phimoe.model.layers.0.moe.gate.weight",
            "phimoe.model.layers.0.moe.experts.1.fc1.weight",
        ),
        (
            "granitemoe",
            "granitemoe.model.layers.0.moe.gate.weight",
            "granitemoe.model.layers.0.moe.experts.1.gate_proj.weight",
        ),
        (
            "afmoe",
            "afmoe.model.layers.0.moe.gating.weight",
            "afmoe.model.layers.0.moe.experts.1.linear_1.weight",
        ),
        (
            "lfm2_moe",
            "lfm2_moe.model.layers.0.feed_forward.gate.weight",
            "lfm2_moe.model.layers.0.feed_forward.experts.1.ffn_up.weight",
        ),
    ] {
        let router = SafetensorsTensorRecord {
            name: router_name.to_string(),
            dtype: "F32".into(),
            shape: vec![16, 2048],
            byte_size: 131_072,
            source_shard: "router.safetensors".into(),
            data_offsets: [0, 131_072],
            labels: classify_tensor(router_name, &[16, 2048]),
        };
        let expert = SafetensorsTensorRecord {
            name: expert_name.to_string(),
            dtype: "F16".into(),
            shape: vec![2, 2],
            byte_size: 8,
            source_shard: "experts.safetensors".into(),
            data_offsets: [0, 8],
            labels: classify_tensor(expert_name, &[2, 2]),
        };

        let candidates = discover_candidates(&[router, expert]);
        assert_eq!(candidates.detected_layout_family, Some(family));
        assert_eq!(candidates.router_tensors, vec![router_name]);
        assert_eq!(candidates.expert_tensors, vec![expert_name]);
        assert_eq!(candidates.expert_groups.len(), 1);
        assert_eq!(candidates.expert_groups[0].expert_indices, vec![1]);
    }
}

#[test]
fn shape_guard_prevents_ambiguous_dense_gate_router_candidate() {
    let labels = classify_tensor("model.layers.0.mlp.gate.weight", &[4096, 2048]);
    assert!(!labels.contains(&"moe_router_candidate"));

    let labels = classify_tensor("model.layers.0.mlp.gate.weight", &[8, 2048]);
    assert!(labels.contains(&"moe_router_candidate"));

    let labels = classify_tensor("model.layers.0.gating.weight", &[4096, 2048]);
    assert!(!labels.contains(&"moe_router_candidate"));
}

#[test]
fn discovery_does_not_double_list_experts_as_routers() {
    let expert_name = "model.router.moe.experts.0.down_proj.weight";
    let tensor = SafetensorsTensorRecord {
        name: expert_name.to_string(),
        dtype: "F16".into(),
        shape: vec![2, 2],
        byte_size: 8,
        source_shard: "experts.safetensors".into(),
        data_offsets: [0, 8],
        labels: classify_tensor(expert_name, &[2, 2]),
    };

    let manifest = build_manifest(
        "single_file",
        None,
        vec![PathBuf::from("experts.safetensors")],
        BTreeMap::new(),
        vec![tensor],
    );
    assert_eq!(manifest.candidates.router_tensors, Vec::<String>::new());
    assert_eq!(manifest.candidates.expert_tensors, vec![expert_name]);
    assert_eq!(manifest.tensors[0].labels, vec!["moe_expert_candidate"]);
}

#[test]
fn named_family_requires_router_and_expert_evidence() {
    let tensor = SafetensorsTensorRecord {
        name: "granitemoe.adapter.weight".into(),
        dtype: "F16".into(),
        shape: vec![2, 2],
        byte_size: 8,
        source_shard: "model.safetensors".into(),
        data_offsets: [0, 8],
        labels: classify_tensor("granitemoe.adapter.weight", &[2, 2]),
    };

    let manifest = build_manifest(
        "single_file",
        None,
        vec![PathBuf::from("model.safetensors")],
        BTreeMap::new(),
        vec![tensor],
    );
    assert_eq!(manifest.candidates.detected_layout_family, None);
    assert!(manifest.candidates.router_tensors.is_empty());
    assert!(manifest.candidates.expert_tensors.is_empty());
}

#[test]
fn unknown_expert_weight_kind_is_retained() {
    let router_name = "model.layers.0.moe.gate.weight";
    let expert_name = "model.layers.0.moe.experts.0.adapter.weight";
    let router = SafetensorsTensorRecord {
        name: router_name.into(),
        dtype: "F32".into(),
        shape: vec![4, 2048],
        byte_size: 32_768,
        source_shard: "router.safetensors".into(),
        data_offsets: [0, 32_768],
        labels: classify_tensor(router_name, &[4, 2048]),
    };
    let expert = SafetensorsTensorRecord {
        name: expert_name.into(),
        dtype: "F16".into(),
        shape: vec![2, 2],
        byte_size: 8,
        source_shard: "experts.safetensors".into(),
        data_offsets: [0, 8],
        labels: classify_tensor(expert_name, &[2, 2]),
    };

    let manifest = build_manifest(
        "directory",
        None,
        vec![
            PathBuf::from("router.safetensors"),
            PathBuf::from("experts.safetensors"),
        ],
        BTreeMap::new(),
        vec![router, expert],
    );
    assert_eq!(
        manifest.candidates.detected_layout_family,
        Some("generic_moe")
    );
    assert_eq!(manifest.candidates.expert_tensors, vec![expert_name]);
    assert_eq!(
        manifest.candidates.expert_groups[0].weight_kinds,
        vec!["unknown"]
    );
}

#[test]
fn expert_groups_keep_parallel_layer_stacks_separate() {
    let encoder = "encoder.layers.0.moe.experts.0.w1.weight";
    let decoder = "decoder.layers.0.moe.experts.0.w1.weight";
    let tensors = [encoder, decoder]
        .into_iter()
        .map(|name| SafetensorsTensorRecord {
            name: name.into(),
            dtype: "F16".into(),
            shape: vec![2, 2],
            byte_size: 8,
            source_shard: "experts.safetensors".into(),
            data_offsets: [0, 8],
            labels: classify_tensor(name, &[2, 2]),
        })
        .collect::<Vec<_>>();

    let manifest = build_manifest(
        "single_file",
        None,
        vec![PathBuf::from("experts.safetensors")],
        BTreeMap::new(),
        tensors,
    );
    let group_keys = manifest
        .candidates
        .expert_groups
        .iter()
        .map(|group| group.group_key.as_str())
        .collect::<Vec<_>>();
    assert_eq!(
        group_keys,
        vec!["decoder.layers.0.moe", "encoder.layers.0.moe"]
    );
}

#[test]
fn groups_experts_across_hugging_face_index_shards() {
    let dir = temp_dir("hf-expert-groups");
    let router_shard = dir.join("model-00001-of-00002.safetensors");
    let expert_shard = dir.join("model-00002-of-00002.safetensors");
    write_safetensors(
        &router_shard,
        r#"{"model.layers.2.block_sparse_moe.gate.weight": {"dtype": "F32", "shape": [8, 2048], "data_offsets": [0, 65536]}}"#,
        65_536,
    );
    write_safetensors(
        &expert_shard,
        r#"{
                "model.layers.2.block_sparse_moe.experts.0.w1.weight": {"dtype": "F16", "shape": [2, 2], "data_offsets": [0, 8]},
                "model.layers.2.block_sparse_moe.experts.1.w1.weight": {"dtype": "F16", "shape": [2, 2], "data_offsets": [8, 16]}
            }"#,
        16,
    );
    fs::write(
            dir.join("model.safetensors.index.json"),
            r#"{
                "weight_map": {
                    "model.layers.2.block_sparse_moe.gate.weight": "model-00001-of-00002.safetensors",
                    "model.layers.2.block_sparse_moe.experts.0.w1.weight": "model-00002-of-00002.safetensors",
                    "model.layers.2.block_sparse_moe.experts.1.w1.weight": "model-00002-of-00002.safetensors"
                }
            }"#,
        )
        .unwrap();

    let manifest = inspect_safetensors_checkpoint(&dir).unwrap();
    assert_eq!(manifest.checkpoint.input_kind, "hf_index");
    assert_eq!(manifest.candidates.expert_groups.len(), 1);
    let group = &manifest.candidates.expert_groups[0];
    assert_eq!(group.group_key, "model.layers.2.block_sparse_moe");
    assert_eq!(group.expert_indices, vec![0, 1]);
    assert_eq!(
        group.source_shards,
        vec!["model-00002-of-00002.safetensors"]
    );
    assert!(
        manifest
            .tensors
            .iter()
            .all(|tensor| tensor.source_shard.ends_with(".safetensors"))
    );
}

#[test]
fn hf_config_accepts_skywork_singleton_num_experts_array() {
    // Skywork/Skywork-MoE-Base-FP8 ships num_experts as [16], not a scalar.
    let cfg: HfConfig = serde_json::from_str(
        r#"{
            "architectures": ["SkyworkForCausalLM"],
            "hidden_size": 2048,
            "num_hidden_layers": 4,
            "vocab_size": 32000,
            "num_experts": [16],
            "num_experts_per_tok": 2
        }"#,
    )
    .expect("Skywork-style config should parse");
    assert_eq!(cfg.resolved_num_experts(), 16);
    assert_eq!(cfg.resolved_expert_used_count(), 2);
}

#[test]
fn hf_config_accepts_scalar_num_experts() {
    let cfg: HfConfig = serde_json::from_str(
        r#"{
            "architectures": ["OlmoeForCausalLM"],
            "hidden_size": 2048,
            "num_hidden_layers": 16,
            "vocab_size": 100278,
            "num_experts": 64,
            "num_experts_per_tok": 8
        }"#,
    )
    .expect("scalar num_experts should parse");
    assert_eq!(cfg.resolved_num_experts(), 64);
    assert_eq!(cfg.resolved_expert_used_count(), 8);
}

#[test]
fn hf_config_rejects_multi_element_num_experts_array() {
    let err = serde_json::from_str::<HfConfig>(
        r#"{
            "architectures": ["SkyworkForCausalLM"],
            "hidden_size": 2048,
            "num_hidden_layers": 4,
            "vocab_size": 32000,
            "num_experts": [8, 16]
        }"#,
    )
    .expect_err("multi-element num_experts array must fail");
    assert!(
        err.to_string().contains("exactly one element"),
        "unexpected error: {err}"
    );
}

#[test]
fn hf_config_reads_gemma4_top_k_experts_from_text_config() {
    // google/gemma-4-26B-A4B-it: MoE fields under text_config, fan-out as top_k_experts.
    let cfg: HfConfig = serde_json::from_str(
        r#"{
            "architectures": ["Gemma4ForConditionalGeneration"],
            "text_config": {
                "hidden_size": 3840,
                "num_hidden_layers": 30,
                "vocab_size": 262144,
                "num_experts": 128,
                "top_k_experts": 8
            }
        }"#,
    )
    .expect("Gemma4 multimodal config should parse");
    assert_eq!(cfg.resolved_num_experts(), 128);
    assert_eq!(cfg.resolved_expert_used_count(), 8);
    assert_eq!(cfg.resolved_hidden_size(), Some(3840));
}

#[test]
fn hf_config_reads_step_moe_num_experts_and_top_k() {
    // stepfun-ai/Step-3.5-Flash: moe_num_experts / moe_top_k (not num_experts*).
    let cfg: HfConfig = serde_json::from_str(
        r#"{
            "architectures": ["Step3p5ForCausalLM"],
            "hidden_size": 2560,
            "num_hidden_layers": 45,
            "vocab_size": 128815,
            "moe_num_experts": 288,
            "moe_top_k": 8
        }"#,
    )
    .expect("Step-3.5-Flash config should parse");
    assert_eq!(cfg.resolved_num_experts(), 288);
    assert_eq!(cfg.resolved_expert_used_count(), 8);
}

#[cfg(windows)]
#[test]
fn windows_same_file_detects_hardlinks_via_file_id() {
    let dir = temp_dir("windows-hardlink");
    let original = dir.join("original.safetensors");
    write_safetensors(
        &original,
        r#"{"a.weight": {"dtype": "F16", "shape": [1], "data_offsets": [0, 2]}}"#,
        2,
    );

    let hardlink = dir.join("hardlink.safetensors");
    std::fs::hard_link(&original, &hardlink).unwrap();

    // Verify paths_refer_to_same_file returns true for hardlinks (uses Win32 FFI file-id comparison)
    assert!(super::paths::paths_refer_to_same_file(&original, &hardlink).unwrap());

    // Verify different files return false
    let different = dir.join("different.safetensors");
    write_safetensors(
        &different,
        r#"{"b.weight": {"dtype": "F16", "shape": [1], "data_offsets": [0, 2]}}"#,
        2,
    );
    assert!(!super::paths::paths_refer_to_same_file(&original, &different).unwrap());
}

#[test]
fn raw_shard_prefixed_metadata_key_does_not_hide_a_later_base_key() {
    let mut metadata = BTreeMap::new();
    merge_shard_metadata(
        &mut metadata,
        "s1.safetensors",
        BTreeMap::from([("shard:a:b".to_string(), "x".to_string())]),
    );
    merge_shard_metadata(
        &mut metadata,
        "s2.safetensors",
        BTreeMap::from([("b".to_string(), "y".to_string())]),
    );

    assert_eq!(metadata.get("b").map(String::as_str), Some("y"));
}

#[test]
fn bf16_to_f32_matches_reference() {
    for (bits, expected) in [
        (0x0000, 0.0),
        (0x3F80, 1.0),
        (0xBF80, -1.0),
        (0x4000, 2.0),
        (0x7F80, f32::INFINITY),
        (0x0080, f32::MIN_POSITIVE),
    ] {
        assert_eq!(bf16_to_f32(bits), expected, "bits={bits:#06x}");
    }

    let subnormal = bf16_to_f32(0x0001);
    assert!(subnormal > 0.0);
    assert!(subnormal.is_subnormal());
    assert!(bf16_to_f32(0x7FC0).is_nan());
}

fn write_minimal_hf_config(dir: &Path) {
    write_hf_config(
        dir,
        &serde_json::json!({"architectures": ["TestModel"], "hidden_size": 16, "num_hidden_layers": 1, "vocab_size": 8}),
    );
}

fn one_f32_header(end_offset: u64) -> serde_json::Value {
    serde_json::json!({"a": {"dtype": "F32", "shape": [1], "data_offsets": [0, end_offset]}})
}

#[test]
fn from_directory_without_index_lets_the_last_shard_own_a_duplicate_tensor() {
    let dir = temp_dir("load-last-shard-wins");
    write_minimal_hf_config(&dir);
    write_safetensors_bytes(
        &dir.join("model-00001-of-00002.safetensors"),
        &one_f32_header(4),
        &1.0f32.to_le_bytes(),
    );
    write_safetensors_bytes(
        &dir.join("model-00002-of-00002.safetensors"),
        &one_f32_header(4),
        &2.0f32.to_le_bytes(),
    );

    let path = dir.to_str().unwrap();
    let checkpoint = MappedSafetensorsCheckpoint::from_directory(path).unwrap();
    assert_eq!(checkpoint.extract_tensor_f32("a", path).unwrap(), vec![2.0]);
}

#[test]
fn from_directory_without_index_ignores_a_malformed_overridden_entry() {
    // The same entry is fatal when its shard owns the tensor, so this fixture
    // really is malformed and the load below succeeds only by skipping it.
    let alone = temp_dir("load-malformed-owner");
    write_minimal_hf_config(&alone);
    write_safetensors_bytes(
        &alone.join("model-00001-of-00002.safetensors"),
        &one_f32_header(999),
        &1.0f32.to_le_bytes(),
    );
    let err = match MappedSafetensorsCheckpoint::from_directory(alone.to_str().unwrap()) {
        Ok(_) => panic!("a malformed owned entry must fail the load"),
        Err(err) => err.to_string(),
    };
    assert!(!err.contains("config.json"), "{err}");

    let dir = temp_dir("load-malformed-overridden");
    write_minimal_hf_config(&dir);
    write_safetensors_bytes(
        &dir.join("model-00001-of-00002.safetensors"),
        &one_f32_header(999),
        &1.0f32.to_le_bytes(),
    );
    write_safetensors_bytes(
        &dir.join("model-00002-of-00002.safetensors"),
        &one_f32_header(4),
        &2.0f32.to_le_bytes(),
    );

    let path = dir.to_str().unwrap();
    let checkpoint = MappedSafetensorsCheckpoint::from_directory(path).unwrap();
    assert_eq!(checkpoint.extract_tensor_f32("a", path).unwrap(), vec![2.0]);
}

#[test]
fn from_directory_with_index_ignores_a_malformed_tensor_outside_the_weight_map() {
    let dir = temp_dir("load-indexed-skip");
    write_minimal_hf_config(&dir);
    let shard = "model-00001-of-00001.safetensors";
    write_safetensors_bytes(
        &dir.join(shard),
        &serde_json::json!({
            "a": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]},
            "not_indexed": {"dtype": "F32", "shape": [1], "data_offsets": [0, 999]},
        }),
        &3.0f32.to_le_bytes(),
    );
    fs::write(
        dir.join("model.safetensors.index.json"),
        serde_json::to_vec(&serde_json::json!({"weight_map": {"a": shard}})).unwrap(),
    )
    .unwrap();

    let path = dir.to_str().unwrap();
    let checkpoint = MappedSafetensorsCheckpoint::from_directory(path).unwrap();
    assert_eq!(checkpoint.extract_tensor_f32("a", path).unwrap(), vec![3.0]);
    assert!(checkpoint.tensor_info("not_indexed").is_none());
}
