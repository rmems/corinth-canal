// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Header-only Safetensors inspection and deterministic manifest generation.

use super::discovery::{classify_tensor, discover_candidates};
use super::json::{parse_json_rejecting_duplicate_keys, parse_metadata_object, stringify_metadata};
use super::model_load;
use super::paths::{
    index_shard_path, is_safetensors_index, parent_or_current, validate_path_stays_under_root,
};
use super::relative_path;
use super::validate::{
    expected_tensor_byte_size, reject_output_checkpoint_conflict, reject_tensor_data_ranges,
};
use crate::error::{HybridError, Result};
use serde::Deserialize;
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File};
use std::io::Read;
use std::path::{Path, PathBuf};

use super::discovery::SafetensorsCandidateSummary;

pub(super) const SAFETENSORS_EXTENSION: &str = "safetensors";
pub(super) const MAX_HEADER_BYTES: usize = 64 * 1024 * 1024;
pub(super) const MAX_INDEX_BYTES: u64 = 64 * 1024 * 1024;
pub(super) const INDEX_UNREFERENCED_SHARDS_KEY: &str = "index:unreferenced_shards";
/// Unambiguous boundary between shard relative path and logical metadata key in
/// `shard:*` manifest keys (avoids ambiguity when the logical key contains `:`).
pub(super) const SHARD_METADATA_KEY_SEP: char = '\u{001e}';

#[derive(Debug, Clone, serde::Serialize)]
pub struct SafetensorsManifest {
    pub manifest_version: u32,
    pub format: &'static str,
    pub checkpoint: SafetensorsCheckpointSource,
    pub tensors: Vec<SafetensorsTensorRecord>,
    pub candidates: SafetensorsCandidateSummary,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct SafetensorsCheckpointSource {
    pub input_kind: String,
    pub index_file: Option<String>,
    pub shard_count: usize,
    pub tensor_count: usize,
    pub metadata: BTreeMap<String, String>,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct SafetensorsTensorRecord {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<usize>,
    pub byte_size: usize,
    pub source_shard: String,
    pub data_offsets: [usize; 2],
    pub labels: Vec<&'static str>,
}

#[derive(Debug, Deserialize)]
pub(super) struct RawIndex {
    #[serde(default)]
    pub(super) metadata: BTreeMap<String, Value>,
    pub(super) weight_map: BTreeMap<String, String>,
}

#[derive(Debug)]
pub(super) struct ShardInspection {
    pub(super) metadata: BTreeMap<String, String>,
    pub(super) tensors: Vec<SafetensorsTensorRecord>,
}

/// Inspect a Safetensors file, sharded Safetensors index, or directory of
/// Safetensors shards and return a deterministic JSON-serializable manifest.
pub fn inspect_safetensors_checkpoint(path: impl AsRef<Path>) -> Result<SafetensorsManifest> {
    let input = path.as_ref();
    let metadata = fs::metadata(input).map_err(|e| model_load(input, e.to_string()))?;

    if metadata.is_dir() {
        return inspect_directory(input);
    }

    if is_safetensors_index(input) {
        let root = parent_or_current(input);
        return inspect_index(root, input);
    }

    if input.extension().and_then(|ext| ext.to_str()) == Some(SAFETENSORS_EXTENSION) {
        return inspect_single_file(input);
    }

    Err(HybridError::UnsupportedFormat(format!(
        "expected .safetensors file, .safetensors.index.json file, or directory, got '{}'",
        input.display()
    )))
}

/// Write a deterministic pretty-printed Safetensors manifest to `output_path`.
pub fn write_safetensors_manifest(
    checkpoint_path: impl AsRef<Path>,
    output_path: impl AsRef<Path>,
) -> Result<SafetensorsManifest> {
    let checkpoint_path = checkpoint_path.as_ref();
    let output_path = output_path.as_ref();
    let manifest = inspect_safetensors_checkpoint(checkpoint_path)?;
    reject_output_checkpoint_conflict(checkpoint_path, output_path, &manifest)?;
    let json = serde_json::to_string_pretty(&manifest)
        .map_err(|e| model_load(output_path, format!("serialize Safetensors manifest: {e}")))?;
    fs::write(output_path, json).map_err(|e| model_load(output_path, e.to_string()))?;
    Ok(manifest)
}

pub(super) fn inspect_single_file(path: &Path) -> Result<SafetensorsManifest> {
    let root = parent_or_current(path);
    let shard = inspect_shard(path, root)?;
    Ok(build_manifest(
        "single_file",
        None,
        vec![path.to_path_buf()],
        shard.metadata,
        shard.tensors,
    ))
}

pub(super) fn inspect_directory(root: &Path) -> Result<SafetensorsManifest> {
    if let Some(index_path) = find_index_file(root)? {
        return inspect_index(root, &index_path);
    }

    let shards = list_safetensors_files(root)?;
    if shards.is_empty() {
        return Err(HybridError::UnsupportedFormat(format!(
            "no .safetensors files found in '{}'",
            root.display()
        )));
    }
    inspect_shards("directory", root, None, shards)
}

pub(super) fn inspect_index(root: &Path, index_path: &Path) -> Result<SafetensorsManifest> {
    let raw = read_index(index_path)?;
    let index_tensor_count = raw.weight_map.len();
    let mut expected_by_shard: BTreeMap<PathBuf, BTreeSet<String>> = BTreeMap::new();
    for (tensor_name, relative) in raw.weight_map {
        let shard_path = index_shard_path(root, index_path, &relative)?;
        expected_by_shard
            .entry(shard_path)
            .or_default()
            .insert(tensor_name);
    }
    let shards = expected_by_shard.keys().cloned().collect::<Vec<_>>();

    let mut metadata = stringify_metadata("index", raw.metadata);
    metadata.remove(INDEX_UNREFERENCED_SHARDS_KEY);
    let indexed_shards = shards.iter().cloned().collect::<BTreeSet<_>>();
    let unreferenced_shards = list_safetensors_files(root)?
        .into_iter()
        .filter(|path| !indexed_shards.contains(path))
        .map(|path| relative_path(&path, root))
        .collect::<Vec<_>>();
    let unreferenced_shards_json = if unreferenced_shards.is_empty() {
        None
    } else {
        Some(serde_json::to_string(&unreferenced_shards).map_err(|e| {
            model_load(
                index_path,
                format!("serialize {INDEX_UNREFERENCED_SHARDS_KEY} list: {e}"),
            )
        })?)
    };
    let index_file = Some(relative_path(index_path, root));
    inspect_index_shards(
        root,
        index_file,
        shards,
        expected_by_shard,
        metadata,
        index_tensor_count,
        unreferenced_shards_json,
    )
}

pub(super) fn inspect_index_shards(
    root: &Path,
    index_file: Option<String>,
    shards: Vec<PathBuf>,
    expected_by_shard: BTreeMap<PathBuf, BTreeSet<String>>,
    mut metadata: BTreeMap<String, String>,
    index_tensor_count: usize,
    unreferenced_shards_json: Option<String>,
) -> Result<SafetensorsManifest> {
    let mut tensors = Vec::new();
    for shard_path in &shards {
        let expected = expected_by_shard.get(shard_path).ok_or_else(|| {
            model_load(
                shard_path,
                "internal error: index shard has no expected tensor set".into(),
            )
        })?;
        let shard = inspect_shard(shard_path, root)?;
        merge_shard_metadata(
            &mut metadata,
            &relative_path(shard_path, root),
            shard.metadata,
        );

        let found = shard
            .tensors
            .iter()
            .map(|tensor| tensor.name.clone())
            .collect::<BTreeSet<_>>();
        if let Some(missing) = expected.difference(&found).next() {
            return Err(model_load(
                shard_path,
                format!(
                    "index maps tensor '{missing}' to this shard, but the shard header does not contain it"
                ),
            ));
        }

        tensors.extend(
            shard
                .tensors
                .into_iter()
                .filter(|tensor| expected.contains(&tensor.name)),
        );
    }

    metadata.insert("index_tensor_count".into(), index_tensor_count.to_string());
    if let Some(encoded) = unreferenced_shards_json {
        metadata.insert(INDEX_UNREFERENCED_SHARDS_KEY.into(), encoded);
    }

    Ok(build_manifest(
        "hf_index", index_file, shards, metadata, tensors,
    ))
}

pub(super) fn inspect_shards(
    input_kind: &str,
    root: &Path,
    index_file: Option<String>,
    shards: Vec<PathBuf>,
) -> Result<SafetensorsManifest> {
    let mut metadata = BTreeMap::new();
    let mut tensors = Vec::new();

    for shard_path in &shards {
        let shard = inspect_shard(shard_path, root)?;
        merge_shard_metadata(
            &mut metadata,
            &relative_path(shard_path, root),
            shard.metadata,
        );
        tensors.extend(shard.tensors);
    }

    Ok(build_manifest(
        input_kind, index_file, shards, metadata, tensors,
    ))
}

pub(super) fn build_manifest(
    input_kind: &str,
    index_file: Option<String>,
    shards: Vec<PathBuf>,
    metadata: BTreeMap<String, String>,
    mut tensors: Vec<SafetensorsTensorRecord>,
) -> SafetensorsManifest {
    tensors.sort_by(|left, right| {
        left.name
            .cmp(&right.name)
            .then(left.source_shard.cmp(&right.source_shard))
    });
    let candidates = discover_candidates(&tensors);

    SafetensorsManifest {
        manifest_version: 2,
        format: "safetensors",
        checkpoint: SafetensorsCheckpointSource {
            input_kind: input_kind.to_string(),
            index_file,
            shard_count: shards.len(),
            tensor_count: tensors.len(),
            metadata,
        },
        tensors,
        candidates,
    }
}

pub(super) fn inspect_shard(path: &Path, root: &Path) -> Result<ShardInspection> {
    let mut file = File::open(path).map_err(|e| model_load(path, e.to_string()))?;
    let file_len = file
        .metadata()
        .map_err(|e| model_load(path, e.to_string()))?
        .len();
    let mut len_bytes = [0u8; 8];
    file.read_exact(&mut len_bytes)
        .map_err(|e| model_load(path, format!("read Safetensors header length: {e}")))?;
    let header_len_u64 = u64::from_le_bytes(len_bytes);
    let header_len = usize::try_from(header_len_u64).map_err(|_| {
        model_load(
            path,
            format!("Safetensors header length {header_len_u64} does not fit in usize"),
        )
    })?;
    if header_len > MAX_HEADER_BYTES {
        return Err(model_load(
            path,
            format!("Safetensors header length {header_len} exceeds limit {MAX_HEADER_BYTES}"),
        ));
    }
    if 8u64
        .checked_add(header_len_u64)
        .is_none_or(|end| end > file_len)
    {
        return Err(model_load(
            path,
            "Safetensors header extends beyond file".into(),
        ));
    }

    let mut header_bytes = vec![0u8; header_len];
    file.read_exact(&mut header_bytes)
        .map_err(|e| model_load(path, format!("read Safetensors header: {e}")))?;
    let header = parse_json_rejecting_duplicate_keys(&header_bytes, path, "Safetensors header")?;
    parse_header(path, root, file_len, header_len_u64, header)
}

pub(super) fn parse_header(
    path: &Path,
    root: &Path,
    file_len: u64,
    header_len: u64,
    header: Value,
) -> Result<ShardInspection> {
    let object = header
        .as_object()
        .ok_or_else(|| model_load(path, "Safetensors header must be a JSON object".to_string()))?;
    let data_len = file_len
        .checked_sub(8)
        .and_then(|len| len.checked_sub(header_len))
        .ok_or_else(|| model_load(path, "Safetensors data range underflow".into()))?;
    let source_shard = relative_path(path, root);
    let mut metadata = BTreeMap::new();
    let mut tensors = Vec::new();

    for (name, value) in object {
        if name == "__metadata__" {
            metadata.extend(parse_metadata_object(path, value)?);
            continue;
        }

        let tensor = value.as_object().ok_or_else(|| {
            model_load(path, format!("tensor '{name}' metadata must be an object"))
        })?;
        let dtype = tensor
            .get("dtype")
            .and_then(Value::as_str)
            .ok_or_else(|| model_load(path, format!("tensor '{name}' is missing dtype")))?
            .to_string();
        let shape = tensor
            .get("shape")
            .and_then(Value::as_array)
            .ok_or_else(|| model_load(path, format!("tensor '{name}' is missing shape")))?
            .iter()
            .map(|dim| {
                dim.as_u64()
                    .and_then(|v| usize::try_from(v).ok())
                    .ok_or_else(|| model_load(path, format!("tensor '{name}' has invalid shape")))
            })
            .collect::<Result<Vec<_>>>()?;
        let offsets = tensor
            .get("data_offsets")
            .and_then(Value::as_array)
            .ok_or_else(|| model_load(path, format!("tensor '{name}' is missing data_offsets")))?;
        if offsets.len() != 2 {
            return Err(model_load(
                path,
                format!("tensor '{name}' data_offsets must have length 2"),
            ));
        }
        let start = offsets[0]
            .as_u64()
            .and_then(|v| usize::try_from(v).ok())
            .ok_or_else(|| model_load(path, format!("tensor '{name}' has invalid start offset")))?;
        let end = offsets[1]
            .as_u64()
            .and_then(|v| usize::try_from(v).ok())
            .ok_or_else(|| model_load(path, format!("tensor '{name}' has invalid end offset")))?;
        if start > end {
            return Err(model_load(
                path,
                format!("tensor '{name}' data_offsets are reversed"),
            ));
        }
        if end as u64 > data_len {
            return Err(model_load(
                path,
                format!("tensor '{name}' extends beyond Safetensors data section"),
            ));
        }
        let byte_size = end - start;
        let expected = expected_tensor_byte_size(&dtype, &shape, path, name)?;
        if expected != byte_size {
            return Err(model_load(
                path,
                format!(
                    "tensor '{name}' byte size mismatch: shape/dtype imply {expected} bytes, data_offsets span {byte_size} bytes"
                ),
            ));
        }

        tensors.push(SafetensorsTensorRecord {
            name: name.clone(),
            dtype,
            shape: shape.clone(),
            byte_size,
            source_shard: source_shard.clone(),
            data_offsets: [start, end],
            labels: classify_tensor(name, &shape),
        });
    }

    reject_tensor_data_ranges(path, &tensors, data_len)?;

    Ok(ShardInspection { metadata, tensors })
}

pub(super) fn read_index(path: &Path) -> Result<RawIndex> {
    let len = fs::metadata(path)
        .map_err(|e| model_load(path, e.to_string()))?
        .len();
    if len > MAX_INDEX_BYTES {
        return Err(model_load(
            path,
            format!("Safetensors index is {len} bytes, exceeding limit {MAX_INDEX_BYTES}"),
        ));
    }
    let bytes = fs::read(path).map_err(|e| model_load(path, e.to_string()))?;
    let value = parse_json_rejecting_duplicate_keys(&bytes, path, "Safetensors index")?;
    serde_json::from_value(value).map_err(|e| model_load(path, format!("parse index JSON: {e}")))
}

pub(super) fn find_index_file(root: &Path) -> Result<Option<PathBuf>> {
    let mut candidates = Vec::new();
    for path in read_dir_paths(root)? {
        if is_safetensors_index(&path) && is_regular_file(&path)? {
            validate_path_stays_under_root(root, &path)?;
            candidates.push(path);
        }
    }
    candidates.sort();
    if candidates.len() > 1 {
        let names = candidates
            .iter()
            .map(|path| relative_path(path, root))
            .collect::<Vec<_>>()
            .join(", ");
        return Err(HybridError::UnsupportedFormat(format!(
            "multiple Safetensors index files found in '{}': {names}; pass the intended index file explicitly",
            root.display()
        )));
    }
    Ok(candidates.pop())
}

pub(super) fn list_safetensors_files(root: &Path) -> Result<Vec<PathBuf>> {
    let mut shards = Vec::new();
    for path in read_dir_paths(root)? {
        if path.extension().and_then(|ext| ext.to_str()) != Some(SAFETENSORS_EXTENSION) {
            continue;
        }
        if !is_regular_file(&path)? {
            continue;
        }
        validate_path_stays_under_root(root, &path)?;
        shards.push(path);
    }
    shards.sort();
    Ok(shards)
}

pub(super) fn is_regular_file(path: &Path) -> Result<bool> {
    fs::metadata(path)
        .map(|metadata| metadata.is_file())
        .map_err(|e| model_load(path, e.to_string()))
}

pub(super) fn read_dir_paths(root: &Path) -> Result<Vec<PathBuf>> {
    let mut paths = Vec::new();
    for entry in fs::read_dir(root).map_err(|e| model_load(root, e.to_string()))? {
        let entry = entry.map_err(|e| model_load(root, format!("read directory entry: {e}")))?;
        paths.push(entry.path());
    }
    Ok(paths)
}

pub(super) fn shard_metadata_namespaced_key(shard_path: &str, key: &str) -> String {
    format!("shard:{shard_path}{SHARD_METADATA_KEY_SEP}{key}")
}

pub(super) fn namespaced_shard_metadata_logical_key(namespaced: &str) -> Option<&str> {
    let rest = namespaced.strip_prefix("shard:")?;
    if let Some((_, logical_key)) = rest.split_once(SHARD_METADATA_KEY_SEP) {
        return Some(logical_key);
    }
    // Legacy keys used `shard:{path}:{key}` with a single `:` separator; this
    // remains ambiguous when `key` contains `:`, so prefer new keys above.
    rest.rsplit_once(':').map(|(_, logical_key)| logical_key)
}

pub(super) fn merge_shard_metadata(
    metadata: &mut BTreeMap<String, String>,
    shard_path: &str,
    shard_metadata: BTreeMap<String, String>,
) {
    for (key, value) in shard_metadata {
        let namespaced_key = shard_metadata_namespaced_key(shard_path, &key);
        match metadata.get(&key) {
            None => {
                if !has_shard_metadata_key(metadata, &key) {
                    metadata.insert(key.clone(), value.clone());
                }
                metadata.insert(namespaced_key, value);
            }
            Some(existing) if existing == &value => {
                metadata.insert(namespaced_key, value);
            }
            Some(_) => {
                metadata.remove(&key);
                metadata.insert(namespaced_key, value);
            }
        }
    }
}

pub(super) fn has_shard_metadata_key(metadata: &BTreeMap<String, String>, key: &str) -> bool {
    metadata
        .keys()
        .any(|existing| namespaced_shard_metadata_logical_key(existing).is_some_and(|k| k == key))
}

pub(super) fn parse_unreferenced_shards_metadata(encoded: &str) -> Vec<String> {
    if let Ok(paths) = serde_json::from_str::<Vec<String>>(encoded) {
        return paths;
    }
    encoded
        .split(',')
        .filter(|shard| !shard.is_empty())
        .map(str::to_string)
        .collect()
}
