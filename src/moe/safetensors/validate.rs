// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Safetensors byte-range and output-path validation helpers.

use super::manifest::{
    INDEX_UNREFERENCED_SHARDS_KEY, SafetensorsManifest, SafetensorsTensorRecord,
    parse_unreferenced_shards_metadata,
};
use super::model_load;
use super::paths::{canonical_existing_or_parent, parent_or_current, paths_refer_to_same_file};
use crate::error::Result;
use std::collections::BTreeSet;
use std::fs;
use std::path::Path;

pub(super) fn reject_tensor_data_ranges(
    path: &Path,
    tensors: &[SafetensorsTensorRecord],
    data_len: u64,
) -> Result<()> {
    let data_len = usize::try_from(data_len).map_err(|_| {
        model_load(
            path,
            format!("Safetensors data section length {data_len} does not fit in usize"),
        )
    })?;
    let mut ranges = tensors
        .iter()
        .map(|tensor| {
            (
                tensor.data_offsets[0],
                tensor.data_offsets[1],
                tensor.name.as_str(),
            )
        })
        .collect::<Vec<_>>();
    ranges.sort_by_key(|(start, end, name)| (*start, *end, *name));

    let mut expected_start = 0usize;
    let mut previous_name: Option<&str> = None;
    for (start, end, name) in ranges {
        if start < expected_start {
            return Err(model_load(
                path,
                format!(
                    "tensor '{name}' data range overlaps tensor '{}'",
                    previous_name.unwrap_or("<unknown>")
                ),
            ));
        }
        if start > expected_start {
            return Err(model_load(
                path,
                format!("tensor '{name}' data range leaves a gap from {expected_start} to {start}"),
            ));
        }
        expected_start = end;
        previous_name = Some(name);
    }

    if expected_start != data_len {
        return Err(model_load(
            path,
            format!(
                "Safetensors tensor data ranges end at {expected_start}, but data section is {data_len} bytes"
            ),
        ));
    }

    Ok(())
}

pub(super) fn expected_tensor_byte_size(
    dtype: &str,
    shape: &[usize],
    path: &Path,
    tensor_name: &str,
) -> Result<usize> {
    let elements = shape.iter().try_fold(1usize, |acc, dim| {
        acc.checked_mul(*dim).ok_or_else(|| {
            model_load(
                path,
                format!("tensor '{tensor_name}' shape element count overflow"),
            )
        })
    })?;

    // Int4 is a packed format: 2 elements per byte.
    // `div_ceil(2)` cannot overflow a `usize` element count.
    if is_packed_int4(dtype) {
        return Ok(elements.div_ceil(2));
    }

    let element_size = dtype_size_bytes(dtype).ok_or_else(|| {
        model_load(
            path,
            format!("tensor '{tensor_name}' has unsupported Safetensors dtype '{dtype}'"),
        )
    })?;
    elements
        .checked_mul(element_size)
        .ok_or_else(|| model_load(path, format!("tensor '{tensor_name}' byte size overflow")))
}

/// Packed Int4 dtypes: two 4-bit elements per byte.
pub(crate) fn is_packed_int4(dtype: &str) -> bool {
    matches!(dtype, "INT4" | "I4" | "U4")
}

pub(crate) fn dtype_size_bytes(dtype: &str) -> Option<usize> {
    match dtype {
        "C128" => Some(16),
        "F64" | "I64" | "U64" | "C64" => Some(8),
        "F32" | "TF32" | "I32" | "U32" => Some(4),
        "F16" | "BF16" | "I16" | "U16" => Some(2),
        "F8_E5M2" | "F8_E4M3" | "F8_E8M0" | "I8" | "U8" | "BOOL" => Some(1),
        // Int4 is a packed format (2 elements per byte); callers must handle
        // packed semantics explicitly rather than using element_size * count.
        "INT4" | "I4" | "U4" => None,
        _ => None,
    }
}

pub(super) fn reject_output_checkpoint_conflict(
    checkpoint_path: &Path,
    output_path: &Path,
    manifest: &SafetensorsManifest,
) -> Result<()> {
    let output = canonical_existing_or_parent(output_path)?;
    let input_metadata = fs::metadata(checkpoint_path)
        .map_err(|e| model_load(checkpoint_path, format!("stat checkpoint input: {e}")))?;
    let root = if input_metadata.is_dir() {
        checkpoint_path
    } else {
        parent_or_current(checkpoint_path)
    };

    let mut checkpoint_files = BTreeSet::new();
    if input_metadata.is_file() {
        checkpoint_files.insert(checkpoint_path.to_path_buf());
    }
    if let Some(index_file) = &manifest.checkpoint.index_file {
        checkpoint_files.insert(root.join(index_file));
    }
    for tensor in &manifest.tensors {
        checkpoint_files.insert(root.join(&tensor.source_shard));
    }
    if let Some(unreferenced_shards) = manifest
        .checkpoint
        .metadata
        .get(INDEX_UNREFERENCED_SHARDS_KEY)
    {
        for shard in parse_unreferenced_shards_metadata(unreferenced_shards) {
            checkpoint_files.insert(root.join(shard));
        }
    }

    for checkpoint_file in checkpoint_files {
        if paths_refer_to_same_file(&checkpoint_file, &output).map_err(|e| {
            model_load(
                &checkpoint_file,
                format!("compare checkpoint/output file identity: {e}"),
            )
        })? {
            return Err(model_load(
                output_path,
                "manifest output path must not overwrite a Safetensors checkpoint or index file"
                    .into(),
            ));
        }
    }

    Ok(())
}
