// SPDX-License-Identifier: Apache-2.0 OR MIT
use super::ggml::*;
use super::safetensors::dtype_size_bytes;
use super::safetensors::test_support::{
    TestDir, temp_dir, write_hf_config, write_safetensors_bytes,
};
use super::*;
use std::io::Write;
use std::path::{Path, PathBuf};

/// Temp GGUF fixture removed on drop, so a failing assertion does not leak it.
/// Declare it before anything that maps the file: locals drop in reverse order,
/// and Windows refuses to delete a file that is still mapped.
pub(super) struct TempGguf(PathBuf);

impl std::ops::Deref for TempGguf {
    type Target = Path;

    fn deref(&self) -> &Path {
        &self.0
    }
}

impl Drop for TempGguf {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

pub(super) fn write_temp_file(bytes: &[u8], label: &str) -> TempGguf {
    TempGguf(write_temp_path(bytes, label))
}

pub(super) fn write_temp_path(bytes: &[u8], label: &str) -> PathBuf {
    let path = std::env::temp_dir().join(format!(
        "corinth_canal_{label}_{}.gguf",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let mut file = std::fs::File::create(&path).unwrap();
    file.write_all(bytes).unwrap();
    path
}

fn push_u32(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn push_u64(out: &mut Vec<u8>, value: u64) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn push_string(out: &mut Vec<u8>, value: &str) {
    push_u64(out, value.len() as u64);
    out.extend_from_slice(value.as_bytes());
}

fn push_kv_u32(out: &mut Vec<u8>, key: &str, value: u32) {
    push_string(out, key);
    push_u32(out, GGUF_VALUE_TYPE_UINT32);
    push_u32(out, value);
}

fn push_kv_string(out: &mut Vec<u8>, key: &str, value: &str) {
    push_string(out, key);
    push_u32(out, GGUF_VALUE_TYPE_STRING);
    push_string(out, value);
}

const OLMOE_NUMERICS: [(&str, u32); 4] = [
    ("olmoe.embedding_length", EMBEDDING_DIM as u32),
    ("olmoe.block_count", 16),
    ("olmoe.expert_count", 64),
    ("olmoe.expert_used_count", 8),
];

fn build_test_gguf(tensors: Vec<(&str, Vec<usize>, u32, Vec<u8>)>, alignment: u32) -> Vec<u8> {
    build_test_gguf_with_numerics(tensors, alignment, &OLMOE_NUMERICS)
}

fn build_test_gguf_with_numerics(
    tensors: Vec<(&str, Vec<usize>, u32, Vec<u8>)>,
    alignment: u32,
    numerics: &[(&str, u32)],
) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(&GGUF_MAGIC);
    push_u32(&mut out, GGUF_VERSION);
    push_u64(&mut out, tensors.len() as u64);
    push_u64(&mut out, 3 + numerics.len() as u64);
    push_kv_u32(&mut out, "general.alignment", alignment);
    push_kv_u32(&mut out, "general.file_type", 1);
    push_kv_string(&mut out, "general.architecture", "olmoe");
    for &(key, value) in numerics {
        push_kv_u32(&mut out, key, value);
    }

    let mut data_offset = 0usize;
    let mut tensor_payloads = Vec::new();
    for (name, dims, ggml_type, payload) in tensors {
        push_string(&mut out, name);
        push_u32(&mut out, dims.len() as u32);
        for dim in &dims {
            push_u64(&mut out, *dim as u64);
        }
        push_u32(&mut out, ggml_type);
        push_u64(&mut out, data_offset as u64);
        data_offset += payload.len();
        tensor_payloads.push(payload);
    }

    while out.len() % alignment as usize != 0 {
        out.push(0);
    }
    for payload in tensor_payloads {
        out.extend_from_slice(&payload);
    }

    out
}

fn build_real_size_checkpoint(gate_payload: Vec<u8>) -> Vec<u8> {
    let attn_q_payload = vec![0u8; EMBEDDING_DIM * EMBEDDING_DIM * 2];
    build_test_gguf(
        vec![
            (
                "blk.0.ffn_gate_inp.weight",
                vec![EMBEDDING_DIM, 64],
                GGML_TYPE_F32,
                gate_payload,
            ),
            (
                "blk.0.attn_q.weight",
                vec![EMBEDDING_DIM, EMBEDDING_DIM],
                GGML_TYPE_F16,
                attn_q_payload,
            ),
            (
                "token_embd.weight",
                vec![EMBEDDING_DIM, 32],
                GGML_TYPE_F16,
                vec![0u8; EMBEDDING_DIM * 32 * 2],
            ),
        ],
        32,
    )
}

fn build_quantized_synapse_checkpoint(gate_payload: Vec<u8>) -> Vec<u8> {
    build_test_gguf(
        vec![
            (
                "blk.0.ffn_gate_inp.weight",
                vec![EMBEDDING_DIM, 64],
                GGML_TYPE_F32,
                gate_payload,
            ),
            (
                "blk.0.attn_q.weight",
                vec![EMBEDDING_DIM, EMBEDDING_DIM],
                GGML_TYPE_IQ3_S,
                Vec::new(),
            ),
            (
                "token_embd.weight",
                vec![EMBEDDING_DIM, 32],
                GGML_TYPE_F16,
                vec![0u8; EMBEDDING_DIM * 32 * 2],
            ),
        ],
        32,
    )
}

/// Build a minimal valid Q8_0 payload for a tensor of `width * n_rows`
/// elements.  Each block uses `scale_bits` as the raw F16 scale and
/// `quant_val` for every quantized byte.
fn build_q8_0_payload(width: usize, n_rows: usize, scale_bits: u16, quant_val: i8) -> Vec<u8> {
    assert!(
        width.is_multiple_of(32),
        "Q8_0 width must be divisible by 32"
    );
    let blocks_per_row = width / 32;
    let row_bytes = blocks_per_row * 34;
    let mut out = vec![0u8; row_bytes * n_rows];
    for row in 0..n_rows {
        let row_start = row * row_bytes;
        for blk in 0..blocks_per_row {
            let blk_start = row_start + blk * 34;
            let [lo, hi] = scale_bits.to_le_bytes();
            out[blk_start] = lo;
            out[blk_start + 1] = hi;
            for q in 0..32 {
                out[blk_start + 2 + q] = quant_val as u8;
            }
        }
    }
    out
}

pub(super) fn build_q8_0_synapse_checkpoint(gate_payload: Vec<u8>) -> Vec<u8> {
    // Q8_0 payload: scale = 1.0 (f16 bits = 0x3c00), quant = 1
    let attn_q_payload = build_q8_0_payload(EMBEDDING_DIM, EMBEDDING_DIM, 0x3c00, 1);
    build_test_gguf(
        vec![
            (
                "blk.0.ffn_gate_inp.weight",
                vec![EMBEDDING_DIM, 64],
                GGML_TYPE_F32,
                gate_payload,
            ),
            (
                "blk.0.attn_q.weight",
                vec![EMBEDDING_DIM, EMBEDDING_DIM],
                GGML_TYPE_Q8_0,
                attn_q_payload,
            ),
            (
                "token_embd.weight",
                vec![EMBEDDING_DIM, 32],
                GGML_TYPE_F16,
                vec![0u8; EMBEDDING_DIM * 32 * 2],
            ),
        ],
        32,
    )
}

/// Build a minimal valid Q5_K payload for a tensor of `width * n_rows`
/// elements. Q5_K block layout (176 bytes per 256 elements):
/// - d (f16, 2 bytes): scale
/// - dmin (f16, 2 bytes): min scale
/// - scales (12 bytes): 6 pairs of (scale, min) for 32-element chunks
/// - qh (32 bytes): high 2 bits for each of 256 quant values
/// - ql (128 bytes): low 4 bits for each of 256 quant values
///
/// For simplicity, this creates a payload where all quant values are 1
/// and scales are set to produce output values of 1.0.
fn build_q5_k_payload(width: usize, n_rows: usize) -> Vec<u8> {
    assert!(
        width.is_multiple_of(256),
        "Q5_K width must be divisible by 256"
    );
    let blocks_per_row = width / 256;
    let row_bytes = blocks_per_row * 176;
    let mut out = vec![0u8; row_bytes * n_rows];

    for row in 0..n_rows {
        let row_start = row * row_bytes;
        for blk in 0..blocks_per_row {
            let blk_start = row_start + blk * 176;

            // d = 1.0 (f16 bits = 0x3c00)
            out[blk_start] = 0x00;
            out[blk_start + 1] = 0x3c;

            // dmin = 0.0 (f16 bits = 0x0000)
            out[blk_start + 2] = 0x00;
            out[blk_start + 3] = 0x00;

            // scales: 6 pairs of (sc, m) for 32-element chunks
            // We want sc=1 for all chunks; dmin=0 makes m irrelevant, so output = 1.0 * 1 - 0.0 = 1.0
            // scale_min_k4 encoding: lower 6 bits for scale, upper 2 bits contribute to min
            for i in 0..12 {
                out[blk_start + 4 + i] = 0x01;
            }

            // qh: high 2 bits for each quant value (all zeros for values 0-15)
            // We want quant values to be 1, so high bits are 0
            for i in 0..32 {
                out[blk_start + 16 + i] = 0x00;
            }

            // ql: each byte packs two 4-bit quant values (low nibble + high nibble).
            // 0x11 sets both nibbles to 1, so every quant value decodes as 1.
            for i in 0..128 {
                out[blk_start + 48 + i] = 0x11;
            }
        }
    }
    out
}

fn build_q6_k_payload(width: usize, n_rows: usize) -> Vec<u8> {
    assert!(
        width.is_multiple_of(256),
        "Q6_K width must be divisible by 256"
    );
    let blocks_per_row = width / 256;
    // Q6_K block: d(2) + scales(16) + ql(128) + qh(64) = 210 bytes
    let row_bytes = blocks_per_row * 210;
    let mut out = vec![0u8; row_bytes * n_rows];

    for row in 0..n_rows {
        let row_start = row * row_bytes;
        for blk in 0..blocks_per_row {
            let blk_start = row_start + blk * 210;

            // d = 1.0 (f16 bits = 0x3c00)
            out[blk_start] = 0x00;
            out[blk_start + 1] = 0x3c;

            // scales: 16 bytes, all set to 1
            for i in 0..16 {
                out[blk_start + 2 + i] = 0x01;
            }

            // ql: each byte packs two 4-bit quant values.
            // 0x00 sets both nibbles to 0.
            for i in 0..128 {
                out[blk_start + 18 + i] = 0x00;
            }

            // qh: high 2 bits for each quant value (all zeros).
            // With ql=0x00 and qh=0x00, combined = 0, value = 0 - 32 = -32.
            // Output = d * scale * value = 1.0 * 1 * (-32) = -32.0 for every element.
            for i in 0..64 {
                out[blk_start + 146 + i] = 0x00;
            }
        }
    }
    out
}

fn build_q6_k_synapse_checkpoint(gate_payload: Vec<u8>) -> Vec<u8> {
    let attn_q_payload = build_q6_k_payload(EMBEDDING_DIM, EMBEDDING_DIM);
    build_test_gguf(
        vec![
            (
                "blk.0.ffn_gate_inp.weight",
                vec![EMBEDDING_DIM, 64],
                GGML_TYPE_F32,
                gate_payload,
            ),
            (
                "blk.0.attn_q.weight",
                vec![EMBEDDING_DIM, EMBEDDING_DIM],
                GGML_TYPE_Q6_K,
                attn_q_payload,
            ),
            (
                "token_embd.weight",
                vec![EMBEDDING_DIM, 32],
                GGML_TYPE_F16,
                vec![0u8; EMBEDDING_DIM * 32 * 2],
            ),
        ],
        32,
    )
}

fn build_q5_k_synapse_checkpoint(gate_payload: Vec<u8>) -> Vec<u8> {
    let attn_q_payload = build_q5_k_payload(EMBEDDING_DIM, EMBEDDING_DIM);
    build_test_gguf(
        vec![
            (
                "blk.0.ffn_gate_inp.weight",
                vec![EMBEDDING_DIM, 64],
                GGML_TYPE_F32,
                gate_payload,
            ),
            (
                "blk.0.attn_q.weight",
                vec![EMBEDDING_DIM, EMBEDDING_DIM],
                GGML_TYPE_Q5_K,
                attn_q_payload,
            ),
            (
                "token_embd.weight",
                vec![EMBEDDING_DIM, 32],
                GGML_TYPE_F16,
                vec![0u8; EMBEDDING_DIM * 32 * 2],
            ),
        ],
        32,
    )
}

fn stub() -> Router {
    Router::load_with_mode("", 8, 1, RoutingMode::StubUniform).expect("stub load should succeed")
}

#[test]
fn test_stub_mode_loads() {
    let model = stub();
    assert!(!model.is_loaded());
    assert_eq!(model.quantization(), "stub");
}

#[test]
fn test_stub_forward_uniform_weights() {
    let mut model = stub();
    let out = model.forward(&vec![0.1; EMBEDDING_DIM]).unwrap();
    for weight in &out.expert_weights {
        assert!((*weight - 0.125).abs() < 1e-5);
    }
}

#[test]
fn test_dense_sim_uses_real_gate_weights() {
    let mut gate = vec![0.0f32; EMBEDDING_DIM * 64];
    for (expert, value) in gate.iter_mut().take(64).enumerate() {
        *value = if expert == 0 { 8.0 } else { -8.0 };
    }
    let gate_bytes: Vec<u8> = gate.iter().flat_map(|value| value.to_le_bytes()).collect();
    let path = write_temp_file(&build_real_size_checkpoint(gate_bytes), "dense-real");

    let mut model =
        Router::load_with_mode(path.to_str().unwrap(), 8, 2, RoutingMode::DenseSim).unwrap();
    let mut embedding = vec![0.0f32; EMBEDDING_DIM];
    embedding[0] = 1.0;
    let out = model.forward(&embedding).unwrap();
    assert_eq!(out.selected_experts[0], 0);
    assert_eq!(model.family(), ModelFamily::Olmoe);
    assert_eq!(model.routing_tensor_name(), "blk.0.ffn_gate_inp.weight");
}

#[test]
fn test_quantized_synapse_probe_uses_routing_f32_fallback() {
    let gate_payload = vec![0u8; EMBEDDING_DIM * 64 * size_of::<f32>()];
    let path = write_temp_file(
        &build_quantized_synapse_checkpoint(gate_payload),
        "iq3-s-synapse",
    );

    let metadata = Router::probe_model(path.to_str().unwrap(), None).unwrap();
    assert_eq!(
        metadata.preferred_gpu_synapse_tensor_name.as_deref(),
        Some("blk.0.attn_q.weight")
    );
    assert_eq!(metadata.real_gpu_synapse_tensor_name, None);
    assert_eq!(metadata.synapse_source, "routing-f32");
}

#[test]
fn test_quantized_attn_q_does_not_advertise_real_gpu_synapse_tensor() {
    let gate_payload = vec![0u8; EMBEDDING_DIM * 64 * 4];
    let attn_q_payload = vec![0u8; 16];

    let checkpoint = build_test_gguf(
        vec![
            (
                "blk.0.ffn_gate_inp.weight",
                vec![EMBEDDING_DIM, 64],
                GGML_TYPE_F32,
                gate_payload,
            ),
            (
                "blk.0.attn_q.weight",
                vec![EMBEDDING_DIM, EMBEDDING_DIM],
                GGML_TYPE_IQ3_S,
                attn_q_payload,
            ),
            (
                "token_embd.weight",
                vec![EMBEDDING_DIM, 32],
                GGML_TYPE_F16,
                vec![0u8; EMBEDDING_DIM * 32 * 2],
            ),
        ],
        32,
    );

    let path = write_temp_file(&checkpoint, "quantized-attn-q");
    let metadata = Router::probe_model(path.to_str().unwrap(), None).unwrap();

    assert_eq!(
        metadata.preferred_gpu_synapse_tensor_name.as_deref(),
        Some("blk.0.attn_q.weight")
    );
    assert_eq!(metadata.real_gpu_synapse_tensor_name, None);
    assert_eq!(metadata.synapse_source, "routing-f32");
}

#[test]
fn test_preferred_synapse_descriptor_iq3s_uses_routing_f32_fallback() {
    let gate_payload = vec![0u8; EMBEDDING_DIM * 64 * size_of::<f32>()];
    let path = write_temp_file(
        &build_quantized_synapse_checkpoint(gate_payload),
        "iq3-s-descriptor",
    );

    let model =
        Router::load_with_mode(path.to_str().unwrap(), 0, 0, RoutingMode::StubUniform).unwrap();
    let descriptor = model
        .preferred_gpu_synapse_tensor_descriptor()
        .expect("preferred descriptor must be exposed for quantized attn_q");

    assert_eq!(descriptor.name, "blk.0.attn_q.weight");
    assert_eq!(descriptor.ggml_type_id, GGML_TYPE_IQ3_S);
    assert_eq!(descriptor.ggml_type_label, "IQ3_S");
    assert_eq!(descriptor.dims, vec![EMBEDDING_DIM, EMBEDDING_DIM]);
    assert!(!descriptor.has_dequant_path);
    assert_eq!(model.real_gpu_synapse_tensor_name(), None);
    assert_eq!(model.synapse_source(), "routing-f32");
}

#[test]
fn test_preferred_synapse_descriptor_f16_has_dequant_path() {
    let gate_payload = vec![0u8; EMBEDDING_DIM * 64 * size_of::<f32>()];
    let path = write_temp_file(&build_real_size_checkpoint(gate_payload), "f16-descriptor");

    let model =
        Router::load_with_mode(path.to_str().unwrap(), 0, 0, RoutingMode::StubUniform).unwrap();
    let descriptor = model
        .preferred_gpu_synapse_tensor_descriptor()
        .expect("preferred descriptor must be exposed for F16 attn_q");

    assert_eq!(descriptor.name, "blk.0.attn_q.weight");
    assert_eq!(descriptor.ggml_type_id, GGML_TYPE_F16);
    assert_eq!(descriptor.ggml_type_label, "F16");
    assert_eq!(descriptor.dims, vec![EMBEDDING_DIM, EMBEDDING_DIM]);
    assert!(descriptor.has_dequant_path);
    assert_eq!(
        model.real_gpu_synapse_tensor_name(),
        Some("blk.0.attn_q.weight")
    );
    assert_eq!(model.synapse_source(), "real");
}

#[test]
fn test_preferred_synapse_descriptor_q8_0_has_dequant_path() {
    let gate_payload = vec![0u8; EMBEDDING_DIM * 64 * size_of::<f32>()];
    let path = write_temp_file(
        &build_q8_0_synapse_checkpoint(gate_payload),
        "q8-0-descriptor",
    );

    let model =
        Router::load_with_mode(path.to_str().unwrap(), 0, 0, RoutingMode::StubUniform).unwrap();
    let descriptor = model
        .preferred_gpu_synapse_tensor_descriptor()
        .expect("preferred descriptor must be exposed for Q8_0 attn_q");

    assert_eq!(descriptor.name, "blk.0.attn_q.weight");
    assert_eq!(descriptor.ggml_type_id, GGML_TYPE_Q8_0);
    assert_eq!(descriptor.ggml_type_label, "Q8_0");
    assert_eq!(descriptor.dims, vec![EMBEDDING_DIM, EMBEDDING_DIM]);
    assert!(descriptor.has_dequant_path);
    assert_eq!(model.real_gpu_synapse_tensor_name(), None);
    assert_eq!(
        model.dequantized_q8_0_synapse_tensor_name(),
        Some("blk.0.attn_q.weight")
    );
    assert_eq!(model.synapse_source(), "dequantized-q8_0");
}

#[test]
fn test_q8_0_synapse_probe_uses_dequantized_source() {
    let gate_payload = vec![0u8; EMBEDDING_DIM * 64 * size_of::<f32>()];
    let path = write_temp_file(&build_q8_0_synapse_checkpoint(gate_payload), "q8-0-probe");

    let metadata = Router::probe_model(path.to_str().unwrap(), None).unwrap();
    assert_eq!(
        metadata.preferred_gpu_synapse_tensor_name.as_deref(),
        Some("blk.0.attn_q.weight")
    );
    assert_eq!(metadata.real_gpu_synapse_tensor_name, None);
    assert_eq!(metadata.synapse_source, "dequantized-q8_0");
}

#[test]
fn test_synapse_tensor_row_major_shape_reports_q8_0_dims() {
    let gate_payload = vec![0u8; EMBEDDING_DIM * 64 * size_of::<f32>()];
    let path = write_temp_file(&build_q8_0_synapse_checkpoint(gate_payload), "q8-0-shape");

    let model =
        Router::load_with_mode(path.to_str().unwrap(), 0, 0, RoutingMode::StubUniform).unwrap();

    let shape = model
        .synapse_tensor_row_major_shape("blk.0.attn_q.weight")
        .expect("Q8_0 synapse tensor shape must be readable");
    assert_eq!(shape, (EMBEDDING_DIM, EMBEDDING_DIM));
}

#[test]
fn test_synapse_tensor_row_major_shape_uses_one_row_for_rank_one_tensor() {
    let gate_payload = vec![0u8; EMBEDDING_DIM * 64 * size_of::<f32>()];
    let checkpoint = build_test_gguf(
        vec![
            (
                "blk.0.ffn_gate_inp.weight",
                vec![EMBEDDING_DIM, 64],
                GGML_TYPE_F32,
                gate_payload,
            ),
            (
                "blk.0.attn_q.weight",
                vec![EMBEDDING_DIM, EMBEDDING_DIM],
                GGML_TYPE_F16,
                vec![0u8; EMBEDDING_DIM * EMBEDDING_DIM * 2],
            ),
            (
                "token_embd.weight",
                vec![EMBEDDING_DIM, 32],
                GGML_TYPE_F16,
                vec![0u8; EMBEDDING_DIM * 32 * 2],
            ),
            ("rank1.tensor", vec![7], GGML_TYPE_F16, vec![0u8; 7 * 2]),
        ],
        32,
    );
    let path = write_temp_file(&checkpoint, "rank-1-shape");

    let model =
        Router::load_with_mode(path.to_str().unwrap(), 0, 0, RoutingMode::StubUniform).unwrap();

    let shape = model
        .synapse_tensor_row_major_shape("rank1.tensor")
        .expect("rank-1 tensor shape must be readable");
    assert_eq!(shape, (1, 7));
}

#[test]
fn test_synapse_tensor_row_major_shape_rejects_zero_dim_tensor() {
    let gate_payload = vec![0u8; EMBEDDING_DIM * 64 * size_of::<f32>()];
    let checkpoint = build_test_gguf(
        vec![
            (
                "blk.0.ffn_gate_inp.weight",
                vec![EMBEDDING_DIM, 64],
                GGML_TYPE_F32,
                gate_payload,
            ),
            (
                "blk.0.attn_q.weight",
                vec![EMBEDDING_DIM, EMBEDDING_DIM],
                GGML_TYPE_F16,
                vec![0u8; EMBEDDING_DIM * EMBEDDING_DIM * 2],
            ),
            (
                "token_embd.weight",
                vec![EMBEDDING_DIM, 32],
                GGML_TYPE_F16,
                vec![0u8; EMBEDDING_DIM * 32 * 2],
            ),
            ("zero-dim.tensor", vec![], GGML_TYPE_F16, Vec::new()),
        ],
        32,
    );
    let path = write_temp_file(&checkpoint, "zero-dim-shape");

    let model =
        Router::load_with_mode(path.to_str().unwrap(), 0, 0, RoutingMode::StubUniform).unwrap();

    let err = model
        .synapse_tensor_row_major_shape("zero-dim.tensor")
        .expect_err("zero-dim tensor must be rejected");
    assert!(matches!(err, HybridError::UnsupportedFormat(_)));
}

#[test]
fn test_preferred_synapse_descriptor_q5_k_has_dequant_path() {
    let gate_payload = vec![0u8; EMBEDDING_DIM * 64 * size_of::<f32>()];
    let path = write_temp_file(
        &build_q5_k_synapse_checkpoint(gate_payload),
        "q5-k-descriptor",
    );

    let model =
        Router::load_with_mode(path.to_str().unwrap(), 0, 0, RoutingMode::StubUniform).unwrap();
    let descriptor = model
        .preferred_gpu_synapse_tensor_descriptor()
        .expect("preferred descriptor must be exposed for Q5_K attn_q");

    assert_eq!(descriptor.name, "blk.0.attn_q.weight");
    assert_eq!(descriptor.ggml_type_id, GGML_TYPE_Q5_K);
    assert_eq!(descriptor.ggml_type_label, "Q5_K");
    assert_eq!(descriptor.dims, vec![EMBEDDING_DIM, EMBEDDING_DIM]);
    assert!(descriptor.has_dequant_path);
    assert_eq!(model.real_gpu_synapse_tensor_name(), None);
    assert_eq!(
        model.dequantized_q5_k_synapse_tensor_name(),
        Some("blk.0.attn_q.weight")
    );
    assert_eq!(model.synapse_source(), "dequantized-q5_k");
}

#[test]
fn test_q5_k_synapse_probe_uses_dequantized_source() {
    let gate_payload = vec![0u8; EMBEDDING_DIM * 64 * size_of::<f32>()];
    let path = write_temp_file(&build_q5_k_synapse_checkpoint(gate_payload), "q5-k-probe");

    let metadata = Router::probe_model(path.to_str().unwrap(), None).unwrap();
    assert_eq!(
        metadata.preferred_gpu_synapse_tensor_name.as_deref(),
        Some("blk.0.attn_q.weight")
    );
    assert_eq!(metadata.real_gpu_synapse_tensor_name, None);
    assert_eq!(metadata.synapse_source, "dequantized-q5_k");
}

#[test]
fn test_q6_k_synapse_probe_uses_dequantized_source() {
    let gate_payload = vec![0u8; EMBEDDING_DIM * 64 * size_of::<f32>()];
    let path = write_temp_file(&build_q6_k_synapse_checkpoint(gate_payload), "q6-k-probe");

    let metadata = Router::probe_model(path.to_str().unwrap(), None).unwrap();
    assert_eq!(
        metadata.preferred_gpu_synapse_tensor_name.as_deref(),
        Some("blk.0.attn_q.weight")
    );
    assert_eq!(metadata.real_gpu_synapse_tensor_name, None);
    assert_eq!(metadata.synapse_source, "dequantized-q6_k");
}

#[test]
fn test_q5_k_dequantize_full_tensor_succeeds() {
    let gate_payload = vec![0u8; EMBEDDING_DIM * 64 * size_of::<f32>()];
    let path = write_temp_file(&build_q5_k_synapse_checkpoint(gate_payload), "q5-k-dequant");

    let model =
        Router::load_with_mode(path.to_str().unwrap(), 0, 0, RoutingMode::StubUniform).unwrap();

    let weights = model
        .dequantized_q5_k_synapse_weights("blk.0.attn_q.weight")
        .expect("Q5_K dequantization must succeed");

    // Verify we get the expected number of elements
    assert_eq!(weights.len(), EMBEDDING_DIM * EMBEDDING_DIM);

    // Broad sanity check: every produced value is finite.
    for &v in &weights {
        assert!(v.is_finite(), "expected finite value, got {v}");
    }

    // build_q5_k_payload sets d=1.0, dmin=0.0, every quant nibble to 1 and all
    // 12 scale bytes to 0x01. scale_min_k4 then yields sc=1 for all 8 sub-blocks
    // (m is 1 or 0, but dmin=0 cancels it), so every element is
    // 1.0 * 1 * 1 - 0.0 = 1.0.
    assert_eq!(weights[0], 1.0_f32, "element 0 should be 1.0");
    assert_eq!(weights[127], 1.0_f32, "element 127 should be 1.0");
    assert_eq!(weights[128], 1.0_f32, "element 128 should be 1.0");
    assert_eq!(weights[255], 1.0_f32, "element 255 should be 1.0");
}

#[test]
fn test_q6_k_dequantize_full_tensor_succeeds() {
    let gate_payload = vec![0u8; EMBEDDING_DIM * 64 * size_of::<f32>()];
    let path = write_temp_file(&build_q6_k_synapse_checkpoint(gate_payload), "q6-k-dequant");

    let model =
        Router::load_with_mode(path.to_str().unwrap(), 0, 0, RoutingMode::StubUniform).unwrap();

    let weights = model
        .dequantized_q6_k_synapse_weights("blk.0.attn_q.weight")
        .expect("Q6_K dequantization must succeed");

    // Verify we get the expected number of elements
    assert_eq!(weights.len(), EMBEDDING_DIM * EMBEDDING_DIM);

    // Verify all values are finite
    for &v in &weights {
        assert!(v.is_finite(), "expected finite value, got {v}");
    }

    // Verify deterministic values from the known payload:
    // - d=1.0, scales=1, ql=0x00, qh=0x00
    // - combined = 0, value = 0 - 32 = -32
    // - output = 1.0 * 1 * (-32) = -32.0 for every element
    assert!(
        (weights[0] - (-32.0_f32)).abs() < 1e-4,
        "element 0 should be -32.0, got {}",
        weights[0]
    );
    assert!(
        (weights[255] - (-32.0_f32)).abs() < 1e-4,
        "element 255 should be -32.0, got {}",
        weights[255]
    );
}

#[test]
fn test_synapse_tensor_row_major_shape_requires_checkpoint() {
    let model = stub();
    let err = model
        .synapse_tensor_row_major_shape("blk.0.attn_q.weight")
        .expect_err("stub router must not expose checkpoint-backed tensor shapes");
    assert!(matches!(err, HybridError::ModelLoad { .. }));
}

#[test]
fn test_spiking_sim_state_can_reset() {
    let mut model = Router::load_with_mode("", 8, 2, RoutingMode::SpikingSim).unwrap();
    let _ = model.forward(&vec![1.0; EMBEDDING_DIM]).unwrap();
    assert!(model.has_state_activity());
    model.reset_state();
    assert!(!model.has_state_activity());
}

#[test]
fn test_dtype_size_bytes_rejects_packed_int4() {
    // dtype_size_bytes returns None for packed Int4 formats
    // (callers must handle packed semantics explicitly)
    assert_eq!(dtype_size_bytes("INT4"), None);
    assert_eq!(dtype_size_bytes("I4"), None);
    assert_eq!(dtype_size_bytes("U4"), None);
}

#[test]
fn test_synapse_source_label_new_variants() {
    use super::adapter::SynapseSource;
    let adapter = super::adapter::ModelAdapter {
        family: ModelFamily::Qwen3Moe,
        architecture: "test".into(),
        hidden_size: 128,
        num_layers: 1,
        num_experts: 8,
        expert_used_count: 2,
        token_embedding_tensor: "test".into(),
        routing_tensor: "test".into(),
        preferred_gpu_synapse_tensor: None,
        synapse_tensor: Some("blk.0.attn_q.weight".into()),
        synapse_source: SynapseSource::DequantizedIQ3M,
        quantization: "IQ3_M".into(),
    };
    assert_eq!(adapter.synapse_source_label(), "dequantized-iq3_m");

    let mut adapter2 = adapter.clone();
    adapter2.synapse_source = SynapseSource::DequantizedInt4;
    adapter2.synapse_tensor = Some("model.layers.0.mlp.gate.weight".into());
    adapter2.quantization = "INT4".into();
    assert_eq!(adapter2.synapse_source_label(), "dequantized-int4");
}

#[test]
fn test_iq3_m_full_tensor_dequantization() {
    // Build a minimal IQ3_M payload for a 256x1 tensor
    let mut block = vec![0u8; 111];
    // d = 1.0 (f16 = 0x3C00)
    block[0] = 0x00;
    block[1] = 0x3C;
    // hmask: all zeros
    // qs: all zeros
    // scales: all zeros
    // scales_h: 0

    let payload = block;
    let checkpoint = build_test_gguf(
        vec![
            (
                "blk.0.ffn_gate_inp.weight",
                vec![EMBEDDING_DIM, 64],
                GGML_TYPE_F32,
                vec![0u8; EMBEDDING_DIM * 64 * 4],
            ),
            (
                "blk.0.attn_q.weight",
                vec![256, 1],
                GGML_TYPE_IQ3_M_BLOCK,
                payload,
            ),
            (
                "token_embd.weight",
                vec![EMBEDDING_DIM, 32],
                GGML_TYPE_F16,
                vec![0u8; EMBEDDING_DIM * 32 * 2],
            ),
        ],
        32,
    );

    let path = write_temp_file(&checkpoint, "iq3_m_full");
    let mapped = probe_and_map_checkpoint(path.to_str().unwrap()).unwrap();

    // Test dequantize_iq3_m_tensor
    let result = mapped.dequantize_iq3_m_tensor("blk.0.attn_q.weight", path.to_str().unwrap());
    assert!(result.is_ok());
    let dequantized = result.unwrap();
    assert_eq!(dequantized.len(), 256);
    // With d=1.0 and all scales=0, all values should be 0
    for &val in &dequantized {
        assert_eq!(val, 0.0, "expected 0.0 with zero scales, got {val}");
    }
}

#[test]
fn test_iq3_m_tensor_error_paths() {
    // Test error handling for wrong ggml_type
    let q8_0_payload = build_q8_0_payload(256, 1, 0x3c00, 1);
    let checkpoint = build_test_gguf(
        vec![
            (
                "blk.0.ffn_gate_inp.weight",
                vec![EMBEDDING_DIM, 64],
                GGML_TYPE_F32,
                vec![0u8; EMBEDDING_DIM * 64 * 4],
            ),
            (
                "blk.0.attn_q.weight",
                vec![256, 1],
                GGML_TYPE_Q8_0,
                q8_0_payload,
            ),
            (
                "token_embd.weight",
                vec![EMBEDDING_DIM, 32],
                GGML_TYPE_F16,
                vec![0u8; EMBEDDING_DIM * 32 * 2],
            ),
        ],
        32,
    );

    let path = write_temp_file(&checkpoint, "iq3_m_err");
    let mapped = probe_and_map_checkpoint(path.to_str().unwrap()).unwrap();

    // Should fail because tensor is Q8_0, not IQ3_M
    let result = mapped.dequantize_iq3_m_tensor("blk.0.attn_q.weight", path.to_str().unwrap());
    assert!(result.is_err());
}

#[test]
fn test_safetensors_int4_tensor_extraction() {
    use super::safetensors::MappedSafetensorsCheckpoint;

    // Build a minimal safetensors file with INT4 data
    // Format: 8-byte header len + JSON header + data
    let tensor_name = "test.int4";
    let shape = vec![2usize, 4]; // 2 rows, 4 elements = 8 elements total = 4 bytes
    let data = vec![0x12u8, 0x34u8, 0x56u8, 0x78u8]; // 4 bytes

    let header = serde_json::json!({
        tensor_name: {
            "dtype": "INT4",
            "shape": shape,
            "data_offsets": [0, data.len()]
        },
        "__metadata__": {}
    });
    let tmp_dir = temp_dir("st_test");
    let st_path = tmp_dir.join("model.safetensors");
    write_safetensors_bytes(&st_path, &header, &data);

    // Write config.json
    let config = serde_json::json!({
        "architectures": ["TestModel"],
        "hidden_size": 128,
        "num_hidden_layers": 2,
        "vocab_size": 1000
    });
    write_hf_config(&tmp_dir, &config);

    // Load and extract
    let checkpoint =
        MappedSafetensorsCheckpoint::from_directory(tmp_dir.to_str().unwrap()).unwrap();
    let extracted = checkpoint
        .extract_tensor_f32(tensor_name, st_path.to_str().unwrap())
        .unwrap();

    // INT4 unpacking: 2 elements per byte
    // 0x12 -> low=2, high=1
    // 0x34 -> low=4, high=3
    // 0x56 -> low=6, high=5
    // 0x78 -> low=8, high=7
    assert_eq!(extracted, vec![2.0, 1.0, 4.0, 3.0, 6.0, 5.0, 8.0, 7.0]);
}

/// HF/Safetensors shape is (rows, cols); GGUF synapse helper must not swap ST dims.
#[test]
fn test_safetensors_synapse_shape_uses_hf_row_major() {
    use super::safetensors::MappedSafetensorsCheckpoint;

    let gate = "model.layers.0.mlp.gate.weight";
    // Rectangular: 8 experts × 16 hidden (not square) — wrong convention would swap to (16, 8).
    let rows = 8usize;
    let cols = 16usize;
    let data = vec![0u8; rows * cols * 4];
    let header = serde_json::json!({
        gate: {
            "dtype": "F32",
            "shape": [rows, cols],
            "data_offsets": [0, data.len()]
        },
        "model.embed_tokens.weight": {
            "dtype": "F32",
            "shape": [32, cols],
            "data_offsets": [data.len(), data.len() + 32 * cols * 4]
        },
        "__metadata__": {}
    });
    let mut payload = data;
    payload.resize(payload.len() + 32 * cols * 4, 0);

    let tmp_dir = temp_dir("st_shape");
    write_safetensors_bytes(&tmp_dir.join("model.safetensors"), &header, &payload);
    let config = serde_json::json!({
        "architectures": ["OlmoeForCausalLM"],
        "hidden_size": cols,
        "num_hidden_layers": 2,
        "num_experts": rows,
        "num_experts_per_tok": 2,
        "vocab_size": 32
    });
    write_hf_config(&tmp_dir, &config);

    let path = tmp_dir.to_str().unwrap();
    // Shape helper is on Router after successful ST load.
    let model = super::Router::load(path, 0, 0).expect("Olmoe ST fixture must load");
    let shape = model
        .synapse_tensor_row_major_shape(gate)
        .expect("ST gate shape must be readable");
    assert_eq!(
        shape,
        (rows, cols),
        "Safetensors must use HF (rows, cols), not GGUF (cols, rows)"
    );
    // Sanity: checkpoint still reports raw HF shape.
    let cp = MappedSafetensorsCheckpoint::from_directory(path).unwrap();
    assert_eq!(cp.tensor_info(gate).unwrap().1, &[rows, cols]);
}

#[test]
fn test_safetensors_forward_routes_through_gate_tensor() {
    let gate = "model.layers.0.mlp.gate.weight";
    let rows = 8usize;
    let cols = 16usize;
    // Expert 3 is the only positive row. The winner must not be index 0, which a
    // tied or all-zero gate could also select.
    let mut payload = Vec::new();
    for row in 0..rows {
        let value: f32 = if row == 3 { 8.0 } else { -8.0 };
        for _ in 0..cols {
            payload.extend_from_slice(&value.to_le_bytes());
        }
    }
    let gate_len = payload.len();
    let embed_len = 32 * cols * 4;
    payload.resize(gate_len + embed_len, 0);

    let header = serde_json::json!({
        gate: {
            "dtype": "F32",
            "shape": [rows, cols],
            "data_offsets": [0, gate_len]
        },
        "model.embed_tokens.weight": {
            "dtype": "F32",
            "shape": [32, cols],
            "data_offsets": [gate_len, gate_len + embed_len]
        },
        "__metadata__": {}
    });
    let tmp_dir = temp_dir("st_forward");
    write_safetensors_bytes(&tmp_dir.join("model.safetensors"), &header, &payload);
    let config = serde_json::json!({
        "architectures": ["OlmoeForCausalLM"],
        "hidden_size": cols,
        "num_hidden_layers": 2,
        "num_experts": rows,
        "num_experts_per_tok": 2,
        "vocab_size": 32
    });
    write_hf_config(&tmp_dir, &config);

    let mut model =
        Router::load_with_mode(tmp_dir.to_str().unwrap(), 0, 0, RoutingMode::DenseSim).unwrap();
    // A zero embedding would tie every score.
    let out = model.forward(&vec![1.0; EMBEDDING_DIM]).unwrap();

    assert_eq!(out.selected_experts[0], 3);
    assert_eq!(out.selected_experts.len(), 2);
    assert_eq!(out.expert_weights.len(), 8);
    assert!(out.expert_weights[3] > 0.5, "{:?}", out.expert_weights);
    assert_eq!(
        model.extract_token_embedding(0).unwrap().len(),
        EMBEDDING_DIM
    );
}

/// Olmoe Safetensors checkpoint whose gate has `gate_shape`; row 3 is the only
/// positive row, so a gate read in `[rows, hidden_size]` order selects expert 3.
fn write_safetensors_gate_checkpoint(
    label: &str,
    gate_shape: [usize; 2],
    hidden_size: usize,
    num_experts: usize,
) -> TestDir {
    let [rows, cols] = gate_shape;
    let mut payload = Vec::new();
    for row in 0..rows {
        let value: f32 = if row == 3 { 8.0 } else { -8.0 };
        for _ in 0..cols {
            payload.extend_from_slice(&value.to_le_bytes());
        }
    }
    let gate_len = payload.len();
    let embed_len = 32 * hidden_size * 4;
    payload.resize(gate_len + embed_len, 0);

    let header = serde_json::json!({
        "model.layers.0.mlp.gate.weight": {
            "dtype": "F32",
            "shape": gate_shape,
            "data_offsets": [0, gate_len]
        },
        "model.embed_tokens.weight": {
            "dtype": "F32",
            "shape": [32, hidden_size],
            "data_offsets": [gate_len, gate_len + embed_len]
        },
        "__metadata__": {}
    });
    let tmp_dir = temp_dir(label);
    write_safetensors_bytes(&tmp_dir.join("model.safetensors"), &header, &payload);
    let config = serde_json::json!({
        "architectures": ["OlmoeForCausalLM"],
        "hidden_size": hidden_size,
        "num_hidden_layers": 2,
        "num_experts": num_experts,
        "num_experts_per_tok": 2,
        "vocab_size": 32
    });
    write_hf_config(&tmp_dir, &config);
    tmp_dir
}

#[test]
fn test_safetensors_forward_accepts_a_requested_subset_of_gate_experts() {
    // The checkpoint has 64 experts and 8 are requested. GGUF has always
    // allowed this; Safetensors used to load it and then fail on forward.
    let tmp_dir = write_safetensors_gate_checkpoint("st_gate_subset", [64, 16], 16, 64);

    let mut model =
        Router::load_with_mode(tmp_dir.to_str().unwrap(), 8, 2, RoutingMode::DenseSim).unwrap();
    let out = model.forward(&vec![1.0; EMBEDDING_DIM]).unwrap();

    assert_eq!(out.expert_weights.len(), 8);
    assert_eq!(out.selected_experts[0], 3);
}

#[test]
fn test_safetensors_load_rejects_gate_without_hidden_size_dimension() {
    // min(32, 32) >= 8, but gate scoring cannot index this shape.
    let tmp_dir = write_safetensors_gate_checkpoint("st_gate_orientation", [32, 32], 16, 8);

    let result = Router::load_with_mode(tmp_dir.to_str().unwrap(), 8, 2, RoutingMode::DenseSim);

    assert!(matches!(
        result,
        Err(HybridError::UnsupportedFormat(msg)) if msg.contains("unsupported orientation")
    ));
}

#[test]
fn test_safetensors_load_rejects_gate_with_too_few_experts() {
    let tmp_dir = write_safetensors_gate_checkpoint("st_gate_few_experts", [4, 16], 16, 8);

    let result = Router::load_with_mode(tmp_dir.to_str().unwrap(), 8, 2, RoutingMode::DenseSim);

    assert!(matches!(
        result,
        Err(HybridError::UnsupportedFormat(msg)) if msg.contains("only exposes 4 experts")
    ));
}

#[test]
fn load_with_family_and_mode_applies_compatible_override_and_rejects_incompatible() {
    let path = write_temp_file(
        &build_quantized_synapse_checkpoint(vec![0u8; EMBEDDING_DIM * 64 * size_of::<f32>()]),
        "family-override",
    );
    let path_str = path.to_str().unwrap();

    let model = Router::load_with_family_and_mode(
        path_str,
        0,
        0,
        Some(ModelFamily::Olmoe),
        RoutingMode::StubUniform,
    )
    .unwrap();
    assert_eq!(model.family(), ModelFamily::Olmoe);
    assert_eq!(model.model_path(), path_str);

    assert!(
        Router::load_with_family_and_mode(
            path_str,
            0,
            0,
            Some(ModelFamily::DeepSeek2),
            RoutingMode::StubUniform,
        )
        .is_err()
    );

    let stub = Router::load_with_family_and_mode(
        "",
        4,
        2,
        Some(ModelFamily::DeepSeek2),
        RoutingMode::StubUniform,
    )
    .unwrap();
    assert_eq!(stub.family(), ModelFamily::DeepSeek2);
}

#[test]
fn test_safetensors_i4_signed_tensor_extraction() {
    use super::safetensors::MappedSafetensorsCheckpoint;

    // Build a minimal safetensors file with I4 (signed) data
    let tensor_name = "test.i4";
    let shape = vec![1usize, 2]; // 2 elements = 1 byte
    let data = vec![0x8Fu8]; // low=15 (-1), high=8 (-8) in signed I4

    let header = serde_json::json!({
        tensor_name: {
            "dtype": "I4",
            "shape": shape,
            "data_offsets": [0, data.len()]
        },
        "__metadata__": {}
    });
    let tmp_dir = temp_dir("st_i4_test");
    let st_path = tmp_dir.join("model.safetensors");
    write_safetensors_bytes(&st_path, &header, &data);

    let config = serde_json::json!({
        "architectures": ["TestModel"],
        "hidden_size": 128,
        "num_hidden_layers": 2,
        "vocab_size": 1000
    });
    write_hf_config(&tmp_dir, &config);

    let checkpoint =
        MappedSafetensorsCheckpoint::from_directory(tmp_dir.to_str().unwrap()).unwrap();
    let extracted = checkpoint
        .extract_tensor_f32(tensor_name, st_path.to_str().unwrap())
        .unwrap();

    // Signed I4: 0x8F -> low=15 -> sign-extend -> -1, high=8 -> sign-extend -> -8
    assert_eq!(extracted, vec![-1.0, -8.0]);
}

#[test]
fn test_safetensors_int4_token_embedding() {
    use super::safetensors::MappedSafetensorsCheckpoint;

    // Build a minimal safetensors file with INT4 token embeddings
    let tensor_name = "token_embd.weight";
    let shape = vec![3usize, 4]; // 3 tokens, 4 dims = 12 elements = 6 bytes
    let data = vec![0x12u8, 0x34u8, 0x56u8, 0x78u8, 0x9Au8, 0xBCu8];

    let header = serde_json::json!({
        tensor_name: {
            "dtype": "INT4",
            "shape": shape,
            "data_offsets": [0, data.len()]
        },
        "__metadata__": {}
    });
    let tmp_dir = temp_dir("st_emb_test");
    let st_path = tmp_dir.join("model.safetensors");
    write_safetensors_bytes(&st_path, &header, &data);

    let config = serde_json::json!({
        "architectures": ["TestModel"],
        "hidden_size": 128,
        "num_hidden_layers": 2,
        "vocab_size": 1000
    });
    write_hf_config(&tmp_dir, &config);

    let checkpoint =
        MappedSafetensorsCheckpoint::from_directory(tmp_dir.to_str().unwrap()).unwrap();

    // Extract token 0: first 4 elements = 2 bytes = [0x12, 0x34] -> [2, 1, 4, 3]
    let emb0 = checkpoint
        .extract_token_embedding(tensor_name, st_path.to_str().unwrap(), 0)
        .unwrap();
    assert_eq!(emb0, vec![2.0, 1.0, 4.0, 3.0]);

    // Extract token 1: next 4 elements = 2 bytes = [0x56, 0x78] -> [6, 5, 8, 7]
    let emb1 = checkpoint
        .extract_token_embedding(tensor_name, st_path.to_str().unwrap(), 1)
        .unwrap();
    assert_eq!(emb1, vec![6.0, 5.0, 8.0, 7.0]);

    // Extract token 2: last 4 elements = 2 bytes = [0x9A, 0xBC] -> [10, 9, 12, 11]
    let emb2 = checkpoint
        .extract_token_embedding(tensor_name, st_path.to_str().unwrap(), 2)
        .unwrap();
    assert_eq!(emb2, vec![10.0, 9.0, 12.0, 11.0]);
}

#[test]
fn test_router_crate_private_dequant_helpers_reachable() {
    // Keep pub(crate) dequant helpers live under --no-default-features (no CUDA
    // temporal caller) without expanding the public Router surface (Codex).
    let mut block = vec![0u8; 111];
    block[0] = 0x00;
    block[1] = 0x3C;
    let iq3_payload = block;
    let checkpoint = build_test_gguf(
        vec![
            (
                "blk.0.ffn_gate_inp.weight",
                vec![EMBEDDING_DIM, 64],
                GGML_TYPE_F32,
                vec![0u8; EMBEDDING_DIM * 64 * 4],
            ),
            (
                "blk.0.attn_q.weight",
                vec![256, 1],
                GGML_TYPE_IQ3_M_BLOCK,
                iq3_payload,
            ),
            (
                "token_embd.weight",
                vec![EMBEDDING_DIM, 32],
                GGML_TYPE_F16,
                vec![0u8; EMBEDDING_DIM * 32 * 2],
            ),
        ],
        32,
    );
    let path = write_temp_file(&checkpoint, "iq3_m_router_helpers");
    let model = super::Router::load(path.to_str().unwrap(), 0, 0).unwrap();
    assert_eq!(
        model.dequantized_iq3_m_synapse_tensor_name(),
        Some("blk.0.attn_q.weight")
    );
    let weights = model
        .dequantized_iq3_m_synapse_weights("blk.0.attn_q.weight")
        .unwrap();
    assert_eq!(weights.len(), 256);
    // Inactive paths return None / error cleanly.
    assert!(model.dequantized_q6_k_synapse_tensor_name().is_none());
    assert!(model.dequantized_int4_synapse_tensor_name().is_none());
    assert!(model.dequantized_int4_synapse_weights("nope").is_err());
    assert!(model.routing_f32_synapse_tensor_name().is_none());
    assert!(model.routing_f32_synapse_weights("nope").is_err());
    let _ = model.dequantized_q8_0_synapse_weights("blk.0.attn_q.weight");
}

#[test]
fn test_adapter_resolve_iq3_m_block_gguf() {
    // Internal IQ3_M_BLOCK id (synthetic fixtures) still selects DequantizedIQ3M.
    let mut block = vec![0u8; 111];
    block[0] = 0x00;
    block[1] = 0x3C;
    let payload = block;

    let checkpoint = build_test_gguf(
        vec![
            (
                "blk.0.ffn_gate_inp.weight",
                vec![EMBEDDING_DIM, 64],
                GGML_TYPE_F32,
                vec![0u8; EMBEDDING_DIM * 64 * 4],
            ),
            (
                "blk.0.attn_q.weight",
                vec![256, 1],
                GGML_TYPE_IQ3_M_BLOCK,
                payload,
            ),
            (
                "token_embd.weight",
                vec![EMBEDDING_DIM, 32],
                GGML_TYPE_F16,
                vec![0u8; EMBEDDING_DIM * 32 * 2],
            ),
        ],
        32,
    );

    let path = write_temp_file(&checkpoint, "iq3_m_adapter");
    let mapped = probe_and_map_checkpoint(path.to_str().unwrap()).unwrap();

    let adapter =
        super::adapter::resolve_adapter(mapped.metadata(), &mapped, None, path.to_str().unwrap())
            .unwrap();
    assert_eq!(
        adapter.synapse_source,
        super::adapter::SynapseSource::DequantizedIQ3M
    );
    assert_eq!(
        adapter.synapse_tensor.as_deref(),
        Some("blk.0.attn_q.weight")
    );
}

#[test]
fn test_adapter_resolve_tok_embeddings_fallback() {
    // token_embd.weight is absent; tok_embeddings.weight should be used instead.
    let gate_payload = vec![0u8; EMBEDDING_DIM * 64 * size_of::<f32>()];
    let attn_q_payload = vec![0u8; 16];

    let checkpoint = build_test_gguf(
        vec![
            (
                "blk.0.ffn_gate_inp.weight",
                vec![EMBEDDING_DIM, 64],
                GGML_TYPE_F32,
                gate_payload,
            ),
            (
                "blk.0.attn_q.weight",
                vec![EMBEDDING_DIM, EMBEDDING_DIM],
                GGML_TYPE_IQ3_S,
                attn_q_payload,
            ),
            (
                "tok_embeddings.weight",
                vec![EMBEDDING_DIM, 32],
                GGML_TYPE_F16,
                vec![0u8; EMBEDDING_DIM * 32 * 2],
            ),
        ],
        32,
    );

    let path = write_temp_file(&checkpoint, "tok-embeddings-fallback");
    let mapped = probe_and_map_checkpoint(path.to_str().unwrap()).unwrap();
    let adapter =
        super::adapter::resolve_adapter(mapped.metadata(), &mapped, None, path.to_str().unwrap())
            .unwrap();

    assert_eq!(adapter.token_embedding_tensor, "tok_embeddings.weight");
    // blk.0.attn_q.weight is IQ3_S (wire type 21), not the internal
    // GGML_TYPE_IQ3_M_BLOCK sentinel, so it must NOT be claimed by the IQ3_M
    // block-dequant path — doing so would decode IQ3_S bytes with the wrong
    // block layout. Selection falls through to the RoutingF32 fallback.
    assert_eq!(
        adapter.synapse_source,
        super::adapter::SynapseSource::RoutingF32
    );
}

#[test]
fn test_adapter_resolve_tok_embeddings_unsupported_type() {
    // token_embd.weight is absent and tok_embeddings.weight has an unsupported
    // quantization, so adapter resolution should fail early.
    let gate_payload = vec![0u8; EMBEDDING_DIM * 64 * size_of::<f32>()];
    let attn_q_payload = vec![0u8; 16];

    let checkpoint = build_test_gguf(
        vec![
            (
                "blk.0.ffn_gate_inp.weight",
                vec![EMBEDDING_DIM, 64],
                GGML_TYPE_F32,
                gate_payload,
            ),
            (
                "blk.0.attn_q.weight",
                vec![EMBEDDING_DIM, EMBEDDING_DIM],
                GGML_TYPE_IQ3_S,
                attn_q_payload,
            ),
            (
                "tok_embeddings.weight",
                vec![EMBEDDING_DIM, 32],
                GGML_TYPE_IQ3_S,
                vec![0u8; 16],
            ),
        ],
        32,
    );

    let path = write_temp_file(&checkpoint, "tok-embeddings-unsupported");
    let mapped = probe_and_map_checkpoint(path.to_str().unwrap()).unwrap();
    let result =
        super::adapter::resolve_adapter(mapped.metadata(), &mapped, None, path.to_str().unwrap());

    assert!(matches!(
        result,
        Err(HybridError::UnsupportedFormat(msg)) if msg.contains("tok_embeddings.weight")
    ));
}

#[test]
fn test_adapter_resolve_token_embedding_invalid_rank() {
    let gate_payload = vec![0u8; EMBEDDING_DIM * 64 * size_of::<f32>()];
    let attn_q_payload = vec![0u8; 16];

    let checkpoint = build_test_gguf(
        vec![
            (
                "blk.0.ffn_gate_inp.weight",
                vec![EMBEDDING_DIM, 64],
                GGML_TYPE_F32,
                gate_payload,
            ),
            (
                "blk.0.attn_q.weight",
                vec![EMBEDDING_DIM, EMBEDDING_DIM],
                GGML_TYPE_IQ3_S,
                attn_q_payload,
            ),
            (
                "token_embd.weight",
                vec![EMBEDDING_DIM],
                GGML_TYPE_F16,
                vec![0u8; EMBEDDING_DIM * 2],
            ),
        ],
        32,
    );

    let path = write_temp_file(&checkpoint, "token-embedding-invalid-rank");
    let mapped = probe_and_map_checkpoint(path.to_str().unwrap()).unwrap();
    let result =
        super::adapter::resolve_adapter(mapped.metadata(), &mapped, None, path.to_str().unwrap());

    assert!(matches!(
        result,
        Err(HybridError::UnsupportedFormat(msg)) if msg.contains("token_embd.weight")
    ));
}

#[test]
fn test_adapter_resolve_token_embedding_mismatched_width() {
    let gate_payload = vec![0u8; EMBEDDING_DIM * 64 * size_of::<f32>()];
    let attn_q_payload = vec![0u8; 16];

    let checkpoint = build_test_gguf(
        vec![
            (
                "blk.0.ffn_gate_inp.weight",
                vec![EMBEDDING_DIM, 64],
                GGML_TYPE_F32,
                gate_payload,
            ),
            (
                "blk.0.attn_q.weight",
                vec![EMBEDDING_DIM, EMBEDDING_DIM],
                GGML_TYPE_IQ3_S,
                attn_q_payload,
            ),
            (
                "token_embd.weight",
                vec![EMBEDDING_DIM / 2, 32],
                GGML_TYPE_F16,
                vec![0u8; EMBEDDING_DIM / 2 * 32 * 2],
            ),
        ],
        32,
    );

    let path = write_temp_file(&checkpoint, "token-embedding-mismatched-width");
    let mapped = probe_and_map_checkpoint(path.to_str().unwrap()).unwrap();
    let result =
        super::adapter::resolve_adapter(mapped.metadata(), &mapped, None, path.to_str().unwrap());

    assert!(matches!(
        result,
        Err(HybridError::UnsupportedFormat(msg)) if msg.contains("token_embd.weight")
    ));
}

#[test]
fn test_adapter_resolve_token_embedding_zero_vocab() {
    let gate_payload = vec![0u8; EMBEDDING_DIM * 64 * size_of::<f32>()];
    let attn_q_payload = vec![0u8; 16];

    let checkpoint = build_test_gguf(
        vec![
            (
                "blk.0.ffn_gate_inp.weight",
                vec![EMBEDDING_DIM, 64],
                GGML_TYPE_F32,
                gate_payload,
            ),
            (
                "blk.0.attn_q.weight",
                vec![EMBEDDING_DIM, EMBEDDING_DIM],
                GGML_TYPE_IQ3_S,
                attn_q_payload,
            ),
            (
                "token_embd.weight",
                vec![EMBEDDING_DIM, 0],
                GGML_TYPE_F16,
                vec![0u8; 0],
            ),
        ],
        32,
    );

    let path = write_temp_file(&checkpoint, "token-embedding-zero-vocab");
    let mapped = probe_and_map_checkpoint(path.to_str().unwrap()).unwrap();
    let result =
        super::adapter::resolve_adapter(mapped.metadata(), &mapped, None, path.to_str().unwrap());

    assert!(matches!(
        result,
        Err(HybridError::UnsupportedFormat(msg)) if msg.contains("token_embd.weight")
    ));
}

#[test]
fn test_adapter_resolve_token_embedding_q8_0_misaligned_width() {
    // A Q8_0 token embedding whose row width is not divisible by the 32-element
    // block size should fail adapter resolution early.
    let gate_payload = vec![0u8; EMBEDDING_DIM * 64 * size_of::<f32>()];
    let attn_q_payload = vec![0u8; 16];
    let hidden_size = 2064usize;
    let vocab_size = 32usize;
    let numerics = OLMOE_NUMERICS.map(|(key, value)| {
        if key == "olmoe.embedding_length" {
            (key, hidden_size as u32)
        } else {
            (key, value)
        }
    });

    let checkpoint = build_test_gguf_with_numerics(
        vec![
            (
                "blk.0.ffn_gate_inp.weight",
                vec![EMBEDDING_DIM, 64],
                GGML_TYPE_F32,
                gate_payload,
            ),
            (
                "blk.0.attn_q.weight",
                vec![EMBEDDING_DIM, EMBEDDING_DIM],
                GGML_TYPE_IQ3_S,
                attn_q_payload,
            ),
            (
                "token_embd.weight",
                vec![hidden_size, vocab_size],
                GGML_TYPE_Q8_0,
                vec![0u8; hidden_size * vocab_size],
            ),
        ],
        32,
        &numerics,
    );

    let path = write_temp_file(&checkpoint, "token-embedding-q8-misaligned");
    let mapped = probe_and_map_checkpoint(path.to_str().unwrap()).unwrap();
    let result =
        super::adapter::resolve_adapter(mapped.metadata(), &mapped, None, path.to_str().unwrap());

    assert!(matches!(
        result,
        Err(HybridError::UnsupportedFormat(msg)) if msg.contains("token_embd.weight")
    ));
}

#[test]
fn test_adapter_does_not_treat_wire_type_31_as_iq3_m() {
    // GGUF wire type 31 is Q4_0_4_4 — must fall through (usually routing-f32),
    // never DequantizedIQ3M (Codex).
    let checkpoint = build_test_gguf(
        vec![
            (
                "blk.0.ffn_gate_inp.weight",
                vec![EMBEDDING_DIM, 64],
                GGML_TYPE_F32,
                vec![0u8; EMBEDDING_DIM * 64 * 4],
            ),
            (
                "blk.0.attn_q.weight",
                vec![256, 1],
                GGML_TYPE_Q4_0_4_4,
                vec![0u8; 256],
            ),
            (
                "token_embd.weight",
                vec![EMBEDDING_DIM, 32],
                GGML_TYPE_F16,
                vec![0u8; EMBEDDING_DIM * 32 * 2],
            ),
        ],
        32,
    );

    let path = write_temp_file(&checkpoint, "type31_not_iq3m");
    let mapped = probe_and_map_checkpoint(path.to_str().unwrap()).unwrap();
    let adapter =
        super::adapter::resolve_adapter(mapped.metadata(), &mapped, None, path.to_str().unwrap())
            .unwrap();
    assert_ne!(
        adapter.synapse_source,
        super::adapter::SynapseSource::DequantizedIQ3M
    );
    assert_eq!(
        adapter.synapse_source,
        super::adapter::SynapseSource::RoutingF32
    );
}

#[test]
fn test_adapter_resolve_token_embedding_missing() {
    let gate_payload = vec![0u8; EMBEDDING_DIM * 64 * size_of::<f32>()];
    let attn_q_payload = vec![0u8; 16];

    let checkpoint = build_test_gguf(
        vec![
            (
                "blk.0.ffn_gate_inp.weight",
                vec![EMBEDDING_DIM, 64],
                GGML_TYPE_F32,
                gate_payload,
            ),
            (
                "blk.0.attn_q.weight",
                vec![EMBEDDING_DIM, EMBEDDING_DIM],
                GGML_TYPE_IQ3_S,
                attn_q_payload,
            ),
        ],
        32,
    );

    let path = write_temp_file(&checkpoint, "token-embedding-missing");
    let mapped = probe_and_map_checkpoint(path.to_str().unwrap()).unwrap();
    let result =
        super::adapter::resolve_adapter(mapped.metadata(), &mapped, None, path.to_str().unwrap());

    assert!(matches!(
        result,
        Err(HybridError::MissingTensor { name, .. }) if name == "token_embd.weight"
    ));
}

#[test]
fn test_adapter_resolve_routing_missing() {
    let attn_q_payload = vec![0u8; 16];

    let checkpoint = build_test_gguf(
        vec![
            (
                "blk.0.attn_q.weight",
                vec![EMBEDDING_DIM, EMBEDDING_DIM],
                GGML_TYPE_IQ3_S,
                attn_q_payload,
            ),
            (
                "token_embd.weight",
                vec![EMBEDDING_DIM, 32],
                GGML_TYPE_F16,
                vec![0u8; EMBEDDING_DIM * 32 * 2],
            ),
        ],
        32,
    );

    let path = write_temp_file(&checkpoint, "routing-missing");
    let mapped = probe_and_map_checkpoint(path.to_str().unwrap()).unwrap();
    let result =
        super::adapter::resolve_adapter(mapped.metadata(), &mapped, None, path.to_str().unwrap());

    assert!(matches!(
        result,
        Err(HybridError::MissingTensor { name, .. }) if name == "ffn_gate_inp.weight"
    ));
}

#[test]
fn test_adapter_resolve_routing_wrong_type() {
    let gate_payload = build_q8_0_payload(EMBEDDING_DIM, 64, 0x3c00, 1);
    let attn_q_payload = vec![0u8; 16];

    let checkpoint = build_test_gguf(
        vec![
            (
                "blk.0.ffn_gate_inp.weight",
                vec![EMBEDDING_DIM, 64],
                GGML_TYPE_Q8_0,
                gate_payload,
            ),
            (
                "blk.0.attn_q.weight",
                vec![EMBEDDING_DIM, EMBEDDING_DIM],
                GGML_TYPE_IQ3_S,
                attn_q_payload,
            ),
            (
                "token_embd.weight",
                vec![EMBEDDING_DIM, 32],
                GGML_TYPE_F16,
                vec![0u8; EMBEDDING_DIM * 32 * 2],
            ),
        ],
        32,
    );

    let path = write_temp_file(&checkpoint, "routing-wrong-type");
    let mapped = probe_and_map_checkpoint(path.to_str().unwrap()).unwrap();
    let result =
        super::adapter::resolve_adapter(mapped.metadata(), &mapped, None, path.to_str().unwrap());

    assert!(matches!(
        result,
        Err(HybridError::UnsupportedFormat(msg)) if msg.contains("must be rank-2 F32")
    ));
}

#[test]
fn test_adapter_resolve_routing_insufficient_experts() {
    let gate_payload = vec![0u8; EMBEDDING_DIM * size_of::<f32>()];
    let attn_q_payload = vec![0u8; 16];

    let checkpoint = build_test_gguf(
        vec![
            (
                "blk.0.ffn_gate_inp.weight",
                vec![EMBEDDING_DIM, 1],
                GGML_TYPE_F32,
                gate_payload,
            ),
            (
                "blk.0.attn_q.weight",
                vec![EMBEDDING_DIM, EMBEDDING_DIM],
                GGML_TYPE_IQ3_S,
                attn_q_payload,
            ),
            (
                "token_embd.weight",
                vec![EMBEDDING_DIM, 32],
                GGML_TYPE_F16,
                vec![0u8; EMBEDDING_DIM * 32 * 2],
            ),
        ],
        32,
    );

    let path = write_temp_file(&checkpoint, "routing-insufficient-experts");
    let mapped = probe_and_map_checkpoint(path.to_str().unwrap()).unwrap();
    let result =
        super::adapter::resolve_adapter(mapped.metadata(), &mapped, None, path.to_str().unwrap());

    assert!(matches!(
        result,
        Err(HybridError::UnsupportedFormat(msg)) if msg.contains("only exposes 1 experts")
    ));
}

#[test]
fn test_adapter_resolve_routing_invalid_orientation() {
    // Routing tensor has enough experts but neither dimension equals hidden_size.
    let gate_payload = vec![0u8; 64 * 128 * size_of::<f32>()];
    let attn_q_payload = vec![0u8; 16];

    let checkpoint = build_test_gguf(
        vec![
            (
                "blk.0.ffn_gate_inp.weight",
                vec![64, 128],
                GGML_TYPE_F32,
                gate_payload,
            ),
            (
                "blk.0.attn_q.weight",
                vec![EMBEDDING_DIM, EMBEDDING_DIM],
                GGML_TYPE_IQ3_S,
                attn_q_payload,
            ),
            (
                "token_embd.weight",
                vec![EMBEDDING_DIM, 32],
                GGML_TYPE_F16,
                vec![0u8; EMBEDDING_DIM * 32 * 2],
            ),
        ],
        32,
    );

    let path = write_temp_file(&checkpoint, "routing-invalid-orientation");
    let mapped = probe_and_map_checkpoint(path.to_str().unwrap()).unwrap();
    let result =
        super::adapter::resolve_adapter(mapped.metadata(), &mapped, None, path.to_str().unwrap());

    assert!(matches!(
        result,
        Err(HybridError::UnsupportedFormat(msg)) if msg.contains("unsupported orientation")
    ));
}

#[test]
fn test_adapter_resolve_quantized_synapse_rank_not_two() {
    // attn_q is rank-1, so select_quantized_synapse should return None early.
    let gate_payload = vec![0u8; EMBEDDING_DIM * 64 * size_of::<f32>()];
    let attn_q_payload = build_q8_0_payload(EMBEDDING_DIM, 1, 0x3c00, 1);

    let checkpoint = build_test_gguf(
        vec![
            (
                "blk.0.ffn_gate_inp.weight",
                vec![EMBEDDING_DIM, 64],
                GGML_TYPE_F32,
                gate_payload,
            ),
            (
                "blk.0.attn_q.weight",
                vec![EMBEDDING_DIM],
                GGML_TYPE_Q8_0,
                attn_q_payload,
            ),
            (
                "token_embd.weight",
                vec![EMBEDDING_DIM, 32],
                GGML_TYPE_F16,
                vec![0u8; EMBEDDING_DIM * 32 * 2],
            ),
        ],
        32,
    );

    let path = write_temp_file(&checkpoint, "quantized-rank-not-two");
    let mapped = probe_and_map_checkpoint(path.to_str().unwrap()).unwrap();
    let adapter =
        super::adapter::resolve_adapter(mapped.metadata(), &mapped, None, path.to_str().unwrap())
            .unwrap();

    assert_eq!(
        adapter.synapse_source,
        super::adapter::SynapseSource::RoutingF32
    );
    assert_eq!(
        adapter.synapse_tensor.as_deref(),
        Some("blk.0.ffn_gate_inp.weight")
    );
}

fn build_standard_adapter_gguf(numerics: &[(&str, u32)]) -> Vec<u8> {
    build_test_gguf_with_numerics(
        vec![
            (
                "blk.0.ffn_gate_inp.weight",
                vec![EMBEDDING_DIM, 64],
                GGML_TYPE_F32,
                vec![0u8; EMBEDDING_DIM * 64 * size_of::<f32>()],
            ),
            (
                "blk.0.attn_q.weight",
                vec![EMBEDDING_DIM, EMBEDDING_DIM],
                GGML_TYPE_IQ3_S,
                vec![0u8; 16],
            ),
            (
                "token_embd.weight",
                vec![EMBEDDING_DIM, 32],
                GGML_TYPE_F16,
                vec![0u8; EMBEDDING_DIM * 32 * 2],
            ),
        ],
        32,
        numerics,
    )
}

fn assert_topology_key_required(key: &str, fixture_name: &str, expected_fragment: &str) {
    let numerics: Vec<(&str, u32)> = OLMOE_NUMERICS
        .into_iter()
        .filter(|(name, _)| *name != key)
        .collect();
    assert_eq!(numerics.len(), OLMOE_NUMERICS.len() - 1, "key={key}");
    let checkpoint = build_standard_adapter_gguf(&numerics);
    let path = write_temp_file(&checkpoint, fixture_name);
    let mapped = probe_and_map_checkpoint(path.to_str().unwrap()).unwrap();

    let result =
        super::adapter::resolve_adapter(mapped.metadata(), &mapped, None, path.to_str().unwrap());

    assert!(matches!(
        result,
        Err(HybridError::UnsupportedFormat(msg)) if msg.contains(expected_fragment)
    ));
}

#[test]
fn test_adapter_resolve_gguf_topology_missing_embedding_length() {
    assert_topology_key_required(
        "olmoe.embedding_length",
        "topology-missing-embedding-length",
        "embedding_length",
    );
}

#[test]
fn test_adapter_resolve_gguf_topology_missing_block_count() {
    assert_topology_key_required(
        "olmoe.block_count",
        "topology-missing-block-count",
        "block_count",
    );
}

#[test]
fn test_adapter_resolve_gguf_topology_missing_expert_count() {
    assert_topology_key_required(
        "olmoe.expert_count",
        "topology-missing-expert-count",
        "expert_count",
    );
}

#[test]
fn test_synapse_source_label_synthetic_fallback() {
    let adapter = super::adapter::ModelAdapter {
        family: ModelFamily::Olmoe,
        architecture: "olmoe".to_owned(),
        hidden_size: EMBEDDING_DIM,
        num_layers: 16,
        num_experts: 64,
        expert_used_count: 8,
        token_embedding_tensor: "token_embd.weight".to_owned(),
        routing_tensor: "blk.0.ffn_gate_inp.weight".to_owned(),
        preferred_gpu_synapse_tensor: None,
        synapse_tensor: None,
        synapse_source: super::adapter::SynapseSource::SyntheticFallback,
        quantization: "Q8_0".to_owned(),
    };

    assert_eq!(adapter.synapse_source_label(), "synthetic-fallback");
}

#[test]
fn test_iq3_m_multi_row_dequantization() {
    // Test dequantize_iq3_m_tensor with multiple rows (covers the row loop)
    let mut block = vec![0u8; 111];
    block[0] = 0x00;
    block[1] = 0x3C;
    let payload = [block.clone(), block].concat();

    let checkpoint = build_test_gguf(
        vec![
            (
                "blk.0.ffn_gate_inp.weight",
                vec![EMBEDDING_DIM, 64],
                GGML_TYPE_F32,
                vec![0u8; EMBEDDING_DIM * 64 * 4],
            ),
            (
                "blk.0.attn_q.weight",
                vec![256, 2],
                GGML_TYPE_IQ3_M_BLOCK,
                payload,
            ),
            (
                "token_embd.weight",
                vec![EMBEDDING_DIM, 32],
                GGML_TYPE_F16,
                vec![0u8; EMBEDDING_DIM * 32 * 2],
            ),
        ],
        32,
    );

    let path = write_temp_file(&checkpoint, "iq3_m_multi");
    let mapped = probe_and_map_checkpoint(path.to_str().unwrap()).unwrap();

    let result = mapped.dequantize_iq3_m_tensor("blk.0.attn_q.weight", path.to_str().unwrap());
    assert!(result.is_ok());
    let dequantized = result.unwrap();
    assert_eq!(dequantized.len(), 512);
    for &val in &dequantized {
        assert_eq!(val, 0.0, "expected 0.0 with zero scales, got {val}");
    }
}

#[test]
fn test_safetensors_u4_unsigned_extraction() {
    use super::safetensors::MappedSafetensorsCheckpoint;

    let tensor_name = "test.u4";
    let shape = vec![1usize, 2];
    let data = vec![0x8Fu8];

    let header = serde_json::json!({
        tensor_name: {
            "dtype": "U4",
            "shape": shape,
            "data_offsets": [0, data.len()]
        },
        "__metadata__": {}
    });
    let tmp_dir = temp_dir("st_u4_test");
    let st_path = tmp_dir.join("model.safetensors");
    write_safetensors_bytes(&st_path, &header, &data);

    let config = serde_json::json!({
        "architectures": ["TestModel"],
        "hidden_size": 128,
        "num_hidden_layers": 2,
        "vocab_size": 1000
    });
    write_hf_config(&tmp_dir, &config);

    let checkpoint =
        MappedSafetensorsCheckpoint::from_directory(tmp_dir.to_str().unwrap()).unwrap();
    let extracted = checkpoint
        .extract_tensor_f32(tensor_name, st_path.to_str().unwrap())
        .unwrap();

    // U4: unsigned, so 0x8F -> low=15, high=8 (no sign extension)
    assert_eq!(extracted, vec![15.0, 8.0]);
}

#[test]
fn test_safetensors_int4_odd_element_count_drops_trailing_nibble() {
    use super::safetensors::MappedSafetensorsCheckpoint;

    // 3 elements occupy 2 bytes, so the second byte's high nibble is padding.
    let cases: [(&str, [u8; 2], [f32; 3]); 2] = [
        ("U4", [0x12, 0x03], [2.0, 1.0, 3.0]),
        ("I4", [0x8F, 0x07], [-1.0, -8.0, 7.0]),
    ];
    for (dtype, data, expected) in cases {
        let header = serde_json::json!({
            "t": {
                "dtype": dtype,
                "shape": [1, 3],
                "data_offsets": [0, 2]
            }
        });
        let tmp_dir = temp_dir("st_odd_int4");
        let st_path = tmp_dir.join("model.safetensors");
        write_safetensors_bytes(&st_path, &header, &data);

        let config = serde_json::json!({
            "architectures": ["TestModel"],
            "hidden_size": 128,
            "num_hidden_layers": 2,
            "vocab_size": 1000
        });
        write_hf_config(&tmp_dir, &config);

        let checkpoint =
            MappedSafetensorsCheckpoint::from_directory(tmp_dir.to_str().unwrap()).unwrap();
        let extracted = checkpoint
            .extract_tensor_f32("t", st_path.to_str().unwrap())
            .unwrap();

        assert_eq!(extracted, expected, "dtype={dtype}");
    }
}

#[test]
fn test_safetensors_extract_token_embedding_errors() {
    use super::safetensors::MappedSafetensorsCheckpoint;

    let tensor_name = "test.f32";
    let shape = vec![2usize, 4];
    let data = vec![0u8; 32];

    let header = serde_json::json!({
        tensor_name: {
            "dtype": "F32",
            "shape": shape,
            "data_offsets": [0, data.len()]
        },
        "__metadata__": {}
    });
    let tmp_dir = temp_dir("st_err_test");
    let st_path = tmp_dir.join("model.safetensors");
    write_safetensors_bytes(&st_path, &header, &data);

    let config = serde_json::json!({
        "architectures": ["TestModel"],
        "hidden_size": 128,
        "num_hidden_layers": 2,
        "vocab_size": 1000
    });
    write_hf_config(&tmp_dir, &config);

    let checkpoint =
        MappedSafetensorsCheckpoint::from_directory(tmp_dir.to_str().unwrap()).unwrap();

    // Test missing tensor
    let result = checkpoint.extract_tensor_f32("missing", st_path.to_str().unwrap());
    assert!(result.is_err());
}

#[test]
fn test_gguf_f16_token_embedding_reads_only_the_requested_row() {
    // f16 bits for 1.0, 2.0, 0.5, -1.0, -2.0, 4.0: 3 rows of width 2.
    let bits: [u16; 6] = [0x3C00, 0x4000, 0x3800, 0xBC00, 0xC000, 0x4400];
    let payload: Vec<u8> = bits.iter().flat_map(|b| b.to_le_bytes()).collect();
    let checkpoint = build_test_gguf(
        vec![("token_embd.weight", vec![2, 3], GGML_TYPE_F16, payload)],
        32,
    );
    let path = write_temp_file(&checkpoint, "f16_token_row");
    let path_str = path.to_str().unwrap();
    let mapped = probe_and_map_checkpoint(path_str).unwrap();

    assert_eq!(
        mapped
            .extract_token_embedding("token_embd.weight", path_str, 1)
            .unwrap(),
        vec![0.5, -1.0]
    );
    let err = mapped
        .extract_token_embedding("token_embd.weight", path_str, 3)
        .unwrap_err();
    assert!(
        matches!(
            err,
            HybridError::InputLengthMismatch {
                expected: 3,
                got: 3
            }
        ),
        "{err}"
    );
}

#[test]
fn synthetic_router_metadata_uses_the_shared_fallback_label() {
    // `run_manifest.json` persists this string and CLAUDE.md pins it, so the
    // stub metadata and the SynapseSource label table must agree. They were
    // two independent string literals before.
    let metadata = RouterMetadata::synthetic(ModelFamily::Olmoe, 8, 2);
    assert_eq!(metadata.synapse_source, SYNTHETIC_FALLBACK_SOURCE);
    assert_eq!(metadata.synapse_source, "synthetic-fallback");
}
