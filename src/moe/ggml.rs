// SPDX-License-Identifier: Apache-2.0 OR MIT
//! GGUF/GGML type constants, labels, and diagnostic helpers.
//!
//! These `u32` values are **GGUF on-disk type ids** (the wire enum defined by
//! the GGUF format / ggml). This crate does **not** link llama.cpp; it only
//! reads the same integer tags that GGUF writers embed in tensor headers.
//! The reference enum lives in ggml/llama.cpp documentation — that is why
//! Codex cites it when our labels disagree with the wire format.

pub(super) const GGUF_MAGIC: [u8; 4] = *b"GGUF";
pub(super) const GGUF_VERSION: u32 = 3;
pub(super) const GGML_TYPE_F32: u32 = 0;
pub(super) const GGML_TYPE_F16: u32 = 1;
pub(super) const GGML_TYPE_Q8_0: u32 = 8;
pub(super) const GGML_TYPE_Q5_K: u32 = 13;
pub(super) const GGML_TYPE_Q6_K: u32 = 14;
pub(super) const GGML_TYPE_IQ3_S: u32 = 21;
/// GGUF wire type 31: historical `Q4_0_4_4` layout (removed from current ggml).
/// Must **not** be treated as an IQ3_M block type.
pub(super) const GGML_TYPE_Q4_0_4_4: u32 = 31;
/// Internal id for our 111-byte IQ3_M *block layout* decoder only.
///
/// Not a GGUF wire type: HuggingFace “IQ3_M” is a mixed-quant *preset*, not
/// type id 31. Synthetic fixtures and unit tests may use this constant;
/// production GGUFs never emit it, so the live adapter only hits this path
/// under test.
pub(super) const GGML_TYPE_IQ3_M_BLOCK: u32 = 0x4949_334D;
pub(super) const GGUF_VALUE_TYPE_UINT8: u32 = 0;
pub(super) const GGUF_VALUE_TYPE_INT8: u32 = 1;
pub(super) const GGUF_VALUE_TYPE_UINT16: u32 = 2;
pub(super) const GGUF_VALUE_TYPE_INT16: u32 = 3;
pub(super) const GGUF_VALUE_TYPE_UINT32: u32 = 4;
pub(super) const GGUF_VALUE_TYPE_INT32: u32 = 5;
pub(super) const GGUF_VALUE_TYPE_FLOAT32: u32 = 6;
pub(super) const GGUF_VALUE_TYPE_BOOL: u32 = 7;
pub(super) const GGUF_VALUE_TYPE_STRING: u32 = 8;
pub(super) const GGUF_VALUE_TYPE_ARRAY: u32 = 9;
pub(super) const GGUF_VALUE_TYPE_UINT64: u32 = 10;
pub(super) const GGUF_VALUE_TYPE_INT64: u32 = 11;
pub(super) const GGUF_VALUE_TYPE_FLOAT64: u32 = 12;

/// Map a GGUF `ggml_type` u32 to a short human label.
pub fn ggml_type_label(ggml_type: u32) -> &'static str {
    match ggml_type {
        GGML_TYPE_F32 => "F32",
        GGML_TYPE_F16 => "F16",
        2 => "Q4_0",
        3 => "Q4_1",
        6 => "Q5_0",
        7 => "Q5_1",
        GGML_TYPE_Q8_0 => "Q8_0",
        9 => "Q8_1",
        10 => "Q2_K",
        11 => "Q3_K",
        12 => "Q4_K",
        GGML_TYPE_Q5_K => "Q5_K",
        GGML_TYPE_Q6_K => "Q6_K",
        15 => "Q8_K",
        16 => "IQ2_XXS",
        17 => "IQ2_XS",
        18 => "IQ3_XXS",
        19 => "IQ1_S",
        20 => "IQ4_NL",
        GGML_TYPE_IQ3_S => "IQ3_S",
        22 => "IQ2_S",
        23 => "IQ4_XS",
        24 => "I8",
        25 => "I16",
        26 => "I32",
        27 => "I64",
        28 => "F64",
        29 => "IQ1_M",
        30 => "BF16",
        GGML_TYPE_Q4_0_4_4 => "Q4_0_4_4",
        GGML_TYPE_IQ3_M_BLOCK => "IQ3_M_BLOCK",
        _ => "unknown",
    }
}

/// Returns `true` iff the runtime can consume `ggml_type` as the source
/// for the GPU synapse tensor today.
pub fn synapse_dequant_path_supported(ggml_type: u32) -> bool {
    ggml_type == GGML_TYPE_F16
        || ggml_type == GGML_TYPE_Q8_0
        || ggml_type == GGML_TYPE_Q5_K
        || ggml_type == GGML_TYPE_Q6_K
        // Internal block-format id only (not wire type 31 / Q4_0_4_4).
        || ggml_type == GGML_TYPE_IQ3_M_BLOCK
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The GGUF wire ids, the internal IQ3_M block id, and ids that must fall
    /// through to "unknown" (4 and 5 are retired wire ids).
    const LABELS: [(u32, &str); 34] = [
        (0, "F32"),
        (1, "F16"),
        (2, "Q4_0"),
        (3, "Q4_1"),
        (6, "Q5_0"),
        (7, "Q5_1"),
        (8, "Q8_0"),
        (9, "Q8_1"),
        (10, "Q2_K"),
        (11, "Q3_K"),
        (12, "Q4_K"),
        (13, "Q5_K"),
        (14, "Q6_K"),
        (15, "Q8_K"),
        (16, "IQ2_XXS"),
        (17, "IQ2_XS"),
        (18, "IQ3_XXS"),
        (19, "IQ1_S"),
        (20, "IQ4_NL"),
        (21, "IQ3_S"),
        (22, "IQ2_S"),
        (23, "IQ4_XS"),
        (24, "I8"),
        (25, "I16"),
        (26, "I32"),
        (27, "I64"),
        (28, "F64"),
        (29, "IQ1_M"),
        (30, "BF16"),
        (31, "Q4_0_4_4"),
        (GGML_TYPE_IQ3_M_BLOCK, "IQ3_M_BLOCK"),
        (4, "unknown"),
        (5, "unknown"),
        (9999, "unknown"),
    ];

    #[test]
    fn ggml_type_label_table() {
        for (id, expected) in LABELS {
            assert_eq!(ggml_type_label(id), expected, "ggml_type={id}");
        }
    }

    #[test]
    fn synapse_dequant_path_supported_is_exactly_the_five_dequant_ids() {
        for (id, _) in LABELS {
            let expected = matches!(
                id,
                GGML_TYPE_F16
                    | GGML_TYPE_Q8_0
                    | GGML_TYPE_Q5_K
                    | GGML_TYPE_Q6_K
                    | GGML_TYPE_IQ3_M_BLOCK
            );
            assert_eq!(
                synapse_dequant_path_supported(id),
                expected,
                "ggml_type={id}"
            );
        }

        // Wire type 31 is Q4_0_4_4, not IQ3_M (Codex / GGUF enum).
        assert_eq!(GGML_TYPE_Q4_0_4_4, 31);
        assert!(!synapse_dequant_path_supported(31));
    }
}
