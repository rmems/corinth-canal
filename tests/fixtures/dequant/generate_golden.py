# SPDX-License-Identifier: Apache-2.0 OR MIT
"""Regenerate the Q5_K / Q6_K dequant golden vectors (RM-1415).

The expected values come from gguf-py (the reference Python package that ships
with llama.cpp), not from corinth-canal's own decoder. That keeps the oracle
independent: a porting mistake in src/moe/gguf/dequant.rs cannot also appear
in its goldens.

    python3 -m venv .venv-golden && .venv-golden/bin/pip install gguf==0.19.0
    .venv-golden/bin/python tests/fixtures/dequant/generate_golden.py

Each `<type>.blocks.bin` holds two raw GGML blocks with seeded random,
non-uniform quant bytes and finite f16 scales; `<type>.expected.f32` holds the
512 little-endian f32 values gguf-py decodes from them.
"""

import pathlib

import numpy as np
from gguf import GGMLQuantizationType
from gguf.quants import dequantize

OUT = pathlib.Path(__file__).resolve().parent
rng = np.random.default_rng(1415)


def f16_bytes(value: float) -> bytes:
    return np.array([value], dtype="<f2").tobytes()


def q6_k_block(d: float) -> bytes:
    # ggml block_q6_K: ql[128], qh[64], int8 scales[16], f16 d (210 bytes).
    ql = rng.integers(0, 256, 128, dtype=np.uint8).tobytes()
    qh = rng.integers(0, 256, 64, dtype=np.uint8).tobytes()
    scales = rng.integers(-64, 64, 16, dtype=np.int8).tobytes()
    return ql + qh + scales + f16_bytes(d)


def q5_k_block(d: float, dmin: float) -> bytes:
    # ggml block_q5_K: f16 d, f16 dmin, scales[12], qh[32], qs[128] (176 bytes).
    scales = rng.integers(0, 256, 12, dtype=np.uint8).tobytes()
    qh = rng.integers(0, 256, 32, dtype=np.uint8).tobytes()
    qs = rng.integers(0, 256, 128, dtype=np.uint8).tobytes()
    return f16_bytes(d) + f16_bytes(dmin) + scales + qh + qs


def write(name: str, qtype: GGMLQuantizationType, raw: bytes) -> None:
    values = dequantize(np.frombuffer(raw, dtype=np.uint8), qtype).astype("<f4").ravel()
    if values.size != 512 or not np.isfinite(values).all():
        raise SystemExit(f"{name}: expected 512 finite values, got {values.size}")
    (OUT / f"{name}.blocks.bin").write_bytes(raw)
    (OUT / f"{name}.expected.f32").write_bytes(values.tobytes())
    print(name, len(raw), "bytes ->", values.size, "values; head", values[:4])


write("q6_k", GGMLQuantizationType.Q6_K, q6_k_block(0.0123) + q6_k_block(0.0071))
write("q5_k", GGMLQuantizationType.Q5_K, q5_k_block(0.0089, 0.0042) + q5_k_block(0.0153, 0.0017))
