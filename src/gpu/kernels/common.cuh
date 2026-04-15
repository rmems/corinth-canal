// ════════════════════════════════════════════════════════════════════
//  common.cuh — Shared definitions for all neuro-spike CUDA kernels
//
//  Target: sm_120 (RTX 5080 Blackwell) · CUDA 13.x
// ════════════════════════════════════════════════════════════════════

#pragma once
#include <cuda_runtime.h>
#include <math.h>
#include <limits.h>

// ── Numeric constants ──────────────────────────────────────────────
#define SHIP_PI       3.14159265358979323846f
#define SHIP_E        2.71828182845904523536f
#define SHIP_EPS      1e-8f

// ── Warp utility ──────────────────────────────────────────────────
#define WARP_SIZE 32

// Fast warp reduction (sum)
__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffffu, val, offset);
    return val;
}

// Fast warp reduction (max)
__device__ __forceinline__ float warp_reduce_max(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        val = fmaxf(val, __shfl_down_sync(0xffffffffu, val, offset));
    return val;
}

// Fast warp reduction (int sum)
__device__ __forceinline__ int warp_reduce_sum_int(int val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffffu, val, offset);
    return val;
}

// Fast warp reduction (int min)
__device__ __forceinline__ int warp_reduce_min_int(int val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        val = min(val, __shfl_down_sync(0xffffffffu, val, offset));
    return val;
}

// ── Simple LCG random (inline, no cuRAND dependency) ──────────────
__device__ __forceinline__ unsigned int lcg_next(unsigned int state) {
    return state * 1664525u + 1013904223u;
}

// Convert an already-advanced LCG state to a float in [0, 1).
// Callers must call lcg_next() before this function to advance
// the RNG state; lcg_float itself is a pure conversion.
__device__ __forceinline__ float lcg_float(unsigned int state) {
    return (float)(state >> 8) * (1.0f / 16777216.0f);
}
