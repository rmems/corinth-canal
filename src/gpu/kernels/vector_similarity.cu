// ════════════════════════════════════════════════════════════════════
//  vector_similarity.cu — GPU cosine-similarity kernels
//
//  Kernels exported:
//    cosine_similarity_batched  — compute similarity of Q query vectors
//                                 against K key vectors (Q×K matrix)
//    cosine_similarity_top_k    — like above but also write top-k indices
//
//  Target: sm_120 (RTX 5080 Blackwell) · CUDA 13.x
// ════════════════════════════════════════════════════════════════════

#include "common.cuh"

#define MAX_ROUTING_TOP_K 32
#define MAX_BLOCK_WARPS   32
#define COSINE_SENTINEL   -2.0f

__device__ __forceinline__
bool topk_better(
    float lhs_score,
    int lhs_idx,
    float rhs_score,
    int rhs_idx)
{
    return lhs_score > rhs_score ||
           (lhs_score == rhs_score && lhs_idx < rhs_idx);
}

__device__ __forceinline__
void topk_insert(
    float score,
    int idx,
    float (&scores)[MAX_ROUTING_TOP_K],
    int (&indices)[MAX_ROUTING_TOP_K],
    int top_k)
{
    if (top_k <= 0) return;

    int tail = top_k - 1;
    if (!topk_better(score, idx, scores[tail], indices[tail])) return;

    scores[tail] = score;
    indices[tail] = idx;

    for (int pos = tail; pos > 0; --pos) {
        if (!topk_better(scores[pos], indices[pos], scores[pos - 1], indices[pos - 1]))
            break;

        float tmp_score = scores[pos - 1];
        int tmp_idx = indices[pos - 1];
        scores[pos - 1] = scores[pos];
        indices[pos - 1] = indices[pos];
        scores[pos] = tmp_score;
        indices[pos] = tmp_idx;
    }
}

__device__ __forceinline__
void warp_reduce_best(
    float& score,
    int& idx,
    int& owner_lane)
{
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        float other_score = __shfl_down_sync(0xffffffffu, score, offset);
        int other_idx = __shfl_down_sync(0xffffffffu, idx, offset);
        int other_owner = __shfl_down_sync(0xffffffffu, owner_lane, offset);

        if (topk_better(other_score, other_idx, score, idx)) {
            score = other_score;
            idx = other_idx;
            owner_lane = other_owner;
        }
    }

    score = __shfl_sync(0xffffffffu, score, 0);
    idx = __shfl_sync(0xffffffffu, idx, 0);
    owner_lane = __shfl_sync(0xffffffffu, owner_lane, 0);
}

// ════════════════════════════════════════════════════════════════════
//  cosine_similarity_batched
//
//  Computes the cosine similarity between every (query, key) pair:
//    out[q * n_keys + k] = dot(Q[q], K[k]) / (|Q[q]| * |K[k]| + eps)
//
//  Grid: (n_keys, n_queries) blocks of (dim,) threads — one block per pair.
//        For large dim use TILE_DIM threads and loop.
//
//  Params
//    queries    [n_queries × dim]  — query matrix (row-major)
//    keys       [n_keys   × dim]  — key matrix
//    out        [n_queries × n_keys] — output similarity matrix
//    n_queries, n_keys, dim
// ════════════════════════════════════════════════════════════════════
extern "C" __global__
void cosine_similarity_batched(
    const float* __restrict__ queries,
    const float* __restrict__ keys,
    float*       __restrict__ out,
    int n_queries,
    int n_keys,
    int dim)
{
    // Block index identifies (query, key) pair
    int q = blockIdx.y;
    int k = blockIdx.x;
    if (q >= n_queries || k >= n_keys) return;

    const float* qv = queries + (long)q * dim;
    const float* kv = keys    + (long)k * dim;

    // Each thread accumulates a partial dot product and L2 norms
    float dot   = 0.0f;
    float norm_q = 0.0f;
    float norm_k = 0.0f;

    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float qi = qv[i];
        float ki = kv[i];
        dot    = fmaf(qi, ki, dot);
        norm_q = fmaf(qi, qi, norm_q);
        norm_k = fmaf(ki, ki, norm_k);
    }

    // Warp-level reduction
    dot    = warp_reduce_sum(dot);
    norm_q = warp_reduce_sum(norm_q);
    norm_k = warp_reduce_sum(norm_k);

    // Block-level reduction via shared memory
    __shared__ float s_dot[WARP_SIZE];
    __shared__ float s_nq [WARP_SIZE];
    __shared__ float s_nk [WARP_SIZE];

    int lane  = threadIdx.x % WARP_SIZE;
    int warpId = threadIdx.x / WARP_SIZE;
    int nWarps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;

    if (lane == 0) {
        s_dot[warpId] = dot;
        s_nq [warpId] = norm_q;
        s_nk [warpId] = norm_k;
    }
    __syncthreads();

    if (threadIdx.x < nWarps) {
        dot    = s_dot[threadIdx.x];
        norm_q = s_nq [threadIdx.x];
        norm_k = s_nk [threadIdx.x];
    } else {
        dot    = 0.0f;
        norm_q = 0.0f;
        norm_k = 0.0f;
    }

    if (warpId == 0) {
        dot    = warp_reduce_sum(dot);
        norm_q = warp_reduce_sum(norm_q);
        norm_k = warp_reduce_sum(norm_k);
    }

    if (threadIdx.x == 0) {
        float denom = sqrtf(norm_q) * sqrtf(norm_k) + SHIP_EPS;
        out[(long)q * n_keys + k] = dot / denom;
    }
}

// ════════════════════════════════════════════════════════════════════
//  cosine_similarity_top_k
//
//  Like cosine_similarity_batched but for each query also writes
//  the top-K key indices (by similarity score) into `top_k_indices`.
//
//  Implementation: stream the similarity row once, keep each thread's
//  local top candidates in registers, then use warp-participating k-way
//  merges to produce the final routing experts.  This keeps shared-memory
//  usage bounded and avoids the single-thread O(N · K) tail.
//
//  Shared memory: static only (~8.3 KB), no dynamic allocation needed.
//  The host should pass 0 for the dynamic shared-memory bytes parameter.
//
//  Params
//    queries         [n_queries × dim]
//    keys            [n_keys   × dim]
//    similarities    [n_queries × n_keys]  — full similarity matrix output
//    top_k_indices   [n_queries × top_k]   — indices of top-k keys per query
//    n_queries, n_keys, dim, top_k
// ════════════════════════════════════════════════════════════════════
extern "C" __global__
void cosine_similarity_top_k(
    const float* __restrict__ queries,
    const float* __restrict__ keys,
    float*       __restrict__ similarities,
    int*         __restrict__ top_k_indices,
    int n_queries,
    int n_keys,
    int dim,
    int top_k)
{
    int q = blockIdx.x;
    if (q >= n_queries) return;

    if (top_k <= 0) return;

    int requested_k = (top_k < n_keys) ? top_k : n_keys;
    int actual_k = (requested_k < MAX_ROUTING_TOP_K) ? requested_k : MAX_ROUTING_TOP_K;
    int* out_idx = top_k_indices + (long)q * top_k;

    if (requested_k <= 0) {
        for (int t = threadIdx.x; t < top_k; t += blockDim.x)
            out_idx[t] = -1;
        return;
    }

    const float* qv = queries + (long)q * dim;
    int lane = threadIdx.x & (WARP_SIZE - 1);
    int warp_id = threadIdx.x / WARP_SIZE;
    int n_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;

    __shared__ float s_query_norm[MAX_BLOCK_WARPS];
    __shared__ float s_inv_query_norm;
    __shared__ float s_warp_scores[MAX_BLOCK_WARPS][MAX_ROUTING_TOP_K];
    __shared__ int s_warp_indices[MAX_BLOCK_WARPS][MAX_ROUTING_TOP_K];

    float q_norm = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float qi = qv[i];
        q_norm = fmaf(qi, qi, q_norm);
    }
    q_norm = warp_reduce_sum(q_norm);

    if (lane == 0)
        s_query_norm[warp_id] = q_norm;
    __syncthreads();

    if (warp_id == 0) {
        float block_q_norm = (lane < n_warps) ? s_query_norm[lane] : 0.0f;
        block_q_norm = warp_reduce_sum(block_q_norm);
        if (lane == 0)
            s_inv_query_norm = 1.0f / sqrtf(fmaxf(block_q_norm, SHIP_EPS));
    }
    __syncthreads();

    float inv_query_norm = s_inv_query_norm;
    float local_scores[MAX_ROUTING_TOP_K];
    int local_indices[MAX_ROUTING_TOP_K];

#pragma unroll
    for (int i = 0; i < MAX_ROUTING_TOP_K; ++i) {
        local_scores[i] = COSINE_SENTINEL;
        local_indices[i] = INT_MAX;
    }

    for (int k = threadIdx.x; k < n_keys; k += blockDim.x) {
        const float* kv = keys + (long)k * dim;
        float dot = 0.0f;
        float norm_k = 0.0f;

        for (int i = 0; i < dim; ++i) {
            float qi = qv[i];
            float ki = kv[i];
            dot = fmaf(qi, ki, dot);
            norm_k = fmaf(ki, ki, norm_k);
        }

        float similarity = dot * inv_query_norm / sqrtf(fmaxf(norm_k, SHIP_EPS));
        similarities[(long)q * n_keys + k] = similarity;
        topk_insert(similarity, k, local_scores, local_indices, actual_k);
    }

    int local_cursor = 0;
    float candidate_score = local_scores[0];
    int candidate_idx = local_indices[0];

    for (int slot = 0; slot < actual_k; ++slot) {
        float best_score = candidate_score;
        int best_idx = candidate_idx;
        int best_lane = lane;
        warp_reduce_best(best_score, best_idx, best_lane);

        if (lane == 0) {
            s_warp_scores[warp_id][slot] = best_score;
            s_warp_indices[warp_id][slot] = (best_score > COSINE_SENTINEL) ? best_idx : -1;
        }

        if (lane == best_lane) {
            ++local_cursor;
            candidate_score = (local_cursor < actual_k) ? local_scores[local_cursor] : COSINE_SENTINEL;
            candidate_idx = (local_cursor < actual_k) ? local_indices[local_cursor] : INT_MAX;
        }
    }
    __syncthreads();

    if (warp_id == 0) {
        int warp_cursor = 0;
        float warp_score = (lane < n_warps) ? s_warp_scores[lane][0] : COSINE_SENTINEL;
        int warp_index = (lane < n_warps) ? s_warp_indices[lane][0] : INT_MAX;

        for (int slot = 0; slot < actual_k; ++slot) {
            float best_score = warp_score;
            int best_idx = warp_index;
            int best_lane = lane;
            warp_reduce_best(best_score, best_idx, best_lane);

            if (lane == 0)
                out_idx[slot] = (best_score > COSINE_SENTINEL) ? best_idx : -1;

            if (lane == best_lane && lane < n_warps) {
                ++warp_cursor;
                warp_score = (warp_cursor < actual_k) ? s_warp_scores[lane][warp_cursor] : COSINE_SENTINEL;
                warp_index = (warp_cursor < actual_k) ? s_warp_indices[lane][warp_cursor] : INT_MAX;
            }
        }
    }

    for (int t = actual_k + threadIdx.x; t < top_k; t += blockDim.x)
        out_idx[t] = -1;
}
