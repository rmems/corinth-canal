// ════════════════════════════════════════════════════════════════════
//  satsolver.cu — Parallel WalkSAT / stochastic local-search kernels
//
//  Origin / Attribution
//  ────────────────────
//  Derived from  kernels/dynexsolve.cu  (Neuromorphic SAT-Solver GPU Kernels)
//  which implements a Continuous-Time Dynamical System (CTDS) relaxing toward
//  satisfying assignments of a Boolean formula in CNF (the Dynex / DynexSolve
//  variant of analog-modeled SAT).
//
//  This file replaces the continuous-voltage CTDS dynamics with a discrete
//  stochastic local-search (WalkSAT) strategy while preserving the same
//  kernel signatures, data layout, and multi-walker parallelism pattern
//  introduced in dynexsolve.cu.
//
//  Kernels exported:
//    satsolver_init          — initialise assignment + clause data
//    satsolver_step          — one WalkSAT flip step per walker
//    satsolver_aux_update    — update auxiliary scores (break / make counts)
//    satsolver_check_solution — check which walkers have found a solution
//    satsolver_extract       — copy best-found assignment to output
//    satsolver_best_reduce_pass1 — per-block min(score, walker) reduction
//    satsolver_best_reduce_pass2 — final min reduction (no atomics)
//
//  Data layout
//    assignment  [n_walkers × n_vars]  — binary variable assignment (uint8)
//    clauses     [n_clauses × clause_len] — literal encoding:
//                  literal = 2*var_idx if positive, 2*var_idx+1 if negative
//    sat_flags   [n_walkers × n_clauses] — 1 if clause satisfied, else 0
//    scores      [n_walkers]             — number of unsatisfied clauses
//    best_score  [gridDim.x] or [1]      — block partials or final best
//    best_walker [gridDim.x] or [1]      — block partials or final best
//
//  Target: sm_120 (RTX 5080 Blackwell) · CUDA 13.x
// ════════════════════════════════════════════════════════════════════

#include "common.cuh"

// Walk probability for random flip (WalkSAT noise parameter)
#define WALKSAT_P   0.57f

// Maximum unsatisfied clauses to track per walker
#define MAX_UNSAT_TRACKED 64

// ── Helper: evaluate a single clause ─────────────────────────────
__device__ __forceinline__
int eval_clause(
    const unsigned char* __restrict__ asgn,
    const int*           __restrict__ clause,
    int clause_len)
{
    if (clause_len <= 0) return 0;
    for (int l = 0; l < clause_len; ++l) {
        unsigned int lit = (unsigned int)clause[l];
        unsigned int var = lit >> 1;
        unsigned int neg = lit & 1u;
        unsigned int val = (unsigned int)asgn[var] ^ neg;
        if (val) return 1;
    }
    return 0;
}

__device__ __forceinline__
bool score_pair_better(
    int lhs_score,
    int lhs_walker,
    int rhs_score,
    int rhs_walker)
{
    return lhs_score < rhs_score ||
           (lhs_score == rhs_score && lhs_walker < rhs_walker);
}

__device__ __forceinline__
void warp_reduce_best_pair(
    int& score,
    int& walker)
{
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        int other_score = __shfl_down_sync(0xffffffffu, score, offset);
        int other_walker = __shfl_down_sync(0xffffffffu, walker, offset);
        if (score_pair_better(other_score, other_walker, score, walker)) {
            score = other_score;
            walker = other_walker;
        }
    }
}

// ════════════════════════════════════════════════════════════════════
//  satsolver_init
//
//  Initialise each walker's assignment to a random binary string and
//  compute the initial number of unsatisfied clauses.
//
//  Params
//    assignment  [n_walkers × n_vars]   — output: random assignment
//    scores      [n_walkers]            — output: initial unsat count
//    clauses     [n_clauses × clause_len]
//    n_walkers, n_vars, n_clauses, clause_len
//    seed        — global random seed
// ════════════════════════════════════════════════════════════════════
extern "C" __global__
void satsolver_init(
    unsigned char* __restrict__ assignment,
    int*           __restrict__ scores,
    const int*     __restrict__ clauses,
    int n_walkers,
    int n_vars,
    int n_clauses,
    int clause_len,
    unsigned int seed)
{
    int walker = blockIdx.x * blockDim.x + threadIdx.x;
    if (walker >= n_walkers) return;

    unsigned char* asgn = assignment + (long)walker * n_vars;
    unsigned int rng    = lcg_next(seed ^ ((unsigned int)walker * 2654435761u));

    for (int v = 0; v < n_vars; ++v) {
        rng = lcg_next(rng);
        asgn[v] = (unsigned char)((rng >> 16) & 1u);
    }

    int unsat = 0;
    for (int c = 0; c < n_clauses; ++c) {
        if (!eval_clause(asgn, clauses + (long)c * clause_len, clause_len))
            ++unsat;
    }
    scores[walker] = unsat;
}

// ════════════════════════════════════════════════════════════════════
//  satsolver_step
//
//  One WalkSAT iteration per walker:
//    1. Pick a random unsatisfied clause.
//    2. With probability p, flip a random variable in that clause.
//    3. Otherwise flip the variable with the lowest break count.
//
//  Params
//    assignment  [n_walkers × n_vars]
//    scores      [n_walkers]            — updated in-place
//    clauses     [n_clauses × clause_len]
//    n_walkers, n_vars, n_clauses, clause_len
//    seed        — per-step seed (combine with step counter at call site)
// ════════════════════════════════════════════════════════════════════
extern "C" __global__
void satsolver_step(
    unsigned char* __restrict__ assignment,
    int*           __restrict__ scores,
    const int*     __restrict__ clauses,
    int n_walkers,
    int n_vars,
    int n_clauses,
    int clause_len,
    unsigned int seed)
{
    int walker = blockIdx.x * blockDim.x + threadIdx.x;
    if (walker >= n_walkers) return;

    unsigned int rng    = lcg_next(seed ^ ((unsigned int)walker * 2246822519u));
    unsigned char* asgn = assignment + (long)walker * n_vars;

    int unsat_buf[MAX_UNSAT_TRACKED];
    int unsat_cnt = 0;
    for (int c = 0; c < n_clauses && unsat_cnt < MAX_UNSAT_TRACKED; ++c) {
        if (!eval_clause(asgn, clauses + (long)c * clause_len, clause_len))
            unsat_buf[unsat_cnt++] = c;
    }

    if (unsat_cnt == 0) {
        scores[walker] = 0;
        return;
    }

    rng = lcg_next(rng);
    int chosen_c = unsat_buf[(rng >> 8) % (unsigned int)unsat_cnt];
    const int* cptr = clauses + (long)chosen_c * clause_len;

    rng = lcg_next(rng);
    float r = lcg_float(rng);

    int flip_var = -1;

    if (r < WALKSAT_P) {
        rng = lcg_next(rng);
        int lit_idx = (int)((rng >> 8) % (unsigned int)clause_len);
        flip_var = (int)((unsigned int)cptr[lit_idx] >> 1);
    } else {
        int best_break = n_clauses + 1;
        for (int l = 0; l < clause_len; ++l) {
            int var = (int)((unsigned int)cptr[l] >> 1);

            int brk = 0;
            asgn[var] ^= 1u;
            for (int c2 = 0; c2 < n_clauses; ++c2) {
                if (!eval_clause(asgn, clauses + (long)c2 * clause_len, clause_len)) {
                    asgn[var] ^= 1u;
                    int was_sat = eval_clause(asgn, clauses + (long)c2 * clause_len, clause_len);
                    asgn[var] ^= 1u;
                    brk += was_sat;
                }
            }
            asgn[var] ^= 1u;

            if (brk < best_break) {
                best_break = brk;
                flip_var   = var;
            }
        }
    }

    if (flip_var >= 0) asgn[flip_var] ^= 1u;

    int unsat = 0;
    for (int c = 0; c < n_clauses; ++c) {
        if (!eval_clause(asgn, clauses + (long)c * clause_len, clause_len))
            ++unsat;
    }
    scores[walker] = unsat;
}

// ════════════════════════════════════════════════════════════════════
//  satsolver_aux_update
//
//  Recompute per-clause satisfaction flags and emit one reduced
//  `(score, walker)` pair per block into `best_score` / `best_walker`.
//  The host can feed those block partials into `satsolver_best_reduce_pass2`
//  to obtain the final global minimum without any atomics.
//
//  Params
//    assignment  [n_walkers × n_vars]
//    sat_flags   [n_walkers × n_clauses]  — output
//    scores      [n_walkers]
//    best_score  [gridDim.x]  — per-block best score output
//    best_walker [gridDim.x]  — per-block best walker output
//    n_walkers, n_vars, n_clauses, clause_len
//    clauses     [n_clauses × clause_len]
// ════════════════════════════════════════════════════════════════════
extern "C" __global__
void satsolver_aux_update(
    const unsigned char* __restrict__ assignment,
    unsigned char*       __restrict__ sat_flags,
    const int*           __restrict__ scores,
    int*                 __restrict__ best_score,
    int*                 __restrict__ best_walker,
    const int*           __restrict__ clauses,
    int n_walkers,
    int n_vars,
    int n_clauses,
    int clause_len)
{
    int walker = blockIdx.x * blockDim.x + threadIdx.x;
    int my_score = INT_MAX;
    int my_walker = INT_MAX;

    if (walker < n_walkers) {
        const unsigned char* asgn = assignment + (long)walker * n_vars;
        unsigned char* flags = sat_flags + (long)walker * n_clauses;

        for (int c = 0; c < n_clauses; ++c)
            flags[c] = (unsigned char)eval_clause(asgn, clauses + (long)c * clause_len, clause_len);

        my_score = scores[walker];
        my_walker = walker;
    }

    int lane = threadIdx.x & (WARP_SIZE - 1);
    int warp_id = threadIdx.x / WARP_SIZE;
    int n_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;

    warp_reduce_best_pair(my_score, my_walker);

    __shared__ int s_score[32];
    __shared__ int s_walker[32];

    if (lane == 0) {
        s_score[warp_id] = my_score;
        s_walker[warp_id] = my_walker;
    }
    __syncthreads();

    if (warp_id == 0) {
        int block_score = (lane < n_warps) ? s_score[lane] : INT_MAX;
        int block_walker = (lane < n_warps) ? s_walker[lane] : INT_MAX;
        warp_reduce_best_pair(block_score, block_walker);

        if (lane == 0) {
            best_score[blockIdx.x] = block_score;
            best_walker[blockIdx.x] = block_walker;
        }
    }
}

// ════════════════════════════════════════════════════════════════════
//  satsolver_check_solution
//
//  Set solved[walker] = 1 if the walker has found a satisfying
//  assignment (scores[walker] == 0), else 0.
//
//  Params
//    scores     [n_walkers]
//    solved     [n_walkers]   — output: 1 if satisfied
//    n_walkers
// ════════════════════════════════════════════════════════════════════
extern "C" __global__
void satsolver_check_solution(
    const int*    __restrict__ scores,
    unsigned char* __restrict__ solved,
    int n_walkers)
{
    int walker = blockIdx.x * blockDim.x + threadIdx.x;
    if (walker >= n_walkers) return;
    solved[walker] = (unsigned char)((scores[walker] == 0) ? 1u : 0u);
}

// ════════════════════════════════════════════════════════════════════
//  satsolver_extract
//
//  Copy the assignment of the best walker to the output buffer.
//  If best_walker[0] is out of range, walker 0 is used as a safe fallback.
//
//  Params
//    assignment   [n_walkers × n_vars]
//    best_walker  [1]   — index of the walker to extract
//    output       [n_vars]  — destination assignment
//    n_vars
//    n_walkers    — total number of walkers (used for bounds check)
// ════════════════════════════════════════════════════════════════════
extern "C" __global__
void satsolver_extract(
    const unsigned char* __restrict__ assignment,
    const int*           __restrict__ best_walker,
    unsigned char*       __restrict__ output,
    int n_vars,
    int n_walkers)
{
    int var = blockIdx.x * blockDim.x + threadIdx.x;
    if (var >= n_vars) return;

    int bw = *best_walker;
    if (bw < 0 || bw >= n_walkers) bw = 0;
    output[var] = assignment[(long)bw * n_vars + var];
}

// ════════════════════════════════════════════════════════════════════
//  satsolver_best_reduce_pass1
//
//  Reduce `scores` to per-block minima with corresponding walker indices.
//  Tie-breaker: lower walker index wins.
//
//  Params
//    scores          [n_walkers]
//    partial_scores  [gridDim.x]
//    partial_walkers [gridDim.x]
//    n_walkers
// ════════════════════════════════════════════════════════════════════
extern "C" __global__
void satsolver_best_reduce_pass1(
    const int* __restrict__ scores,
    int* __restrict__ partial_scores,
    int* __restrict__ partial_walkers,
    int n_walkers)
{
    long walker = (long)blockIdx.x * blockDim.x + threadIdx.x;
    int my_score = (walker < n_walkers) ? scores[walker] : INT_MAX;
    int my_walker = (walker < n_walkers) ? (int)walker : INT_MAX;

    int lane = threadIdx.x & (WARP_SIZE - 1);
    int warp_id = threadIdx.x / WARP_SIZE;
    int n_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;

    warp_reduce_best_pair(my_score, my_walker);

    __shared__ int s_score[32];
    __shared__ int s_walker[32];

    if (lane == 0) {
        s_score[warp_id] = my_score;
        s_walker[warp_id] = my_walker;
    }
    __syncthreads();

    if (warp_id == 0) {
        int block_score = (threadIdx.x < n_warps) ? s_score[lane] : INT_MAX;
        int block_walker = (threadIdx.x < n_warps) ? s_walker[lane] : INT_MAX;
        warp_reduce_best_pair(block_score, block_walker);

        if (threadIdx.x == 0) {
            partial_scores[blockIdx.x] = block_score;
            partial_walkers[blockIdx.x] = block_walker;
        }
    }
}

// ════════════════════════════════════════════════════════════════════
//  satsolver_best_reduce_pass2
//
//  Final reduction over partial block minima from `satsolver_best_reduce_pass1`
//  or `satsolver_aux_update`.
//  Intended launch: one block.
//
//  Params
//    partial_scores  [n_partials]
//    partial_walkers [n_partials]
//    best_score      [1]
//    best_walker     [1]
//    n_partials
// ════════════════════════════════════════════════════════════════════
extern "C" __global__
__launch_bounds__(256)
void satsolver_best_reduce_pass2(
    const int* __restrict__ partial_scores,
    const int* __restrict__ partial_walkers,
    int* __restrict__ best_score,
    int* __restrict__ best_walker,
    int n_partials)
{
    int tid = threadIdx.x;
    int my_score = INT_MAX;
    int my_walker = INT_MAX;

    for (int i = tid; i < n_partials; i += blockDim.x) {
        int s = partial_scores[i];
        int w = partial_walkers[i];
        if (s < my_score || (s == my_score && w < my_walker)) {
            my_score = s;
            my_walker = w;
        }
    }

    int lane = tid & (WARP_SIZE - 1);
    int warp_id = tid / WARP_SIZE;
    int n_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;

    warp_reduce_best_pair(my_score, my_walker);

    __shared__ int s_score[32];
    __shared__ int s_walker[32];

    if (lane == 0) {
        s_score[warp_id] = my_score;
        s_walker[warp_id] = my_walker;
    }
    __syncthreads();

    if (warp_id == 0) {
        int block_score = (tid < n_warps) ? s_score[lane] : INT_MAX;
        int block_walker = (tid < n_warps) ? s_walker[lane] : INT_MAX;
        warp_reduce_best_pair(block_score, block_walker);

        if (tid == 0) {
            best_score[0] = block_score;
            best_walker[0] = block_walker;
        }
    }
}
