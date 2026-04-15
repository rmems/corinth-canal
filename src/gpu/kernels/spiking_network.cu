// ════════════════════════════════════════════════════════════════════
//  spiking_network.cu — LIF / STDP spiking-network CUDA kernels
//
//  Kernels exported (name-exact for PTX symbol lookup in kernel.rs):
//    poisson_encode       — rate-coded Poisson spike train
//    project_snapshot_current — project telemetry snapshot to per-neuron current
//    lif_step             — LIF neuron step (unweighted)
//    lif_step_weighted    — LIF step with synaptic weight matrix
//    gif_step_weighted    — GIF (Generalized Integrate-and-Fire) step with adaptation,
//                           adaptive threshold, and weighted synapses (matches CPU funnel.rs)
//    spike_rate           — windowed firing-rate estimator
//    reset_membrane       — reset membrane to resting potential
//    stdp_update          — spike-timing-dependent plasticity
//    neuro_bias_logits    — add neuromodulator bias to logit array
//    membrane_dv_dt_reduce_pass1 — per-block reduction of |dv/dt|
//    routing_entropy_reduce_pass1 — per-block reduction of routing entropy
//    latent_reduce_pass2          — final reduction of pass1 partials
//    saaq_find_best_walker        — SAAQ argmax reduction (membrane - scale*adaptation); writes single u32 best_walker
//
//  Parameters follow the 16-neuron / 16-channel architecture in
//  neuro-spike-core/src/snn/engine.rs.
//
//  Target: sm_120 (RTX 5080 Blackwell) · CUDA 13.x
// ════════════════════════════════════════════════════════════════════

#include "common.cuh"

// ── LIF model constants ───────────────────────────────────────────
#define LIF_DECAY        0.85f   // membrane decay per tick (1 - dt/tau_m)
#define LIF_THRESHOLD    1.0f    // normalised firing threshold
#define LIF_RESET        0.0f    // reset potential after spike
#define LIF_REFRACT_TICK 2       // integer ticks of absolute refractory period

// ── GIF model constants (match SparseGifHiddenLayer in funnel.rs) ──
#define GIF_LEAK             0.92f
#define GIF_DRIVE_SCALE      0.75f
#define GIF_THRESHOLD_BASE   0.65f
#define GIF_ADAPTATION_SCALE 0.22f
#define GIF_ADAPTATION_DECAY 0.94f
#define GIF_RESET_RATIO      0.35f
#define GIF_ADAPTATION_TERM  0.05f

// ── STDP constants (match stdp.rs) ───────────────────────────────
#define STDP_A_PLUS   0.01f
#define STDP_A_MINUS  0.012f
#define STDP_TAU      20.0f
#define STDP_W_MIN    0.0f
#define STDP_W_MAX    2.0f

// ════════════════════════════════════════════════════════════════════
//  poisson_encode
//
//  Convert a normalised float stimulus [0,1] into a Poisson spike (0/1)
//  for each input channel.
//
//  Params
//    stimuli   [n_channels] — normalised input rates in [0,1]
//    spikes    [n_channels] — output: 1 if spike fired, 0 otherwise
//    n         — number of channels to process
//    seed      — per-tick random seed (XOR with thread idx for variety)
// ════════════════════════════════════════════════════════════════════
extern "C" __global__
void poisson_encode(
    const float* __restrict__ stimuli,
    unsigned int* __restrict__ spikes,
    int n,
    unsigned int seed)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    unsigned int rng = lcg_next(seed ^ (unsigned int)tid);
    float threshold = stimuli[tid];
    threshold = fmaxf(0.0f, fminf(1.0f, threshold));

    rng = lcg_next(rng);
    float r = lcg_float(rng);
    spikes[tid] = (r < threshold) ? 1u : 0u;
}

// ════════════════════════════════════════════════════════════════════
//  project_snapshot_current
//
//  Lightweight projection kernel that maps a 4-channel telemetry snapshot
//  into per-neuron external current for the LIF step.
//
//  Params
//    snapshot      [4]         — gpu_temp, gpu_power, cpu_temp, cpu_power
//    input_current [n_neurons] — projected per-neuron current (output)
//    n_neurons
// ════════════════════════════════════════════════════════════════════
extern "C" __global__
void project_snapshot_current(
    const float* __restrict__ snapshot,
    float* __restrict__ input_current,
    int n_neurons)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_neurons) return;

    float gpu_temp_c = snapshot[0];
    float gpu_power_w = snapshot[1];
    float cpu_temp_c = snapshot[2];
    float cpu_power_w = snapshot[3];

    float temp_norm = fmaxf(-1.0f, fminf(1.0f, (gpu_temp_c - 60.0f) / 30.0f));
    float gpu_power_norm = fmaxf(-1.0f, fminf(1.0f, (gpu_power_w - 220.0f) / 220.0f));
    float cpu_temp_norm = fmaxf(-1.0f, fminf(1.0f, (cpu_temp_c - 60.0f) / 30.0f));
    float cpu_power_norm = fmaxf(-1.0f, fminf(1.0f, (cpu_power_w - 120.0f) / 120.0f));

    float channel;
    switch (tid & 3) {
        case 0:
            channel = temp_norm;
            break;
        case 1:
            channel = gpu_power_norm;
            break;
        case 2:
            channel = cpu_temp_norm;
            break;
        default:
            channel = cpu_power_norm;
            break;
    }

    unsigned int phase_seed = (unsigned int)tid * 1664525u + 1013904223u;
    float phase = ((phase_seed >> 8) & 1023u) / 1023.0f;
    float jitter = (phase - 0.5f) * 0.25f;

    float drive = 0.9f + channel * 0.45f + jitter;
    input_current[tid] = fmaxf(0.0f, drive);
}

// ════════════════════════════════════════════════════════════════════
//  lif_step
//
//  LIF neuron update without synaptic weights — each neuron receives
//  a pre-summed current `I_ext`.
//
//  Params
//    membrane    [n_neurons]  — membrane potential (read-write)
//    I_ext       [n_neurons]  — external current injection
//    refract     [n_neurons]  — refractory counter (uint, read-write)
//    spikes_out  [n_neurons]  — output spike flags (0/1)
//    n_neurons   — number of LIF neurons
// ════════════════════════════════════════════════════════════════════
extern "C" __global__
void lif_step(
    float* __restrict__        membrane,
    const float* __restrict__  I_ext,
    unsigned int* __restrict__ refract,
    unsigned int* __restrict__ spikes_out,
    int n_neurons)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_neurons) return;

    unsigned int ref = refract[tid];
    float v = membrane[tid];
    unsigned int spike = 0u;

    if (ref > 0u) {
        v = LIF_RESET;
        refract[tid] = ref - 1u;
    } else {
        v = fmaf(LIF_DECAY, v, I_ext[tid]);

        if (v >= LIF_THRESHOLD) {
            spike = 1u;
            v = LIF_RESET;
            refract[tid] = (unsigned int)LIF_REFRACT_TICK;
        }
    }

    membrane[tid]   = v;
    spikes_out[tid] = spike;
}

// ════════════════════════════════════════════════════════════════════
//  lif_step_weighted
//
//  LIF neuron update with a full weight matrix.  Each neuron's input
//  current is the dot product of input_spikes and its weight row.
//
//  Uses shared memory to cache a tile of the weight row.
//
//  Params
//    membrane      [n_neurons]              — read-write
//    weights       [n_neurons × n_inputs]  — row-major weight matrix
//    input_spikes  [n_inputs]               — binary spike inputs (float 0/1)
//    refract       [n_neurons]              — refractory counter
//    spikes_out    [n_neurons]              — output spikes
//    n_neurons, n_inputs
// ════════════════════════════════════════════════════════════════════
extern "C" __global__
void lif_step_weighted(
    float* __restrict__        membrane,
    const float* __restrict__  weights,
    const float* __restrict__  input_spikes,
    unsigned int* __restrict__ refract,
    unsigned int* __restrict__ spikes_out,
    int n_neurons,
    int n_inputs)
{
    extern __shared__ float s_inputs[];

    for (int i = threadIdx.x; i < n_inputs; i += blockDim.x)
        s_inputs[i] = input_spikes[i];
    __syncthreads();

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_neurons) return;

    unsigned int ref = refract[tid];
    float v = membrane[tid];
    unsigned int spike = 0u;

    if (ref > 0u) {
        v = LIF_RESET;
        refract[tid] = ref - 1u;
    } else {
        const float* w_row = weights + (long)tid * n_inputs;
        float I = 0.0f;
        for (int j = 0; j < n_inputs; ++j)
            I = fmaf(w_row[j], s_inputs[j], I);

        v = fmaf(LIF_DECAY, v, I);

        if (v >= LIF_THRESHOLD) {
            spike = 1u;
            v = LIF_RESET;
            refract[tid] = (unsigned int)LIF_REFRACT_TICK;
        }
    }

    membrane[tid]   = v;
    spikes_out[tid] = spike;
}

// ════════════════════════════════════════════════════════════════════
//  gif_step_weighted
//
//  GIF (Generalized Integrate-and-Fire) neuron update with synaptic weights,
//  per-neuron adaptation state, and dynamic threshold. Matches the CPU
//  SparseGifHiddenLayer::run() logic from funnel.rs for consistency with
//  the 2048-neuron SNN and OLMoE projector.
//
//  Adaptation decays every step; threshold = base + adaptation*scale;
//  on spike: adaptation += 1.0, soft-reset membrane.
//
//  Uses shared memory for input_spikes tile. Refractory is respected.
//  For full temporal loop, input_spikes can come from previous spikes_out.
//
//  Params
//    membrane     [n_neurons]              — read-write membrane potential
//    adaptation   [n_neurons]              — read-write GIF adaptation
//    weights      [n_neurons × n_inputs]   — row-major weight matrix
//    input_spikes [n_inputs]               — binary spike inputs (0.0/1.0)
//    refract      [n_neurons]              — refractory counter
//    spikes_out   [n_neurons]              — output spike flags
//    n_neurons, n_inputs
// ════════════════════════════════════════════════════════════════════
extern "C" __global__
void gif_step_weighted(
    float* __restrict__        membrane,
    float* __restrict__        adaptation,
    const float* __restrict__  weights,
    const float* __restrict__  input_spikes,
    unsigned int* __restrict__ refract,
    unsigned int* __restrict__ spikes_out,
    int n_neurons,
    int n_inputs)
{
    extern __shared__ float s_inputs[];

    // Load input spikes into shared memory (same pattern as lif_step_weighted)
    for (int i = threadIdx.x; i < n_inputs; i += blockDim.x)
        s_inputs[i] = input_spikes[i];
    __syncthreads();

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_neurons) return;

    unsigned int ref = refract[tid];
    float v = membrane[tid];
    float a = adaptation[tid];
    unsigned int spike = 0u;

    if (ref > 0u) {
        v = LIF_RESET;
        refract[tid] = ref - 1u;
        adaptation[tid] = a * GIF_ADAPTATION_DECAY;  // decay even during refractory
    } else {
        a *= GIF_ADAPTATION_DECAY;

        const float* w_row = weights + (long)tid * n_inputs;
        float drive = 0.0f;
        for (int j = 0; j < n_inputs; ++j)
            drive = fmaf(w_row[j], s_inputs[j], drive);

        v = v * GIF_LEAK + drive * GIF_DRIVE_SCALE - a * GIF_ADAPTATION_TERM;

        float threshold = GIF_THRESHOLD_BASE + a * GIF_ADAPTATION_SCALE;
        if (v >= threshold) {
            spike = 1u;
            v -= threshold * GIF_RESET_RATIO;
            a += 1.0f;
            refract[tid] = (unsigned int)LIF_REFRACT_TICK;
        }
        adaptation[tid] = a;
    }

    membrane[tid]   = v;
    spikes_out[tid] = spike;
}

// ════════════════════════════════════════════════════════════════════
//  spike_rate
//
//  Estimate windowed firing rate for each neuron by accumulating a
//  decaying trace.
//
//  rate[i] = ALPHA * rate[i] + (1 - ALPHA) * spike[i]
//
//  Params
//    rates       [n_neurons] — exponential rate estimate (read-write)
//    spikes      [n_neurons] — current spike flags (float 0/1)
//    n_neurons
//    alpha       — trace decay coefficient (e.g. 0.95)
// ════════════════════════════════════════════════════════════════════
extern "C" __global__
void spike_rate(
    float* __restrict__       rates,
    const float* __restrict__ spikes,
    int n_neurons,
    float alpha)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_neurons) return;

    float r = rates[tid];
    float s = spikes[tid];
    rates[tid] = fmaf(alpha, r, fmaf(1.0f - alpha, s, 0.0f));
}

// ════════════════════════════════════════════════════════════════════
//  reset_membrane
//
//  Unconditionally reset all membrane potentials to `v_reset`.
//  Used at episode boundaries or full network resets.
//
//  Params
//    membrane   [n_neurons] — membrane array
//    n_neurons
//    v_reset    — target reset value (typically 0.0)
// ════════════════════════════════════════════════════════════════════
extern "C" __global__
void reset_membrane(
    float* __restrict__ membrane,
    int n_neurons,
    float v_reset)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_neurons) return;
    membrane[tid] = v_reset;
}

// ════════════════════════════════════════════════════════════════════
//  stdp_update
//
//  Pairwise STDP weight update:
//    If post spiked:  dw += A+ * exp(-|dt| / tau)  (LTP)
//    If pre  spiked:  dw -= A- * exp(-|dt| / tau)  (LTD)
//
//  Weights are clamped to [W_MIN, W_MAX] after each update.
//
//  Params
//    weights       [n_post × n_pre]  — synaptic weight matrix (row-major)
//    pre_spikes    [n_pre]           — binary pre-synaptic spike flags
//    post_spikes   [n_post]          — binary post-synaptic spike flags
//    pre_traces    [n_pre]           — pre-synaptic eligibility traces
//    post_traces   [n_post]          — post-synaptic eligibility traces
//    n_post, n_pre
//    dt_ms         — time elapsed since last pair event (ms)
// ════════════════════════════════════════════════════════════════════
extern "C" __global__
void stdp_update(
    float* __restrict__       weights,
    const float* __restrict__ pre_spikes,
    const float* __restrict__ post_spikes,
    float* __restrict__       pre_traces,
    float* __restrict__       post_traces,
    int n_post,
    int n_pre,
    float dt_ms)
{
    int post = blockIdx.y * blockDim.y + threadIdx.y;
    int pre  = blockIdx.x * blockDim.x + threadIdx.x;
    if (post >= n_post || pre >= n_pre) return;

    float decay = __expf(-dt_ms / STDP_TAU);

    float pre_tr  = fmaf(decay, pre_traces[pre],  pre_spikes[pre]);
    float post_tr = fmaf(decay, post_traces[post], post_spikes[post]);

    if (threadIdx.y == 0) {
        pre_traces[pre] = pre_tr;
    }
    if (threadIdx.x == 0) {
        post_traces[post] = post_tr;
    }

    long idx = (long)post * n_pre + pre;
    float w = weights[idx];

    float ps = post_spikes[post];
    float prs = pre_spikes[pre];

    w += STDP_A_PLUS  * ps  * pre_tr;
    w -= STDP_A_MINUS * prs * post_tr;

    w = fmaxf(STDP_W_MIN, fminf(STDP_W_MAX, w));
    weights[idx] = w;
}

// ════════════════════════════════════════════════════════════════════
//  neuro_bias_logits
//
//  Add a neuromodulator-scaled bias vector to a logit array.
//  logits[i] += scale * bias[i]
//
//  Used to inject dopamine / cortisol / ACh bias into the readout
//  layer before softmax.
//
//  Params
//    logits   [n]  — float logit array (read-write)
//    bias     [n]  — per-element bias template
//    n
//    scale    — scalar modulator gain
// ════════════════════════════════════════════════════════════════════
extern "C" __global__
void neuro_bias_logits(
    float* __restrict__       logits,
    const float* __restrict__ bias,
    int n,
    float scale)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    logits[tid] = fmaf(scale, bias[tid], logits[tid]);
}

// ════════════════════════════════════════════════════════════════════
//  membrane_dv_dt_reduce_pass1
//
//  Computes block partials for:
//    abs_dv_dt_i = abs(membrane_now[i] - membrane_prev[i]) / dt
//  Emits per-block:
//    partial_sum[blockIdx.x] = sum(abs_dv_dt_i)
//    partial_max[blockIdx.x] = max(abs_dv_dt_i)
//
//  Host performs a second GPU pass (`latent_reduce_pass2`) over partials.
// ════════════════════════════════════════════════════════════════════
extern "C" __global__
void membrane_dv_dt_reduce_pass1(
    const float* __restrict__ membrane_prev,
    const float* __restrict__ membrane_now,
    float* __restrict__ partial_sum,
    float* __restrict__ partial_max,
    int n_neurons,
    float dt)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float inv_dt = 1.0f / fmaxf(fabsf(dt), SHIP_EPS);
    float x = 0.0f;

    if (tid < n_neurons) {
        x = fabsf(membrane_now[tid] - membrane_prev[tid]) * inv_dt;
    }

    float sum_v = warp_reduce_sum(x);
    float max_v = warp_reduce_max(x);

    int lane = threadIdx.x & (WARP_SIZE - 1);
    int warp_id = threadIdx.x / WARP_SIZE;
    int n_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;

    __shared__ float s_sum[32];
    __shared__ float s_max[32];

    if (lane == 0) {
        s_sum[warp_id] = sum_v;
        s_max[warp_id] = max_v;
    }
    __syncthreads();

    if (warp_id == 0) {
        float bsum = (threadIdx.x < n_warps) ? s_sum[lane] : 0.0f;
        float bmax = (threadIdx.x < n_warps) ? s_max[lane] : 0.0f;
        bsum = warp_reduce_sum(bsum);
        bmax = warp_reduce_max(bmax);

        if (threadIdx.x == 0) {
            partial_sum[blockIdx.x] = bsum;
            partial_max[blockIdx.x] = bmax;
        }
    }
}

// ════════════════════════════════════════════════════════════════════
//  routing_entropy_reduce_pass1
//
//  Computes routing entropy per node:
//    H(node) = -sum_j p(node,j) * log2(p(node,j))
//  Expects row-major probabilities:
//    routing_probs [n_nodes x n_routes]
//
//  Emits per-block partial sum/max of H(node).
// ════════════════════════════════════════════════════════════════════
extern "C" __global__
void routing_entropy_reduce_pass1(
    const float* __restrict__ routing_probs,
    float* __restrict__ partial_sum,
    float* __restrict__ partial_max,
    int n_nodes,
    int n_routes)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float entropy = 0.0f;

    if (tid < n_nodes) {
        const float* row = routing_probs + (long)tid * n_routes;
        for (int r = 0; r < n_routes; ++r) {
            float p = fmaxf(row[r], 0.0f);
            if (p > SHIP_EPS) {
                entropy = fmaf(-p, log2f(p), entropy);
            }
        }
    }

    float sum_v = warp_reduce_sum(entropy);
    float max_v = warp_reduce_max(entropy);

    int lane = threadIdx.x & (WARP_SIZE - 1);
    int warp_id = threadIdx.x / WARP_SIZE;
    int n_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;

    __shared__ float s_sum[32];
    __shared__ float s_max[32];

    if (lane == 0) {
        s_sum[warp_id] = sum_v;
        s_max[warp_id] = max_v;
    }
    __syncthreads();

    if (warp_id == 0) {
        float bsum = (threadIdx.x < n_warps) ? s_sum[lane] : 0.0f;
        float bmax = (threadIdx.x < n_warps) ? s_max[lane] : 0.0f;
        bsum = warp_reduce_sum(bsum);
        bmax = warp_reduce_max(bmax);

        if (threadIdx.x == 0) {
            partial_sum[blockIdx.x] = bsum;
            partial_max[blockIdx.x] = bmax;
        }
    }
}

// ════════════════════════════════════════════════════════════════════
//  latent_reduce_pass2
//
//  Final reduction over pass1 partials:
//    out_sum[0] = sum(partial_sum[0..n_partials))
//    out_max[0] = max(partial_max[0..n_partials))
//
//  Intended launch: a single block (e.g. 256 threads).
// ════════════════════════════════════════════════════════════════════
extern "C" __global__
__launch_bounds__(256)
void latent_reduce_pass2(
    const float* __restrict__ partial_sum,
    const float* __restrict__ partial_max,
    float* __restrict__ out_sum,
    float* __restrict__ out_max,
    int n_partials)
{
    int tid = threadIdx.x;
    float local_sum = 0.0f;
    float local_max = 0.0f;

    for (int i = tid; i < n_partials; i += blockDim.x) {
        local_sum += partial_sum[i];
        local_max = fmaxf(local_max, partial_max[i]);
    }

    float sum_v = warp_reduce_sum(local_sum);
    float max_v = warp_reduce_max(local_max);

    int lane = tid & (WARP_SIZE - 1);
    int warp_id = tid / WARP_SIZE;
    int n_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;

    __shared__ float s_sum[32];
    __shared__ float s_max[32];

    if (lane == 0) {
        s_sum[warp_id] = sum_v;
        s_max[warp_id] = max_v;
    }
    __syncthreads();

    if (warp_id == 0) {
        float bsum = (tid < n_warps) ? s_sum[lane] : 0.0f;
        float bmax = (tid < n_warps) ? s_max[lane] : 0.0f;
        bsum = warp_reduce_sum(bsum);
        bmax = warp_reduce_max(bmax);
        if (tid == 0) {
            out_sum[0] = bsum;
            out_max[0] = bmax;
        }
    }
}

// ════════════════════════════════════════════════════════════════════
//  saaq_find_best_walker
//
//  On-device SAAQ (Spiking Activity and Adaptive Quantization) reduction.
//  Computes per-neuron score = membrane[tid] - (adaptation_scale * adaptation[tid])
//  and finds argmax walker (higher score better). Matches the clarified Rust heuristic.
//
//  Launched with <<<8, 256>>> for exact 2048-neuron Blackwell alignment.
//  Uses warp-level max reduction (no atomics, no shared mem beyond warp). Only
//  the winning u32 index is written to output buffer. This eliminates the
//  Rust for-loop over temporal_*_to_vec() in hybrid.rs::forward_gpu_temporal.
//
//  The pipeline now stays entirely in VRAM after initial latent DMA.
//  Rust downloads only this 4-byte result per tick for telemetry/CSV.
//
//  Params
//    membrane          [n_neurons] — GIF membrane potentials
//    adaptation        [n_neurons] — GIF adaptation state
//    best_walker_out   [1]         — output: best walker index (u32)
//    n_neurons
//    adaptation_scale  — e.g. GIF_ADAPTATION_SCALE (0.22f)
// ════════════════════════════════════════════════════════════════════
extern "C" __global__
__launch_bounds__(256)
void saaq_find_best_walker(
    const float* __restrict__ membrane,
    const float* __restrict__ adaptation,
    unsigned int* __restrict__ best_walker_out,
    int n_neurons,
    float adaptation_scale)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    float my_score = -1e30f;  // sentinel for max reduction
    int my_walker = 0;

    if (tid < n_neurons) {
        float score = membrane[tid] - (adaptation_scale * adaptation[tid]);
        my_score = score;
        my_walker = tid;
    }

    // Warp-level max reduction (argmax on score, tie-break by lower walker idx)
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        float other_score = __shfl_down_sync(0xffffffffu, my_score, offset);
        int other_walker = __shfl_down_sync(0xffffffffu, my_walker, offset);
        if (other_score > my_score || (other_score == my_score && other_walker < my_walker)) {
            my_score = other_score;
            my_walker = other_walker;
        }
    }

    // Block-level reduction using shared memory (8 warps max for 256 threads)
    int lane = threadIdx.x & (WARP_SIZE - 1);
    int warp_id = threadIdx.x / WARP_SIZE;
    int n_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;

    __shared__ float s_scores[32];
    __shared__ int s_walkers[32];

    if (lane == 0) {
        s_scores[warp_id] = my_score;
        s_walkers[warp_id] = my_walker;
    }
    __syncthreads();

    if (warp_id == 0) {
        float bscore = (threadIdx.x < n_warps) ? s_scores[lane] : -1e30f;
        int bwalker = (threadIdx.x < n_warps) ? s_walkers[lane] : 0;

        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            float other_score = __shfl_down_sync(0xffffffffu, bscore, offset);
            int other_walker = __shfl_down_sync(0xffffffffu, bwalker, offset);
            if (other_score > bscore || (other_score == bscore && other_walker < bwalker)) {
                bscore = other_score;
                bwalker = other_walker;
            }
        }

        if (threadIdx.x == 0) {
            *best_walker_out = (unsigned int)bwalker;
        }
    }
}
