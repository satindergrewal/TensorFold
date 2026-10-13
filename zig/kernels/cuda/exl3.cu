// EXL3 trellis decode kernels for the Zig-native CUDA engine.
//
// The scalar spec is zig/src/core/exl3_format.zig; dequantize defines the weight as
//   W' = diag(suh) H_k W_q H_n diag(svh)      (H = block-diagonal 128 Hadamard)
// so the projection y = x W' factorizes into three stages around the raw-trellis GEMM:
//   pre : x_tilde = rot128_rows(x .* suh)      (per token, K axis, 128 blocks)
//   gemm: y_pre = x_tilde W_q                  (this file's decode GEMM)
//   post: y = rot128_cols(y_pre) .* svh        (N axis, in place)
// suh/svh are elementwise fp16 sign vectors [k] and [n], exactly as stored.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdint.h>

namespace tf_exl3 {

// --- Codebook constants (exl3_format.zig) ---

constexpr uint32_t McgMul = 0xCBAC1FED;
constexpr uint32_t Mask = 0x8FFF8FFF;
constexpr uint32_t Flip = 0x3B603B60;

// --- Width: per-tensor rate. halves = 2*bits (K4=8, K3=6; 7 is the half rate) ---

struct Exl3Width {
    int halves;

    __device__ __forceinline__ bool is_half() const { return halves & 1; }

    // E(p): where value p's 16-bit window ends in the tile bitstream (exclusive), in bits.
    __device__ __forceinline__ uint32_t stream_end(int p) const {
        if (!is_half()) return (uint32_t)((p + 1) * halves / 2);
        int ka = (halves - 1) / 2;
        return (uint32_t)(((p + 1) * (2 * ka + 1) - ((p + 1) & 1)) / 2);
    }
};

// --- Circular bitstream bit read: MSB-first within little-endian pairs of int16 words ---

__device__ __forceinline__ uint32_t stream_bit(const int16_t* words, uint32_t index) {
    uint32_t w32 = index >> 5;
    uint32_t within = 31 - (index & 31);
    uint16_t lo = (uint16_t)__ldg(words + w32 * 2);
    uint16_t hi = (uint16_t)__ldg(words + w32 * 2 + 1);
    uint32_t word = (uint32_t)lo | ((uint32_t)hi << 16);
    return (word >> within) & 1;
}

// The 16-bit state of stream value p: the window ending at E(p), ring = the tile's bits.

__device__ __forceinline__ uint32_t tile_state(const Exl3Width& w, const int16_t* tile, int p) {
    uint32_t ring = (uint32_t)(256 * w.halves / 2);
    uint32_t end = w.stream_end(p);
    uint32_t start = (end + ring - 16) % ring;
    uint32_t s = 0;
    #pragma unroll
    for (int b = 0; b < 16; ++b) s = (s << 1) | stream_bit(tile, (start + b) % ring);
    return s;
}

// --- MCG codebook: state -> fp16 pair sum (exl3_format.codebookValue) ---

__device__ __forceinline__ float codebook_value_mcg(uint32_t state) {
    uint32_t x = state * McgMul;
    uint32_t masked = (x & Mask) ^ Flip;
    __half2 h2 = *reinterpret_cast<const __half2*>(&masked);
    return __half2float(__low2half(h2)) + __half2float(__high2half(h2));
}

// --- Stream placement -> (row, col) in the 16x16 tile (exl3_format.tilePlacement) ---

struct Placement { uint8_t row, col; };

__device__ __forceinline__ Placement tile_placement(int p) {
    int lane = p >> 3;
    int j = p & 7;
    return {
        (uint8_t)(2 * (lane & 3) + (j & 1) + 8 * ((j >> 1) & 1)),
        (uint8_t)((lane >> 2) + 8 * (j >> 2)),
    };
}

// --- The 128 Hadamard's entry: h(i,j) = (-1)^popcount(i&j) / sqrt(128) ---

__device__ __forceinline__ float had_entry(int i, int j) {
    return (__popc(i & j) & 1) ? -0.08838834764831845f : 0.08838834764831845f;
}

} // namespace tf_exl3

namespace tf_exl3_kernel {

using namespace tf_exl3;

constexpr int TILE = 16;      // the trellis tile
constexpr int ROWS_MAX = 16;  // the staged x rows per block

// --- pre: x_tilde[r, jc] = sum_i had(jc, i) x[r, base+i] suh[base+i], 128 blocks along K ---
// Grid (k/128, m), 128 threads; the row's block stages once in shared memory.

__global__ void exl3_pre_rot(
    const __nv_bfloat16* __restrict__ x,  // [m, k]
    const uint16_t* __restrict__ suh,     // [k] fp16 bits
    __nv_bfloat16* __restrict__ out,      // [m, k]
    int m, int k
) {
    const int r = blockIdx.y;
    const int base = blockIdx.x * 128;
    const int t = threadIdx.x;

    __shared__ float s_v[128];
    float v = 0.0f;
    if (r < m) {
        v = __bfloat162float(x[(size_t)r * k + base + t]);
        v *= __half2float(__ushort_as_half(__ldg(suh + base + t)));
    }
    s_v[t] = v;
    __syncthreads();

    if (r >= m) return;
    float sum = 0.0f;
    #pragma unroll 8
    for (int i = 0; i < 128; ++i) sum += had_entry(t, i) * s_v[i];
    out[(size_t)r * k + base + t] = __float2bfloat16(sum);
}

// --- post: y .^ svh then the rotation, 128 blocks along N, in place ---
// Grid (n/128, m), 128 threads.

__global__ void exl3_post_rot(
    __nv_bfloat16* __restrict__ y,       // [m, n]
    const uint16_t* __restrict__ svh,    // [n] fp16 bits
    int m, int n
) {
    const int r = blockIdx.y;
    const int base = blockIdx.x * 128;
    const int t = threadIdx.x;

    __shared__ float s_v[128];
    float v = 0.0f;
    if (r < m) v = __bfloat162float(y[(size_t)r * n + base + t]);
    s_v[t] = v;
    __syncthreads();

    if (r >= m) return;
    float sum = 0.0f;
    #pragma unroll 8
    for (int i = 0; i < 128; ++i) sum += had_entry(t, i) * s_v[i];
    // the sign is diag on the RIGHT of the rotation: columns of the rotated output
    sum *= __half2float(__ushort_as_half(__ldg(svh + base + t)));
    y[(size_t)r * n + base + t] = __float2bfloat16(sum);
}

// --- The trellis GEMM: y_pre [rows, N] = x_tilde [rows, K] W_q[K, N] ---
// One block per (16-row chunk, N tile); K walked tile by tile; the tile decode is
// exl3_format's state/placement/codebook path verbatim. x_tilde carries suh and the
// K rotation already; svh and the N rotation belong to post.

__global__ void exl3_decode_matmul(
    const __nv_bfloat16* __restrict__ x,   // [rows, K], rotated/signed
    const int16_t* __restrict__ trellis,   // [tiles_k, tiles_n, words]
    __nv_bfloat16* __restrict__ y,         // [rows, N]
    int rows, int k, int n,
    int tiles_k, int tiles_n, int halves
) {
    const int row_base = blockIdx.y * ROWS_MAX;
    const int rows_here = min(ROWS_MAX, rows - row_base);
    const int tn = blockIdx.x;
    const int tid = threadIdx.x;
    const int words = halves * 8;
    const Exl3Width w = { halves };

    __shared__ float s_w[TILE][TILE];     // the decoded tile, fp32 for the product
    __shared__ float s_x[ROWS_MAX][TILE]; // this K tile of every row

    // this thread's output element, the same at every K tile: the accumulator
    // stays fp32 across the whole walk and rounds to bf16 once
    float acc = 0.0f;
    const int my = (int)tid < rows_here * TILE ? (int)tid : -1;
    const int my_r = my >= 0 ? my / TILE : 0;
    const int my_c = my >= 0 ? my % TILE : 0;

    for (int tk = 0; tk < tiles_k; ++tk) {
        const int16_t* tile = trellis + ((size_t)tk * tiles_n + tn) * words;

        // decode: two values per thread at 128 threads
        for (int p = tid; p < 256; p += blockDim.x) {
            Placement pl = tile_placement(p);
            s_w[pl.row][pl.col] = codebook_value_mcg(tile_state(w, tile, p));
        }

        // stage x: the tile's 16 K values of every row; one thread per column
        if (tid < TILE) {
            for (int r = 0; r < rows_here; ++r)
                s_x[r][tid] = __bfloat162float(x[(size_t)(row_base + r) * k + tk * TILE + tid]);
        }
        __syncthreads();

        if (my >= 0) {
            #pragma unroll
            for (int i = 0; i < TILE; ++i) acc += s_x[my_r][i] * s_w[i][my_c];
        }
        __syncthreads();
    }

    if (my >= 0)
        y[(size_t)(row_base + my_r) * n + (size_t)tn * TILE + my_c] = __float2bfloat16(acc);
}

// --- Scalar tile decode for the Zig host tests: decoded[p], stream position p's value ---

__global__ void exl3_verify_decode(
    const int16_t* __restrict__ trellis,
    int halves,
    __nv_bfloat16* __restrict__ decoded  // [256]
) {
    int p = threadIdx.x;
    if (p >= 256) return;
    Exl3Width w = { halves };
    decoded[p] = __float2bfloat16(codebook_value_mcg(tile_state(w, trellis, p)));
}

// --- Grouped expert dispatch: one launch per (slot, tiles_n-chunk) ---
// The per-expert weights sit in a device descriptor table (ptr, suh, svh, dims);
// picks stay on device. grid.x = tiles_n, grid.y = slot (rows==1 serving).

struct Exl3Desc {
    const int16_t* trellis;
    const uint16_t* suh;
    const uint16_t* svh;
    int k, n, tiles_k, tiles_n, halves;
};

extern "C" __global__ void exl3_grouped_pre_rot(
    const __nv_bfloat16* __restrict__ x,  // [1, k] the row
    const Exl3Desc* __restrict__ descs,   // [experts]
    const int* __restrict__ picks,        // [topk]
    __nv_bfloat16* __restrict__ xt,       // [topk, k] per-slot x-tilde
    int topk
) {
    const int slot = blockIdx.y;
    if (picks[slot] < 0) return;
    const Exl3Desc d = descs[picks[slot]];
    const int base = blockIdx.x * 128;
    const int t = threadIdx.x;
    const int k = d.k;
    __shared__ float s_v[128];
    float v = __bfloat162float(x[base + t]);
    v *= __half2float(__ushort_as_half(__ldg((const uint16_t*)d.suh + base + t)));
    s_v[t] = v;
    __syncthreads();
    float sum = 0.0f;
    #pragma unroll 8
    for (int i = 0; i < 128; ++i) sum += had_entry(t, i) * s_v[i];
    xt[(size_t)slot * k + base + t] = __float2bfloat16(sum);
}

extern "C" __global__ void exl3_grouped_post_rot(
    __nv_bfloat16* __restrict__ y,        // [topk, n] in/out
    const Exl3Desc* __restrict__ descs,
    const int* __restrict__ picks,
    int topk
) {
    const int slot = blockIdx.y;
    if (picks[slot] < 0) return;
    const Exl3Desc d = descs[picks[slot]];
    const int base = blockIdx.x * 128;
    const int t = threadIdx.x;
    const int n = d.n;
    __shared__ float s_v[128];
    float v = __bfloat162float(__ldg(y + (size_t)slot * n + base + t));
    v *= __half2float(__ushort_as_half(__ldg((const uint16_t*)d.svh + base + t)));
    s_v[t] = v;
    __syncthreads();
    float sum = 0.0f;
    #pragma unroll 8
    for (int i = 0; i < 128; ++i) sum += had_entry(t, i) * s_v[i];
    y[(size_t)slot * n + base + t] = __float2bfloat16(sum);
}

extern "C" __global__ void exl3_grouped_project(
    const __nv_bfloat16* __restrict__ x,   // [1, k] one row
    const Exl3Desc* __restrict__ descs,    // [experts]
    const int* __restrict__ picks,         // [topk]
    __nv_bfloat16* __restrict__ y,         // [topk, n] slot outputs
    int topk
) {
    const int slot = blockIdx.y;
    const int ex = picks[slot];
    if (ex < 0) return;
    const Exl3Desc d = descs[ex];
    const int tn = blockIdx.x;
    const int tid = threadIdx.x;
    const int words = d.halves * 8;
    const Exl3Width w = { d.halves };

    __shared__ float s_w[TILE][TILE];
    __shared__ float s_x[1][TILE];

    float acc = 0.0f;
    const int my = (int)tid < TILE ? (int)tid : -1;

    for (int tk = 0; tk < d.tiles_k; ++tk) {
        const int16_t* tile = d.trellis + ((size_t)tk * d.tiles_n + tn) * words;
        for (int p = tid; p < 256; p += blockDim.x) {
            Placement pl = tile_placement(p);
            s_w[pl.row][pl.col] = codebook_value_mcg(tile_state(w, tile, p));
        }
        if (tid < TILE) s_x[0][tid] = __bfloat162float(x[(size_t)slot * d.k + tk * TILE + tid]);
        __syncthreads();
        if (my >= 0) {
            #pragma unroll
            for (int i = 0; i < TILE; ++i) acc += s_x[0][i] * s_w[i][my];
        }
        __syncthreads();
    }
    if (my >= 0)
        y[(size_t)slot * d.n + (size_t)tn * TILE + my] = __float2bfloat16(acc);
}

} // namespace tf_exl3_kernel
