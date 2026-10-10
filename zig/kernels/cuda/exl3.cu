// EXL3 trellis decode kernels for the Zig-native CUDA engine.
// Device functions translate exl3_format.zig's scalar reference (the spec) into
// warp-level CUDA: tile state extraction from the bitstream, MCG codebook decode,
// and MMA integration following the affine-4 kernel's launch pattern.

#include <cuda_bf16.h>
#include <stdint.h>

namespace tf_exl3 {

// ─── Codebook constants (from exl3_format.zig) ───

constexpr uint32_t McgMul = 0xCBAC1FED;
constexpr uint32_t Mask = 0x8FFF8FFF;
constexpr uint32_t Flip = 0x3B603B60;
constexpr int hadamard_dim = 128;

// ─── Width: per-tensor bit width ───
// halves = 2×bits (K4=8, K3=6, K3.5=7 — half-bit is mul1 only)

struct Exl3Width {
    int halves;  // 2×bit count
    __device__ __forceinline__ int bits() const { return halves / 2; }
    __device__ __forceinline__ bool is_half() const { return halves & 1; }
    __device__ __forceinline__ int tile_words() const { return halves * 8; }
};

// ─── Stream end positions: E(p) for p=0..255 ───
// Where value p's 16-bit state window ends in the tile's circular bitstream.

__device__ __forceinline__ uint32_t stream_end(const Exl3Width& w, int p) {
    if (!w.is_half()) {
        // whole width: (p+1) * bits
        return (uint32_t)((p + 1) * w.halves / 2);
    } else {
        // half width: alternating KA and KA+1 bits
        int ka = (w.halves - 1) / 2;
        return (uint32_t)(((p + 1) * (2 * ka + 1) - ((p + 1) & 1)) / 2);
    }
}

// ─── Circular bitstream bit read ───
// Bits are MSB-first within little-endian pairs of int16 words.

__device__ __forceinline__ uint32_t stream_bit(const int16_t* words, uint32_t index, uint32_t ring) {
    uint32_t w32 = index >> 5;
    uint32_t within = 31 - (index & 31);
    uint16_t lo = (uint16_t)__ldg(words + w32 * 2);
    uint16_t hi = (uint16_t)__ldg(words + w32 * 2 + 1);
    uint32_t word = (uint32_t)lo | ((uint32_t)hi << 16);
    return (word >> within) & 1;
}

// ─── Tile state extraction: all 256 states from one trellis tile ───
// Each state is a 16-bit window into the circular bitstream.

__device__ void tile_states(const Exl3Width& w, const int16_t* tile, uint32_t* states) {
    uint32_t ring = (uint32_t)(256 * w.halves / 2);
    // ends[p] computed inline to avoid shared memory overhead
    for (int p = 0; p < 256; ++p) {
        uint32_t end = stream_end(w, p);
        uint32_t start = (end + ring - 16) % ring;
        uint32_t s = 0;
        #pragma unroll
        for (int b = 0; b < 16; ++b) {
            s = (s << 1) | stream_bit(tile, (start + b) % ring, ring);
        }
        states[p] = s;
    }
}

// ─── MCG codebook: state → fp16 value ───
// x = state * McgMul; masked = (x & Mask) ^ Flip; result = fp16(lo) + fp16(hi)

__device__ __forceinline__ __nv_bfloat16 codebook_value_mcg(uint32_t state) {
    uint32_t x = state * McgMul;
    uint32_t masked = (x & Mask) ^ Flip;
    // two fp16 halves from the 32-bit masked value
    __half2 h2 = *reinterpret_cast<__half2*>(&masked);
    __half lo = __low2half(h2);
    __half hi = __high2half(h2);
    // sum: one fp32 accumulation, then round to bf16
    float sum = __half2float(lo) + __half2float(hi);
    return __float2bfloat16(sum);
}

// ─── Sign unpacking: su/sv int16 [n/16] → +1/-1 fp16 [n] ───

__device__ __forceinline__ float sign_of(uint16_t packed, int bit) {
    return ((packed >> bit) & 1) ? -1.0f : 1.0f;
}

// ─── MMA placement: stream position → (row, col) in the 16×16 tile ───
// Maps from the codebook's stream order to the tile's memory layout.

struct Placement { uint8_t row, col; };

__device__ __forceinline__ Placement tile_placement(int p) {
    int lane = p >> 3;
    int j = p & 7;
    return {
        (uint8_t)(2 * (lane & 3) + (j & 1) + 8 * ((j >> 1) & 1)),
        (uint8_t)((lane >> 2) + 8 * (j >> 2))
    };
}

} // namespace tf_exl3

// ─── Main decode kernel: x [rows, K] × W_q (trellis) → y [rows, N] ───
// Input x is pre-rotated (H_K applied by caller). Output needs H_N applied by caller.
// Simple two-phase: decode tile to shared memory, then GEMM. Optimization: fuse later.
//
// Block: 128 threads (4 warps), each warp handles a 16×32 output chunk.
// Grid: (tiles_n / 2) blocks. Each block processes 16 K-rows × 64 N-cols.

namespace tf_exl3_kernel {

using namespace tf_exl3;

constexpr int TILE = 16;           // trellis tile is 16×16
constexpr int SMEM_K = TILE;       // 16 K values per tile
constexpr int SMEM_N = TILE * 4;   // 64 N values (4 tiles per warp group)

struct Exl3Tensor {
    // Device pointers to the tensor's data
    const int16_t* trellis;  // [tiles_k, tiles_n, tile_words]
    const int16_t* suh;      // [K/16] packed signs for K axis
    const int16_t* svh;      // [N/16] packed signs for N axis
    int tiles_k, tiles_n;
    int tile_words;
    int k, n;
    int halves;  // 2×bits for this tensor
};

__global__ void exl3_decode_matmul(
    const __nv_bfloat16* __restrict__ x,   // [rows, K] (pre-rotated)
    const int16_t* __restrict__ trellis,   // [tiles_k, tiles_n, tile_words]
    const int16_t* __restrict__ suh,       // [K/16]
    const int16_t* __restrict__ svh,       // [N/16]
    __nv_bfloat16* __restrict__ y,          // [rows, N] (needs post-rotation)
    int rows, int k, int n,
    int tiles_k, int tiles_n, int halves
) {
    const int tile_words = halves * 8;
    
    // Each block handles one 16-column chunk of N
    const int tn = blockIdx.x;  // which N tile
    const int tid = threadIdx.x;
    const int warp = tid / 32;
    const int lane = tid % 32;
    
    // Shared memory for the decoded weight tile and input
    __shared__ __nv_bfloat16 s_w[TILE][TILE];  // decoded weight [16,16]
    __shared__ __nv_bfloat16 s_x[16][TILE];    // input rows [16,16]
    
    // Zero the output accumulator
    __nv_bfloat16 acc[16];  // each thread's partial output for one row
    
    // Loop over K tiles
    for (int tk = 0; tk < tiles_k; ++tk) {
        // Phase 1: Decode the trellis tile into shared memory
        // All 256 values in the tile need to be decoded
        __syncthreads();
        
        const int16_t* tile = trellis + ((size_t)tk * tiles_n + tn) * tile_words;
        
        // Use the width from the kernel parameter
        Exl3Width w = { halves };
        uint32_t ring = (uint32_t)(256 * halves / 2);
        
        for (int p = tid; p < 256; p += 128) {
            uint32_t end = stream_end(w, p);
            uint32_t start = (end + ring - 16) % ring;
            uint32_t s = 0;
            #pragma unroll
            for (int b = 0; b < 16; ++b) {
                s = (s << 1) | stream_bit(tile, (start + b) % ring, ring);
            }
            Placement pl = tile_placement(p);
            __nv_bfloat16 val = codebook_value_mcg(s);
            
            // Apply the N-axis sign (svh) — one sign per 16 columns
            int svh_word = (tn * TILE + pl.col) / 16;
            int svh_bit = (tn * TILE + pl.col) % 16;
            float sv = sign_of((uint16_t)__ldg(svh + svh_word), svh_bit);
            val = __float2bfloat16(__bfloat162float(val) * sv);
            
            s_w[pl.row][pl.col] = val;
        }
        
        // Load the input into shared memory
        for (int r = tid / 16; r < rows; r += 128 / 16) {
            for (int c = tid % 16; c < TILE; c += 16) {
                int kc = tk * TILE + c;
                // Apply the K-axis sign (suh) — one sign per 16 K values
                int suh_word = kc / 16;
                int suh_bit = kc % 16;
                float su = sign_of((uint16_t)__ldg(suh + suh_word), suh_bit);
                float xv = __bfloat162float(x[r * k + kc]) * su;
                s_x[r][c] = __float2bfloat16(xv);
            }
        }
        __syncthreads();
        
        // Phase 2: GEMM (simple dot product per thread)
        // Each thread handles one (row, col) output element
        for (int r = 0; r < rows; ++r) {
            for (int c = tid % TILE; c < TILE; c += 32) {
                float sum = 0.0f;
                for (int kk = 0; kk < TILE; ++kk) {
                    sum += __bfloat162float(s_x[r][kk]) * __bfloat162float(s_w[kk][c]);
                }
                // Accumulate into y (atomic for now, MMA later)
                int nc = tn * TILE + c;
                if (tk == 0) {
                    y[r * n + nc] = __float2bfloat16(sum);
                } else {
                    float prev = __bfloat162float(y[r * n + nc]);
                    y[r * n + nc] = __float2bfloat16(prev + sum);
                }
            }
        }
    }
}

// ─── Scalar verification kernel (for testing against exl3_format.zig) ───

__global__ void exl3_verify_decode(
    const int16_t* __restrict__ trellis,
    int tile_words, int halves,
    __nv_bfloat16* __restrict__ decoded  // [256] — one tile, stream order
) {
    int p = threadIdx.x;
    if (p >= 256) return;
    
    Exl3Width w = { halves };
    uint32_t ring = (uint32_t)(256 * halves / 2);
    uint32_t end = stream_end(w, p);
    uint32_t start = (end + ring - 16) % ring;
    uint32_t s = 0;
    for (int b = 0; b < 16; ++b) {
        s = (s << 1) | stream_bit(trellis, (start + b) % ring, ring);
    }
    decoded[p] = codebook_value_mcg(s);
}

} // namespace tf_exl3_kernel
