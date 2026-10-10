// GLM DSA's indexer: the block scores and the key-list selection.
// score[row][blk] = sum_h w[row,h] * relu(dot(q_i[row,h,:], pool[blk,:])) over
// 64 heads of 128, bf16 dots ascending-dim; blocks valid to (p0+row+1)/4.
// The select: the top-(top) blocks by score, ties to the lower block, written
// in block order, then the partial block's members as singleton keys, -1 fill.
// Afp32 radix would match metal exactly; a insertion scan is simpler and the
// order rule is the same - the parity decides.

#include <cuda_bf16.h>
#include <math.h>
#include <stdint.h>

namespace tf_glm_index {

constexpr int HEADS = 64;
constexpr int DIM = 128;
constexpr int KPOOL = 4;

// grid (blocks-tiles of 128, rows), 256 threads: a block per thread pair? one
// block per thread, dots over 128 dims in fp32 from bf16.
extern "C" __global__ void index_scores(const __nv_bfloat16* __restrict__ q_i, // [rows, q_stride] at the indexer offset
                                        int q_stride,
                                        const float* __restrict__ iw, // [rows, HEADS]
                                        const __nv_bfloat16* __restrict__ pool, // [pool_rows, DIM]
                                        float* __restrict__ scores, // [rows, s_stride]
                                        int s_stride, int rows, int p0) {
    const int row = blockIdx.y;
    const int blk = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    const int blocks = (p0 + row + 1) / KPOOL;
    if (blk >= blocks) return;
    const __nv_bfloat16* qr = q_i + (size_t)row * q_stride;
    const __nv_bfloat16* pr = pool + (size_t)blk * DIM;
    float total = 0.0f;
    for (int h = 0; h < HEADS; ++h) {
        const __nv_bfloat16* qh = qr + (size_t)h * DIM;
        float dot = 0.0f;
        for (int d = 0; d < DIM; ++d) dot = fmaf(__bfloat162float(qh[d]), __bfloat162float(pr[d]), dot);
        total = fmaf(iw[(size_t)row * HEADS + h], fmaxf(dot, 0.0f), total);
    }
    scores[(size_t)row * s_stride + blk] = total;
}

// The selection, one block of 1024 a row (the counts are small: 512+ blocks).
// out[0..take] block ids ascending by id among the chosen; then the tail keys;
// -1 to width. Chosen = top `top` by score (ties lower id) when top < blocks.
extern "C" __global__ void index_select(const float* __restrict__ scores, int s_stride,
                                        int* __restrict__ out, int i_stride,
                                        int rows, int p0, int top, int width) {
    const int row = blockIdx.x;
    const int t = threadIdx.x;
    const int blocks = (p0 + row + 1) / KPOOL;
    const int take = min(top, blocks);
    const float* s = scores + (size_t)row * s_stride;
    int* o = out + (size_t)row * i_stride;

    // threshold by a shared scan of ranks: thread t owns blocks t, t+1024...
    // rank = number of blocks strictly better + equal-with-lower-id: computing
    // per block is O(blocks) each; 512*1024 too big. Counting sort by fp32 key
    // bits, 3 radix passes like metal? For parity the ORDER of equals is id, so:
    // rank(b) = #{c: key(c) > key(b)} + #{c<b: key(c)==key(b)}; chosen iff rank < take.
    // O(blocks^2/1024) per thread chunk - 512^2/1024 = 256 ops, fine.
    for (int b = t; b < blocks; b += blockDim.x) {
        const float kb = s[b];
        const uint32_t ub = __float_as_uint(kb);
        const uint32_t key_b = (ub & 0x80000000u) ? ~ub : (ub | 0x80000000u);
        int better = 0, equal_lower = 0;
        for (int c = 0; c < blocks; ++c) {
            const uint32_t uc = __float_as_uint(s[c]);
            const uint32_t key_c = (uc & 0x80000000u) ? ~uc : (uc | 0x80000000u);
            if (key_c > key_b) ++better;
            else if (key_c == key_b && c < b) ++equal_lower;
        }
        const int rank = better + equal_lower;
        o[b] = rank < take ? b : -2; // -2 = not chosen
    }
    __syncthreads();
    // compaction into a second shared list, then copy back
    __shared__ int chosen[2048];
    __shared__ int cnt;
    if (t == 0) cnt = 0;
    __syncthreads();
    for (int b = t; b < blocks; b += blockDim.x) {
        if (o[b] != -2) {
            const int at = atomicAdd(&cnt, 1);
            chosen[at] = b;
        }
    }
    __syncthreads();
    for (int i = t; i < take; i += blockDim.x) o[i] = chosen[i];
    __syncthreads();
    // the tail: the partial block's members after the chosen blocks
    const int tail = (p0 + row + 1) % KPOOL;
    const int base = take;
    for (int i = base + t; i < width; i += blockDim.x) {
        const int j = i - base;
        o[i] = j < tail ? (p0 + row + 1 - tail + j) : -1;
    }
}

} // namespace tf_glm_index
