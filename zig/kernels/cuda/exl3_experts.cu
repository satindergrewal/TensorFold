// EXL3 routed experts on CUDA, on grouped.zig's plan: each row's arithmetic is exl3.cu's, with the rotation inline.
#include "exl3.cuh"
#include "experts.cuh"

// A layer's experts: `mats` matrices an expert (gate and up, or down), each trellis `stride` bytes and k + n halves.
struct TfExl3Experts {
  unsigned k, n, bits, tiles_n, mats, epi, book, pad;
  unsigned long long stride;
};

// MIXED-WIDTH (our fork): per-matrix descriptor; one per expert-matrix, in
// expert-major order. Full device pointers - no stacks, no repack: each
// expert's trellis and halves live where the loader put them.
struct TfExl3Mat {
  const unsigned char* trellis;
  const __half* suh;
  const __half* svh;
  unsigned int k, n, tiles_n, bits;
};

// A warp's two column sums for each of R rows into zs, after its last input block (lanes 4 c + q hold rows 2 q ...).
template <int R>
__device__ __forceinline__ void tf_exl3x_fold(float (&total)[2][R], unsigned sg, unsigned lane, float* zs) {
#pragma unroll
  for (int i = 0; i < R; ++i) {
    for (int c = 0; c < 2; ++c) {
      float v = total[c][i];
      v = __fadd_rn(v, __shfl_xor_sync(TF_FULL, v, 1));
      v = __fadd_rn(v, __shfl_xor_sync(TF_FULL, v, 2));
      if (lane % 4 == 0) zs[i * 128 + 16 * sg + lane / 4 + 8 * c] = v;
    }
  }
}

// Few rows: the shared register path (exl3.cuh), the shared-memory path's sums without its barriers.
template <int K, int R>
__device__ __forceinline__ void tf_exl3x_free(const __nv_bfloat16* X, int x_stride, const unsigned (&xr)[16],
                                              unsigned rows, const uint8_t* T, const __half* suh, const TfExl3Experts g,
                                              unsigned cb, unsigned sg, unsigned lane, float* zs) {
  constexpr unsigned TB = 32 * K;
  const __nv_bfloat16* xrow[R];
#pragma unroll
  for (int i = 0; i < R; ++i) xrow[i] = X + (uint64_t)xr[i] * x_stride;
  float total[2][R];
  // the hot decode loop keeps its codebook a constant: one switch a call
  if (g.book == 2) tf_exl3_free<K, 2, R>(xrow, rows, suh, T + (uint64_t)(8 * cb + sg) * TB, (uint64_t)g.tiles_n * TB, 0, g.k / 128, lane, total);
  else if (g.book == 1) tf_exl3_free<K, 1, R>(xrow, rows, suh, T + (uint64_t)(8 * cb + sg) * TB, (uint64_t)g.tiles_n * TB, 0, g.k / 128, lane, total);
  else tf_exl3_free<K, 0, R>(xrow, rows, suh, T + (uint64_t)(8 * cb + sg) * TB, (uint64_t)g.tiles_n * TB, 0, g.k / 128, lane, total);
  tf_exl3x_fold<R>(total, sg, lane, zs);
}

// More rows: warps rotate the block's rows into shared memory once, then every warp reads them there.
template <int K, int R>
__device__ __forceinline__ void tf_exl3x_shared(const __nv_bfloat16* X, int x_stride, const unsigned (&xr)[16],
                                                unsigned rows, const uint8_t* T, const __half* suh, const TfExl3Experts g,
                                                unsigned cb, unsigned sg, unsigned lane, float* xs, float* zs) {
  constexpr unsigned TB = 32 * K;
  const unsigned r0 = 2 * (lane % 4), kb = g.k / 128;
  const uint64_t row_bytes = (uint64_t)g.tiles_n * TB;
  const uint8_t* col = T + (uint64_t)(8 * cb + sg) * TB;
  float total[2][R], part[2][R];
#pragma unroll
  for (int i = 0; i < R; ++i) total[0][i] = total[1][i] = 0.0f;
  uint4 next = tf_exl3_words<K>(reinterpret_cast<const uint32_t*>(col), lane);
  for (unsigned ib = 0; ib < kb; ++ib) {
    for (unsigned i = sg; i < R; i += 8) {
      float v[4];
      tf_exl3_rotated(X + (uint64_t)xr[i] * x_stride + ib * 128, suh + ib * 128, i < rows, lane, v);
      for (int q = 0; q < 4; ++q) xs[i * 128 + lane + 32 * q] = v[q];
    }
    __syncthreads();
#pragma unroll
    for (int i = 0; i < R; ++i) part[0][i] = part[1][i] = 0.0f;
    for (unsigned r = 0; r < 8; ++r) {
      const uint4 now = next;
      next = tf_exl3_words<K>(reinterpret_cast<const uint32_t*>(col + (uint64_t)(r < 7 ? 8 * ib + r + 1 : (ib + 1 < kb ? 8 * ib + 8 : 8 * ib + 7)) * row_bytes), lane);
      __half w[8];
      tf_exl3_unpack_any<K>(now, lane, g.book, w);
      tf_exl3_rows<R>(w, xs, r, r0, part);
    }
#pragma unroll
    for (int i = 0; i < R; ++i) {
      total[0][i] = __fadd_rn(total[0][i], part[0][i]);
      total[1][i] = __fadd_rn(total[1][i], part[1][i]);
    }
    __syncthreads();
  }
  tf_exl3x_fold<R>(total, sg, lane, zs);
}

// A row's 128 columns of one matrix out of zs: the transform, then svh times 2^-7, one rounding.
__device__ __forceinline__ void tf_exl3x_out(const float* zs, unsigned i, const __half* svh, unsigned cb, unsigned lane,
                                             float (&v)[4]) {
  for (int q = 0; q < 4; ++q) v[q] = zs[i * 128 + lane + 32 * q];
  tf_exl3_fwht(v, lane);
  for (int q = 0; q < 4; ++q) v[q] = __fmul_rn(v[q], __fmul_rn(__half2float(svh[cb * 128 + lane + 32 * q]), 0.0078125f));
}

// Units (item, 128 columns), 16 pairs at a time; epi 0 fp32, 2 SwiGLU bf16, 3 bf16 (experts.cuh's numbers).
template <int K>
__device__ __forceinline__ void tf_exl3x_units(const __nv_bfloat16* X, int x_stride, int slots, const uint8_t* W,
                                               const __half* H, const int* items, const int* counts, const int* members,
                                               void* out, float limit, int skip, const TfExl3Experts g, float* xs,
                                               float* zs, const TfExl3Mat* __restrict__ mats_desc = nullptr) {
  const unsigned tid = threadIdx.x, sg = tid / 32, lane = tid % 32, nb = g.n / 128; // g.n = n_max; narrow experts skip below
  const unsigned units = (unsigned)__ldg(counts) * nb;
  __shared__ unsigned xr[16], pr[16];
  for (unsigned unit = blockIdx.x; unit < units; unit += gridDim.x) {
    const unsigned it = unit / nb, cb = unit % nb;
    const int e = __ldg(items + 3 * it), first = __ldg(items + 3 * it + 1), cnt = __ldg(items + 3 * it + 2);
    if (e == skip) continue;
    const TfExl3Mat* md = mats_desc + (size_t)e * g.mats; // MIXED: per-expert geometry
    for (int r0 = 0; r0 < cnt; r0 += 16) {
      const unsigned rows = cnt - r0 < 16 ? cnt - r0 : 16;
      if (tid < 16) {
        const int p = tid < rows ? __ldg(members + first + r0 + tid) : 0;
        pr[tid] = p;
        xr[tid] = slots ? p / slots : p;
      }
      __syncthreads();
      unsigned rx[16];
#pragma unroll
      for (int i = 0; i < 16; ++i) rx[i] = xr[i];
      // the capacity the chunk's rows need: a row's sums are the same at every capacity and on either path
      for (unsigned m = 0; m < g.mats; ++m) {
        const uint8_t* t = reinterpret_cast<const uint8_t*>(md[m].trellis);
        const __half* suh = md[m].suh;
        float* z = zs + m * 2048;
        if (cb * 128u >= md[m].n) continue; // MIXED: a narrow expert's padded column block
        if (md[m].bits != 3 && md[m].bits != 4) return; // trap: bad desc
        // MIXED: the per-expert bits pick the specialization (our checkpoint: 6 or 8)
        #define TF_X3_RUN(KK) do { \
          TfExl3Experts lg = g; lg.bits = md[m].bits; lg.tiles_n = md[m].tiles_n; lg.k = md[m].k; lg.n = md[m].n; \
          if (rows <= 1) tf_exl3x_free<KK, 1>(X, x_stride, rx, rows, t, suh, lg, cb, sg, lane, z); \
          else if (rows <= 2) tf_exl3x_free<KK, 2>(X, x_stride, rx, rows, t, suh, lg, cb, sg, lane, z); \
          else tf_exl3x_shared<KK, 16>(X, x_stride, rx, rows, t, suh, lg, cb, sg, lane, xs, z); \
        } while (0)
        switch (md[m].bits) {
          case 3: TF_X3_RUN(3); break; // halves 6 (our K3)
          case 4: TF_X3_RUN(4); break; // halves 8 (our K4)
          default: TF_X3_RUN(K); break; // uniform fallback (their behavior)
        }
        #undef TF_X3_RUN
      }
      __syncthreads();
      for (unsigned i = sg; i < rows; i += 8) {
        float a[4], b[4];
        tf_exl3x_out(zs, i, md[0].svh, cb, lane, a);
        if (g.epi == 2) tf_exl3x_out(zs + 2048, i, md[1].svh, cb, lane, b);
        for (int q = 0; q < 4; ++q) {
          const uint64_t o = (uint64_t)pr[i] * g.n + cb * 128 + lane + 32 * q;
          if (g.epi == 0) reinterpret_cast<float*>(out)[o] = a[q];
          else reinterpret_cast<__nv_bfloat16*>(out)[o] = __float2bfloat16_rn(g.epi == 2 ? swiglu(a[q], b[q], limit) : a[q]);
        }
      }
      __syncthreads();
    }
  }
}

__device__ __forceinline__ void tf_exl3x_width(const __nv_bfloat16* X, int x_stride, int slots, const uint8_t* W,
                                               const __half* H, const int* items, const int* counts, const int* members,
                                               void* out, float limit, int skip, const TfExl3Experts g, float* xs,
                                               float* zs, const TfExl3Mat* __restrict__ mats_desc = nullptr) {
  switch (g.bits) {
    case 1: tf_exl3x_units<1>(X, x_stride, slots, W, H, items, counts, members, out, limit, skip, g, xs, zs, mats_desc); break;
    case 2: tf_exl3x_units<2>(X, x_stride, slots, W, H, items, counts, members, out, limit, skip, g, xs, zs, mats_desc); break;
    case 3: tf_exl3x_units<3>(X, x_stride, slots, W, H, items, counts, members, out, limit, skip, g, xs, zs, mats_desc); break;
    case 4: tf_exl3x_units<4>(X, x_stride, slots, W, H, items, counts, members, out, limit, skip, g, xs, zs, mats_desc); break;
    case 5: tf_exl3x_units<5>(X, x_stride, slots, W, H, items, counts, members, out, limit, skip, g, xs, zs, mats_desc); break;
    case 6: tf_exl3x_units<6>(X, x_stride, slots, W, H, items, counts, members, out, limit, skip, g, xs, zs, mats_desc); break;
    case 7: tf_exl3x_units<7>(X, x_stride, slots, W, H, items, counts, members, out, limit, skip, g, xs, zs, mats_desc); break;
    case 8: tf_exl3x_units<8>(X, x_stride, slots, W, H, items, counts, members, out, limit, skip, g, xs, zs, mats_desc); break;
    default: break;
  }
}

// One kernel for every width and codebook; the layer's geometry says which, the epilogue and the matrices.
extern "C" __global__ void __launch_bounds__(256) tf_exl3_experts(const __nv_bfloat16* X, int x_stride, int slots,
                                                                 const uint8_t* W, const __half* H, const int* items,
                                                                 const int* counts, const int* members, void* out,
                                                                 float limit, int skip, const TfExl3Experts g,
                                                                 const TfExl3Mat* __restrict__ mats_desc) {
  __shared__ __align__(16) float xs[16 * 128], zs[2 * 16 * 128];
  // MIXED: bypass the per-launch width switch; units handles per-expert bits
  tf_exl3x_units<8>(X, x_stride, slots, W, H, items, counts, members, out, limit, skip, g, xs, zs, mats_desc);
}

// Prompt rows in exl3.cu's prompt arithmetic: an item's 64 pairs a block, rows rotated inline, gate and up together.
template <int K, int M>
__device__ __forceinline__ void tf_exl3x_prompt_unit(const __nv_bfloat16* X, int x_stride, const unsigned* xr,
                                                     const unsigned* pr, unsigned rows, const uint8_t* W, const __half* H,
                                                     void* out, float limit, unsigned e, unsigned cb,
                                                     const TfExl3Experts g, float* buf) {
  constexpr unsigned TB = 32 * K;
  const unsigned tid = threadIdx.x, sg = tid / 32, lane = tid % 32, g8 = lane / 4, t4 = lane % 4, kb = g.k / 128;
  const unsigned tiles = (rows + 15) / 16;
  __half* xs = reinterpret_cast<__half*>(buf);  // [M][64 rows][128] during the sums, buf [64][128] fp32 after them
  const uint64_t row_bytes = (uint64_t)g.tiles_n * TB;
  const uint8_t* col[M];
  const __half* suh[M];
  uint4 next[M];
  float acc[M][4][2][4];
#pragma unroll
  for (int m = 0; m < M; ++m) {
    const uint64_t at = (uint64_t)e * g.mats + m;
    col[m] = W + at * g.stride + (uint64_t)(8 * cb + sg) * TB;
    suh[m] = H + at * (g.k + g.n);
    next[m] = tf_exl3_words<K>(reinterpret_cast<const uint32_t*>(col[m]), lane);
    for (int i = 0; i < 4; ++i)
      for (int n = 0; n < 2; ++n) acc[m][i][n][0] = acc[m][i][n][1] = acc[m][i][n][2] = acc[m][i][n][3] = 0.0f;
  }
  for (unsigned ib = 0; ib < kb; ++ib) {
#pragma unroll
    for (int m = 0; m < M; ++m)
      for (unsigned i = sg; i < 64; i += 8) {
        float v[4];
        for (int q = 0; q < 4; ++q) {
          const unsigned c = ib * 128 + lane + 32 * q;
          v[q] = i < rows ? __fmul_rn(__bfloat162float(X[(uint64_t)xr[i] * x_stride + c]), __half2float(suh[m][c])) : 0.0f;
        }
        tf_exl3_fwht(v, lane);
        for (int q = 0; q < 4; ++q) xs[(m * 64 + i) * TF_EXL3_XS + lane + 32 * q] = __float2half_rn(__fmul_rn(v[q], 0.0625f));
      }
    __syncthreads();
    for (unsigned r = 0; r < 8; ++r) {
#pragma unroll
      for (int m = 0; m < M; ++m) {
        const uint4 now = next[m];
        next[m] = tf_exl3_words<K>(reinterpret_cast<const uint32_t*>(col[m] + (uint64_t)(r < 7 ? 8 * ib + r + 1 : (ib + 1 < kb ? 8 * ib + 8 : 8 * ib + 7)) * row_bytes), lane);
        __half w[8];
        tf_exl3_unpack_any<K>(now, lane, g.book, w);
        const uint32_t b[2][2] = {{tf_exl3_pair(w[0], w[1]), tf_exl3_pair(w[2], w[3])}, {tf_exl3_pair(w[4], w[5]), tf_exl3_pair(w[6], w[7])}};
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          if (i >= tiles) break;
          const __half* x = xs + (m * 64 + 16 * i + g8) * TF_EXL3_XS + 16 * r + 2 * t4;
          const uint32_t a[4] = {*reinterpret_cast<const uint32_t*>(x), *reinterpret_cast<const uint32_t*>(x + 8 * TF_EXL3_XS),
                                 *reinterpret_cast<const uint32_t*>(x + 8), *reinterpret_cast<const uint32_t*>(x + 8 * TF_EXL3_XS + 8)};
          tf_exl3_mma(acc[m][i][0], a, b[0][0], b[0][1]);
          tf_exl3_mma(acc[m][i][1], a, b[1][0], b[1][1]);
        }
      }
    }
    __syncthreads();
  }
  // each matrix in turn: its sums into buf, a warp a row through the transform and svh times 2^-3
  float first[8][4];
#pragma unroll
  for (int m = 0; m < M; ++m) {
#pragma unroll
    for (int i = 0; i < 4; ++i)
      for (int n = 0; n < 2; ++n)
        for (int h = 0; h < 2; ++h) {
          const unsigned c = 16 * sg + 8 * n + 2 * t4, row = 16 * i + g8 + 8 * h;
          buf[row * 128 + c] = acc[m][i][n][2 * h];
          buf[row * 128 + c + 1] = acc[m][i][n][2 * h + 1];
        }
    __syncthreads();
    const __half* svh = H + ((uint64_t)e * g.mats + m) * (g.k + g.n) + g.k;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const unsigned i = sg + 8 * j;
      if (i >= rows) break;
      float v[4];
      for (int q = 0; q < 4; ++q) v[q] = buf[i * 128 + lane + 32 * q];
      tf_exl3_fwht(v, lane);
      for (int q = 0; q < 4; ++q) {
        const unsigned c = cb * 128 + lane + 32 * q;
        v[q] = __fmul_rn(v[q], __fmul_rn(__half2float(svh[c]), 0.125f));
        const uint64_t o = (uint64_t)pr[i] * g.n + c;
        if (m + 1 < M) first[j][q] = v[q];
        else if (g.epi == 0) reinterpret_cast<float*>(out)[o] = v[q];
        else reinterpret_cast<__nv_bfloat16*>(out)[o] = __float2bfloat16_rn(g.epi == 2 ? swiglu(first[j][q], v[q], limit) : v[q]);
      }
    }
    __syncthreads();
  }
}

template <int K>
__device__ __forceinline__ void tf_exl3x_prompt_units(const __nv_bfloat16* X, int x_stride, int slots, const uint8_t* W,
                                                      const __half* H, const int* items, const int* counts,
                                                      const int* members, void* out, float limit, int skip,
                                                      const TfExl3Experts g, float* buf) {
  const unsigned tid = threadIdx.x, nb = g.n / 128;
  const unsigned units = (unsigned)__ldg(counts) * nb;
  __shared__ unsigned xr[64], pr[64];
  for (unsigned unit = blockIdx.x; unit < units; unit += gridDim.x) {
    const unsigned it = unit / nb, cb = unit % nb;
    const int e = __ldg(items + 3 * it), first = __ldg(items + 3 * it + 1), cnt = __ldg(items + 3 * it + 2);
    if (e == skip) continue;
    for (int r0 = 0; r0 < cnt; r0 += 64) {
      const unsigned rows = cnt - r0 < 64 ? cnt - r0 : 64;
      if (tid < 64) {
        const int p = tid < rows ? __ldg(members + first + r0 + tid) : 0;
        pr[tid] = p;
        xr[tid] = slots ? p / slots : p;
      }
      __syncthreads();
      if (g.mats == 2) tf_exl3x_prompt_unit<K, 2>(X, x_stride, xr, pr, rows, W, H, out, limit, e, cb, g, buf);
      else tf_exl3x_prompt_unit<K, 1>(X, x_stride, xr, pr, rows, W, H, out, limit, e, cb, g, buf);
    }
  }
}

#define TF_EXL3X_PROMPT_CASE(B) \
  case B: tf_exl3x_prompt_units<B>(X, x_stride, slots, W, H, items, counts, members, out, limit, skip, g, buf); break;
// One prompt kernel for every width and codebook. buf holds two matrices' fp16 rows, then fp32 sums.
extern "C" __global__ void __launch_bounds__(256) tf_exl3_experts_prompt(const __nv_bfloat16* X, int x_stride, int slots,
                                                                        const uint8_t* W, const __half* H, const int* items,
                                                                        const int* counts, const int* members, void* out,
                                                                        float limit, int skip, const TfExl3Experts g) {
  __shared__ __align__(16) float buf[64 * TF_EXL3_XS];
  switch (g.bits) {
    TF_EXL3X_PROMPT_CASE(1) TF_EXL3X_PROMPT_CASE(2) TF_EXL3X_PROMPT_CASE(3) TF_EXL3X_PROMPT_CASE(4)
    TF_EXL3X_PROMPT_CASE(5) TF_EXL3X_PROMPT_CASE(6) TF_EXL3X_PROMPT_CASE(7) TF_EXL3X_PROMPT_CASE(8)
    default: break;
  }
}

// -- mixed-width mma decode -------------------------------------------------------------------------
// The decode path for GLM-5.3-Flash's mixed-width EXL3 checkpoints: the grid/warp/mma skeleton of
// BobClawblaw/TensorFold pr/glm-flash-cuda @ 3c6ed009 (zig/kernels/cuda/glm/exl3.cu, grouped<MT,NT,W>:
// grid (item, n-block, mat*SK+split), warps on fixed k ranges, mma.m16n8k16, fp32 partials summed in warp
// order), decoding OUR trellis tiles through exl3.cuh (tf_exl3_words/tf_exl3_unpack_any/tf_exl3_pair) with
// per-matrix TfExl3Mat descriptors. rot_in/gateup/down mirror that file's glm_exl3_rot_in/gateup/down with
// our slots (top_k+1; the shared expert's slot skipped) and per-expert flat scale tables.

namespace {

__device__ __forceinline__ float tf_exl3m_bf16r(float x) {
    return __bfloat162float(__float2bfloat16_rn(x));
}

template <int MT, int NT, int W, int K>
__device__ __forceinline__ void tf_exl3m_body(const half* __restrict__ X0, const half* __restrict__ X1,
        const TfExl3Mat* __restrict__ mdb, const int* __restrict__ items, const int* __restrict__ counts,
        const int* __restrict__ members, float* __restrict__ Z, int N, int P, int SK, int skip, unsigned book,
        int mats, int* __restrict__ rows_sh, float* __restrict__ red) {
    const int item = blockIdx.x;
    if (item >= counts[0]) return;
    const int e = items[3 * item];
    if (e == skip) return;
    const int first = items[3 * item + 1], cnt = items[3 * item + 2];
    const int mat = blockIdx.z / SK;
    const TfExl3Mat& md = mdb[(size_t)e * mats + mat];
    const int split = blockIdx.z % SK;
    const half* X = mat ? X1 : X0;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g = lane >> 2, t = lane & 3;
    const int KT = md.k >> 4, NTILES = md.tiles_n;
    constexpr int ROWS = 16 * MT;

    for (int i = threadIdx.x; i < ROWS; i += W * 32) rows_sh[i] = i < cnt ? members[first + i] : -1;
    __syncthreads();
    int r0[MT], r1[MT];
    const half* x0[MT];
    const half* x1[MT];
#pragma unroll
    for (int m = 0; m < MT; ++m) {
        r0[m] = rows_sh[m * 16 + g];
        r1[m] = rows_sh[m * 16 + g + 8];
        x0[m] = X + (size_t)(r0[m] < 0 ? 0 : r0[m]) * md.k + 2 * t;
        x1[m] = X + (size_t)(r1[m] < 0 ? 0 : r1[m]) * md.k + 2 * t;
    }

    const int per_split = KT / SK, per_warp = per_split / W;
    const int kt0 = split * per_split + warp * per_warp;
    const int nt0 = blockIdx.y * NT;
    const uint32_t* tile = reinterpret_cast<const uint32_t*>(md.trellis) + ((size_t)kt0 * NTILES + nt0) * (8 * K);

    float acc[MT][NT][2][4];
#pragma unroll
    for (int m = 0; m < MT; ++m)
#pragma unroll
        for (int i = 0; i < NT; ++i)
#pragma unroll
            for (int h = 0; h < 2; ++h)
#pragma unroll
                for (int c = 0; c < 4; ++c) acc[m][i][h][c] = 0.f;

    for (int kt = kt0; kt < kt0 + per_warp; ++kt) {
        const int k = kt * 16;
        uint32_t a[MT][4];
#pragma unroll
        for (int m = 0; m < MT; ++m) {
            const bool ok0 = r0[m] >= 0, ok1 = r1[m] >= 0;
            a[m][0] = ok0 ? *reinterpret_cast<const uint32_t*>(x0[m] + k) : 0u;
            a[m][1] = ok1 ? *reinterpret_cast<const uint32_t*>(x1[m] + k) : 0u;
            a[m][2] = ok0 ? *reinterpret_cast<const uint32_t*>(x0[m] + k + 8) : 0u;
            a[m][3] = ok1 ? *reinterpret_cast<const uint32_t*>(x1[m] + k + 8) : 0u;
        }
#pragma unroll
        for (int i = 0; i < NT; ++i) {
            const uint4 tw = tf_exl3_words<K>(tile + i * (8 * K), lane);
            __half w[8];
            tf_exl3_unpack_any<K>(tw, lane, book, w);
            const uint32_t b0 = tf_exl3_pair(w[0], w[1]), b1 = tf_exl3_pair(w[2], w[3]);
            const uint32_t b2 = tf_exl3_pair(w[4], w[5]), b3 = tf_exl3_pair(w[6], w[7]);
#pragma unroll
            for (int m = 0; m < MT; ++m) {
                tf_exl3_mma(acc[m][i][0], a[m], b0, b1);
                tf_exl3_mma(acc[m][i][1], a[m], b2, b3);
            }
        }
        tile += (size_t)NTILES * (8 * K);
    }

#pragma unroll
    for (int m = 0; m < MT; ++m)
#pragma unroll
        for (int i = 0; i < NT; ++i)
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int col = i * 16 + h * 8 + 2 * t;
                const size_t p0 = ((size_t)warp * ROWS + m * 16 + g) * (NT * 16) + col;
                const size_t p1 = ((size_t)warp * ROWS + m * 16 + g + 8) * (NT * 16) + col;
                red[p0] = acc[m][i][h][0];
                red[p0 + 1] = acc[m][i][h][1];
                red[p1] = acc[m][i][h][2];
                red[p1 + 1] = acc[m][i][h][3];
            }
    __syncthreads();
    for (int idx = threadIdx.x; idx < ROWS * NT * 16; idx += W * 32) {
        const int row = idx / (NT * 16), col = idx % (NT * 16);
        const int r = rows_sh[row];
        if (r < 0) continue;
        float s = red[row * (NT * 16) + col];
#pragma unroll
        for (int w = 1; w < W; ++w) s += red[((size_t)w * ROWS + row) * (NT * 16) + col];
        Z[(((size_t)mat * SK + split) * P + r) * N + blockIdx.y * NT * 16 + col] = s;
    }
}

}  // namespace

// One mma pass: gate and up (mats 2, X0/X1 the two rotations) or down (mats 1, X0 = X1). Z[mat][split][pair][n].
template <int MT, int NT, int W>
__device__ __forceinline__ void tf_exl3m_grouped(const half* __restrict__ X0, const half* __restrict__ X1,
        const TfExl3Mat* __restrict__ mdb, const int* __restrict__ items, const int* __restrict__ counts,
        const int* __restrict__ members, float* __restrict__ Z, int N, int P, int SK, int skip, unsigned book,
        int mats) {
    if (blockIdx.x >= counts[0]) return;
    const int e = items[3 * blockIdx.x];
    if (e == skip) return;
    extern __shared__ char tf_exl3m_smem[];  // rows then the warps' partial sums: one block, both widths
    int* rows_sh = reinterpret_cast<int*>(tf_exl3m_smem);
    float* red = reinterpret_cast<float*>(tf_exl3m_smem + ((16 * MT * 4 + 15) & ~15));
    const unsigned bits = mdb[(size_t)e * mats + blockIdx.z / SK].bits;
    if (bits == 3) tf_exl3m_body<MT, NT, W, 3>(X0, X1, mdb, items, counts, members, Z, N, P, SK, skip, book, mats, rows_sh, red);
    else if (bits == 4) tf_exl3m_body<MT, NT, W, 4>(X0, X1, mdb, items, counts, members, Z, N, P, SK, skip, book, mats, rows_sh, red);
}

// X rows (bf16, x_stride apart) -> the pairs' gate and up rotations: xg/xu [pairs][K] half. One warp a program.
extern "C" __global__ void tf_exl3m_rot_in(const __nv_bfloat16* __restrict__ x, int x_stride, int slots,
        const int* __restrict__ pick, const half* __restrict__ suh_g, const half* __restrict__ suh_u,
        half* __restrict__ xg, half* __restrict__ xu, int K, int skip) {
    const int p = blockIdx.x, blk = blockIdx.y, mat = blockIdx.z;
    const int lane = threadIdx.x;
    const int e = pick[p];
    const bool live = e != skip;
    const half* suh = (mat ? suh_u : suh_g) + (size_t)(live ? e : 0) * K + blk * 128 + 4 * lane;
    const __nv_bfloat16* xr = x + (size_t)(p / slots) * x_stride + blk * 128 + 4 * lane;
    float v[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) v[j] = live ? __bfloat162float(xr[j]) * __half2float(suh[j]) : 0.f;
    tf_exl3_fwht(v, lane);
    half* o = (mat ? xu : xg) + (size_t)p * K + blk * 128 + 4 * lane;
#pragma unroll
    for (int j = 0; j < 4; ++j) o[j] = __float2half_rn(v[j] * 0.08838834764831845f);
}

// items of 32 pairs, 2 n-tiles a block, 4 warps: the middle shape — one tile decode feeds twice the rows of
// grouped16 at half grouped64's register pressure (shared windows' hot experts hold 20-30 pairs).
extern "C" __global__ void __launch_bounds__(128) tf_exl3m_grouped32(const half* __restrict__ X0,
        const half* __restrict__ X1, const TfExl3Mat* __restrict__ mdb, const int* __restrict__ items,
        const int* __restrict__ counts, const int* __restrict__ members, float* __restrict__ Z, int N, int P,
        int SK, int skip, unsigned book, int mats) {
    tf_exl3m_grouped<2, 2, 4>(X0, X1, mdb, items, counts, members, Z, N, P, SK, skip, book, mats);
}

// items of 64 pairs, 2 n-tiles a block, 4 warps (their glm_exl3_grouped64's shape): one tile decode feeds
// four times the rows, so shared windows amortize the weight reads instead of scaling with rows.
extern "C" __global__ void __launch_bounds__(128) tf_exl3m_grouped64(const half* __restrict__ X0,
        const half* __restrict__ X1, const TfExl3Mat* __restrict__ mdb, const int* __restrict__ items,
        const int* __restrict__ counts, const int* __restrict__ members, float* __restrict__ Z, int N, int P,
        int SK, int skip, unsigned book, int mats) {
    // NT=1: half the accumulator registers of the first try's <4,2,4> (which regressed) at 16 columns a block
    tf_exl3m_grouped<4, 1, 4>(X0, X1, mdb, items, counts, members, Z, N, P, SK, skip, book, mats);
}

// items of 16 pairs, 8 n-tiles a block, 4 warps (their glm_exl3_grouped's shape).
extern "C" __global__ void __launch_bounds__(128) tf_exl3m_grouped16(const half* __restrict__ X0,
        const half* __restrict__ X1, const TfExl3Mat* __restrict__ mdb, const int* __restrict__ items,
        const int* __restrict__ counts, const int* __restrict__ members, float* __restrict__ Z, int N, int P,
        int SK, int skip, unsigned book, int mats) {
    tf_exl3m_grouped<1, 8, 4>(X0, X1, mdb, items, counts, members, Z, N, P, SK, skip, book, mats);
}

// The FUSED down pass (TF_GLM_MMA_FDOWN, on when SK_DOWN=1): the grouped16 shape's accumulation (MT=1, NT=8 -
// one 128-column Hadamard block a program) with the down epilogue in-kernel: the warps' partial sums land in
// shared, one warp a row runs the 128-wide FWHT, the 0.0883 * svh_d scale applies, and the result writes ey
// directly. No Z round trip and one launch less a layer (the python engine's fuse=1 design). The shared
// expert's item writes that pair's ey zeros, exactly as the epilogue it replaces did.
template <int K>
__device__ __forceinline__ void tf_exl3m_fdown_body(const half* __restrict__ X, const TfExl3Mat& md,
        const int* __restrict__ members, int first, int cnt, const half* __restrict__ svh_d,
        float* __restrict__ ey, int D, int slots, int ys, bool dead, unsigned book, int* __restrict__ rows_sh,
        float* __restrict__ red, float* __restrict__ buf) {
    constexpr int NT = 8, W = 4, ROWS = 16;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g = lane >> 2, t = lane & 3;
    const int KT = md.k >> 4, NTILES = md.tiles_n;

    for (int i = threadIdx.x; i < ROWS; i += W * 32) rows_sh[i] = i < cnt ? members[first + i] : -1;
    __syncthreads();
    int r0v = rows_sh[g], r1v = rows_sh[g + 8];
    const half* x0 = X + (size_t)(r0v < 0 ? 0 : r0v) * md.k + 2 * t;
    const half* x1 = X + (size_t)(r1v < 0 ? 0 : r1v) * md.k + 2 * t;

    const int per_warp = KT / W;
    const int kt0 = warp * per_warp;
    const int nt0 = blockIdx.y * NT;
    const uint32_t* tile = reinterpret_cast<const uint32_t*>(md.trellis) + ((size_t)kt0 * NTILES + nt0) * (8 * K);

    float acc[NT][2][4];
#pragma unroll
    for (int i = 0; i < NT; ++i)
#pragma unroll
        for (int h = 0; h < 2; ++h)
#pragma unroll
            for (int cc = 0; cc < 4; ++cc) acc[i][h][cc] = 0.f;

    if (!dead) {
        for (int kt = kt0; kt < kt0 + per_warp; ++kt) {
            const int k = kt * 16;
            uint32_t a[4];
            const bool ok0 = r0v >= 0, ok1 = r1v >= 0;
            a[0] = ok0 ? *reinterpret_cast<const uint32_t*>(x0 + k) : 0u;
            a[1] = ok1 ? *reinterpret_cast<const uint32_t*>(x1 + k) : 0u;
            a[2] = ok0 ? *reinterpret_cast<const uint32_t*>(x0 + k + 8) : 0u;
            a[3] = ok1 ? *reinterpret_cast<const uint32_t*>(x1 + k + 8) : 0u;
#pragma unroll
            for (int i = 0; i < NT; ++i) {
                const uint4 tw = tf_exl3_words<K>(tile + i * (8 * K), lane);
                __half w[8];
                tf_exl3_unpack_any<K>(tw, lane, book, w);
                const uint32_t b0 = tf_exl3_pair(w[0], w[1]), b1 = tf_exl3_pair(w[2], w[3]);
                const uint32_t b2 = tf_exl3_pair(w[4], w[5]), b3 = tf_exl3_pair(w[6], w[7]);
                tf_exl3_mma(acc[i][0], a, b0, b1);
                tf_exl3_mma(acc[i][1], a, b2, b3);
            }
            tile += (size_t)NTILES * (8 * K);
        }
    }

#pragma unroll
    for (int i = 0; i < NT; ++i)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int col = i * 16 + h * 8 + 2 * t;
            const size_t p0 = ((size_t)warp * ROWS + g) * (NT * 16) + col;
            const size_t p1 = ((size_t)warp * ROWS + g + 8) * (NT * 16) + col;
            red[p0] = acc[i][h][0];
            red[p0 + 1] = acc[i][h][1];
            red[p1] = acc[i][h][2];
            red[p1 + 1] = acc[i][h][3];
        }
    __syncthreads();
    for (int idx = threadIdx.x; idx < ROWS * NT * 16; idx += W * 32) {
        const int row = idx / (NT * 16), col = idx % (NT * 16);
        float s = red[row * (NT * 16) + col];
#pragma unroll
        for (int w = 1; w < W; ++w) s += red[((size_t)w * ROWS + row) * (NT * 16) + col];
        buf[row * (NT * 16) + col] = s;
    }
    __syncthreads();
    const half* svh = dead ? nullptr : svh_d;
#pragma unroll
    for (int j = 0; j < ROWS / W; ++j) {  // W warps cover the 16 rows: 4 each (j < 8 walked rows 16..31)
        const int row = warp + W * j;
        const int p = rows_sh[row];
        if (p < 0) continue;
        float v[4];
#pragma unroll
        for (int q = 0; q < 4; ++q) v[q] = buf[row * 128 + lane + 32 * q];
        tf_exl3_fwht(v, lane);
        float* o = ey + ((size_t)(p / slots) * ys + (p % slots)) * D + blockIdx.y * 128;
#pragma unroll
        for (int q = 0; q < 4; ++q)
            o[lane + 32 * q] = dead ? 0.f : v[q] * 0.08838834764831845f * __half2float(svh[(size_t)0 * D + blockIdx.y * 128 + lane + 32 * q]);
    }
}

extern "C" __global__ void __launch_bounds__(128) tf_exl3m_fdown(const half* __restrict__ X,
        const TfExl3Mat* __restrict__ mdb, const int* __restrict__ items, const int* __restrict__ counts,
        const int* __restrict__ members, const int* __restrict__ pick, const half* __restrict__ svh_d,
        float* __restrict__ ey, int D, int P, int slots, int ys, int skip, unsigned book) {
    const int item = blockIdx.x;
    if (item >= counts[0]) return;
    const int e = items[3 * item], first = items[3 * item + 1], cnt = items[3 * item + 2];
    const bool dead = e == skip;
    extern __shared__ char fdsm[];
    int* rows_sh = reinterpret_cast<int*>(fdsm);
    float* red = reinterpret_cast<float*>(fdsm + 64);
    float* buf = reinterpret_cast<float*>(fdsm + 64 + 4 * 16 * 128 * 4);
    // the shared expert's item (e == skip) has no mdb row: a dummy mat with k = D keeps the row-stride
    // reads in bounds; its mma loop never runs and the epilogue writes the pair's ey zeros
    TfExl3Mat mds = TfExl3Mat{};
    mds.k = D;
    mds.bits = 4;
    const TfExl3Mat& md = dead ? mds : mdb[(size_t)e];
    const half* svh = dead ? svh_d : svh_d + (size_t)e * D;
    if (md.bits == 3)
        tf_exl3m_fdown_body<3>(X, md, members, first, cnt, svh, ey, D, slots, ys, dead, book, rows_sh, red, buf);
    else if (md.bits == 4)
        tf_exl3m_fdown_body<4>(X, md, members, first, cnt, svh, ey, D, slots, ys, dead, book, rows_sh, red, buf);
}

// The gate/up sums: splits in order, rotated, svh, GLM's limited SwiGLU (their rounding chain), then the down
// projection's input rotation: xd[pair][N] half.
extern "C" __global__ void tf_exl3m_gateup(const float* __restrict__ Z, const int* __restrict__ pick,
        const half* __restrict__ svh_g, const half* __restrict__ svh_u, const half* __restrict__ suh_d,
        half* __restrict__ xd, int P, int N, int SK, float limit, int skip) {
    const int p = blockIdx.x, blk = blockIdx.y;
    const int lane = threadIdx.x;
    const int e = pick[p];
    const int n = blk * 128 + 4 * lane;
    float gv[4], uv[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        float sg = 0.f, su = 0.f;
        for (int s = 0; s < SK; ++s) {
            sg += Z[((size_t)(0 * SK + s) * P + p) * N + n + j];
            su += Z[((size_t)(1 * SK + s) * P + p) * N + n + j];
        }
        gv[j] = sg;
        uv[j] = su;
    }
    tf_exl3_fwht(gv, lane);
    tf_exl3_fwht(uv, lane);
    float v[4];
    if (e == skip) {
#pragma unroll
        for (int j = 0; j < 4; ++j) v[j] = 0.f;
    } else {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float gg = fminf(tf_exl3m_bf16r(gv[j] * 0.08838834764831845f * __half2float(svh_g[(size_t)e * N + n + j])), limit);
            const float uu = fminf(fmaxf(tf_exl3m_bf16r(uv[j] * 0.08838834764831845f * __half2float(svh_u[(size_t)e * N + n + j])), -limit), limit);
            v[j] = tf_exl3m_bf16r(tf_exl3m_bf16r(gg / (1.f + expf(-gg))) * uu) * __half2float(suh_d[(size_t)e * N + n + j]);
        }
    }
    tf_exl3_fwht(v, lane);
    half* o = xd + (size_t)p * N + n;
#pragma unroll
    for (int j = 0; j < 4; ++j) o[j] = __float2half_rn(v[j] * 0.08838834764831845f);
}

// The down pass's sums: fp32 into the slot rows y[row][slot][D] (ys slots a row; the shared slot skipped).
extern "C" __global__ void tf_exl3m_down(const float* __restrict__ Z, const int* __restrict__ pick,
        const half* __restrict__ svh_d, float* __restrict__ y, int P, int D, int SK, int ys, int slots, int skip) {
    const int p = blockIdx.x, blk = blockIdx.y;
    const int lane = threadIdx.x;
    const int e = pick[p];
    const int n = blk * 128 + 4 * lane;
    float v[4];
    if (e == skip) {
#pragma unroll
        for (int j = 0; j < 4; ++j) v[j] = 0.f;
    } else {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            float s = 0.f;
            for (int k = 0; k < SK; ++k) s += Z[((size_t)k * P + p) * D + n + j];
            v[j] = s;
        }
        tf_exl3_fwht(v, lane);
#pragma unroll
        for (int j = 0; j < 4; ++j) v[j] = v[j] * 0.08838834764831845f * __half2float(svh_d[(size_t)e * D + n + j]);
    }
    float* o = y + ((size_t)(p / slots) * ys + (p % slots)) * D + n;
#pragma unroll
    for (int j = 0; j < 4; ++j) o[j] = v[j];
}
