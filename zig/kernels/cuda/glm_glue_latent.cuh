// latent.py's BF16 absorb, expand and cache write by hand (the MLA layers' 512-wide latent, 256-wide heads).
#pragma once

#include "glm_triton.cuh"

using namespace glm_triton;

// Tree verify (draft trees): per-row cache positions for the window write. With g_tree_on 0 these are
// unused and row r lands at pos[0] + r as always; a tree round sets g_tree_pos[r] to row r's depth-based
// slot so siblings no longer collide. Defined beside g_tree_on/g_tree_mask in glm_glue_attn.cuh.
extern "C" {
extern __device__ unsigned g_tree_on;
extern __device__ unsigned g_tree_pos[16];
}


// latent._absorb_q(_rows), block (head, 32 latent columns, row block) of 128 threads; rows per block change no bits.
extern "C" __global__ void __launch_bounds__(128)
    tf_glm_absorb_q(const __nv_bfloat16* q, const __nv_bfloat16* wk, __nv_bfloat16* qa, int64_t rows, int64_t heads,
                    int64_t rb) {
  constexpr int D = 256, LW = 512;
  __shared__ float smem[128];
  const int t = threadIdx.x, k0 = t >> 2, nq = t & 3, warp = t >> 5;
  const int64_t h = blockIdx.x, n0 = (int64_t)blockIdx.y * 32 + nq * 8, r0 = blockIdx.z * rb;
  float w[8][8];
  for (int rep = 0; rep < 8; ++rep)
    for (int j = 0; j < 8; ++j) w[rep][j] = bf(wk[(h * D + k0 + 32 * rep) * LW + n0 + j]);
  for (int64_t r = r0; r < r0 + rb && r < rows; ++r) {
    float qv[8];
    for (int rep = 0; rep < 8; ++rep) qv[rep] = bf(q[(r * heads + h) * D + k0 + 32 * rep]);
    for (int j = 0; j < 8; ++j) {
      float a = fmul(w[1][j], qv[1]);
      a = __fmaf_rn(w[0][j], qv[0], a);
      for (int rep = 2; rep < 8; ++rep) a = __fmaf_rn(w[rep][j], qv[rep], a);
      for (int o = 16; o >= 4; o >>= 1) a = a + __shfl_xor_sync(0xffffffffu, a, o);
      if ((t & 28) == 0) smem[(nq * 8 + j) * 4 + warp] = a;
    }
    __syncthreads();
    float v = smem[t];
    v = v + __shfl_xor_sync(0xffffffffu, v, 2);
    v = v + __shfl_xor_sync(0xffffffffu, v, 1);
    if ((t & 3) == 0) qa[(r * heads + h) * LW + (int64_t)blockIdx.y * 32 + t / 4] = __float2bfloat16_rn(v);
    __syncthreads();
  }
}

// latent._expand_v(_rows), block (head, 16 outputs, row block) of 128 threads: o . wv by fma, lanes xor 16 .. 1.
extern "C" __global__ void __launch_bounds__(128)
    tf_glm_expand_v(const __nv_bfloat16* ol, const __nv_bfloat16* wv, __nv_bfloat16* out, int64_t rows,
                    int64_t heads, int64_t rb) {
  constexpr int DV = 256, LW = 512;
  __shared__ float smem[32];
  const int t = threadIdx.x, lane = t & 31, warp = t >> 5, kb = (warp & 1) * 256 + lane * 8, nb = warp >> 1;
  const int64_t h = blockIdx.x, n0 = (int64_t)blockIdx.y * 16, r0 = blockIdx.z * rb;
  float w[8][8];
  for (int rep = 0; rep < 8; ++rep)
    for (int j = 0; j < 8; ++j) w[rep][j] = bf(wv[(h * DV + n0 + nb + 2 * rep) * LW + kb + j]);
  for (int64_t r = r0; r < r0 + rb && r < rows; ++r) {
    float o[8];
    for (int j = 0; j < 8; ++j) o[j] = bf(ol[(r * heads + h) * LW + kb + j]);
    for (int rep = 0; rep < 8; ++rep) {
      float a = fmul(w[rep][1], o[1]);
      a = __fmaf_rn(w[rep][0], o[0], a);
      for (int j = 2; j < 8; ++j) a = __fmaf_rn(w[rep][j], o[j], a);
      for (int s = 16; s > 0; s >>= 1) a = a + __shfl_xor_sync(0xffffffffu, a, s);
      if (lane == 0) smem[(nb + 2 * rep) * 2 + (warp & 1)] = a;
    }
    __syncthreads();
    if (t < 32) {
      float v = smem[t];
      v = v + __shfl_xor_sync(0xffffffffu, v, 1);
      if ((t & 1) == 0) out[(r * heads + h) * DV + n0 + t / 2] = __float2bfloat16_rn(v);
    }
    __syncthreads();
  }
}

// latent._lat_write (BF16 cache): row r of `lat` into cache slot pos + r.
extern "C" __global__ void tf_glm_lat_write(const __nv_bfloat16* lat, int64_t lat_stride, __nv_bfloat16* cache,
                                            const int* pos, int64_t lw) {
  const int64_t r = blockIdx.x, at = (g_tree_on != 0 && r < 16) ? (int64_t)g_tree_pos[r] : (int64_t)pos[0] + r;
  for (int64_t k = threadIdx.x; k < lw; k += blockDim.x) cache[at * lw + k] = lat[r * lat_stride + k];
}

#include <cuda_fp8.h>

// FP8 (E4M3) latent cache: TF_GLM_KV_FP8=1 halves the latent cache bytes.
// Write: bf16 row → e4m3 in the cache (half the bytes per value).
extern "C" __global__ void tf_glm_lat_write_fp8(const __nv_bfloat16* lat, int64_t lat_stride, unsigned char* cache,
                                                 const int* pos, int64_t lw) {
  const int64_t r = blockIdx.x, at = (int64_t)pos[0] + r;
  for (int64_t k = threadIdx.x; k < lw; k += blockDim.x)
    cache[at * lw + k] = *reinterpret_cast<const unsigned char*>(&__nv_fp8_e4m3(lat[r * lat_stride + k]));
}

// Read: `count` e4m3 rows (starting at `start`) dequantized into a bf16 scratch the attention reads.
// The scratch is sized [count][lw] bf16, allocated once per stream's decode buffers.
extern "C" __global__ void tf_glm_lat_dequant(const unsigned char* cache, __nv_bfloat16* out,
                                               int64_t start, int64_t count, int64_t lw) {
  const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count * lw) return;
  const int64_t row = i / lw, k = i % lw;
  out[i] = __nv_bfloat16(__nv_fp8_e4m3(reinterpret_cast<const __nv_fp8_e4m3*>(cache + (start + row) * lw)[k]));
}

// Read selected tokens (sparse attention): e4m3 rows at indices tok[0..count) into a bf16 scratch.
extern "C" __global__ void tf_glm_lat_dequant_tok(const unsigned char* cache, const int* tok,
                                                   __nv_bfloat16* out, int64_t count, int64_t w, int64_t lw) {
  const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count * lw) return;
  const int64_t row = i / lw, k = i % lw;
  const int64_t src = tok[row];
  out[i] = __nv_bfloat16(__nv_fp8_e4m3(reinterpret_cast<const __nv_fp8_e4m3*>(cache + src * lw)[k]));
}


// -- NVFP4 latent cache (TF_GLM_KV_NVFP4): e2m1 values (4 bits) + e4m3 scales per 16-value block.
// 512 values per row -> 256 bytes packed fp4 + 32 bytes e4m3 scales = 288 bytes vs 1024 bf16 (3.56x compression).
// The quantization: per-block amax / 6.0 -> e4m3 scale; values rounded to nearest e2m1 (±{0,0.5,1,1.5,2,3,4,6}).

// e2m1's 8 magnitudes
__device__ __forceinline__ float nvfp4_val(unsigned char code) {
  const float vals[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
  return (code & 8) ? -vals[code & 7] : vals[code & 7];
}

// nearest e2m1 code for a scaled value
__device__ __forceinline__ unsigned char nvfp4_code(float v) {
  float av = fabsf(v);
  unsigned char sign = v < 0.0f ? 8 : 0;
  if (av < 0.25f) return sign;
  if (av < 0.75f) return sign | 1;
  if (av < 1.25f) return sign | 2;
  if (av < 1.75f) return sign | 3;
  if (av < 2.5f) return sign | 4;
  if (av < 3.5f) return sign | 5;
  if (av < 5.0f) return sign | 6;
  return sign | 7;
}

// Write: bf16 row [lw] -> packed e2m1 [lw/2 bytes] + e4m3 scales [lw/16 bytes] in the cache slot.
// cache layout per row: [lw/2 packed bytes][lw/16 e4m3 scale bytes] (contiguous per row for coalescing).
extern "C" __global__ void tf_glm_lat_write_nvfp4(const __nv_bfloat16* lat, int64_t lat_stride,
                                                    unsigned char* cache, const int* pos, int64_t lw) {
  const int64_t r = blockIdx.x, at = (int64_t)pos[0] + r;
  const int64_t packed = lw / 2, scales = lw / 16;
  unsigned char* row_cache = cache + at * (packed + scales);
  unsigned char* codes = row_cache;
  unsigned char* sc = row_cache + packed;
  const __nv_bfloat16* src = lat + r * lat_stride;
  // each thread handles one 16-value block: computes amax, scale, quantizes
  const int nblocks = (int)scales;
  for (int b = threadIdx.x; b < nblocks; b += blockDim.x) {
    float vals[16];
    float amax = 0.0f;
    #pragma unroll
    for (int j = 0; j < 16; ++j) {
      vals[j] = __bfloat162float(src[b * 16 + j]);
      amax = fmaxf(amax, fabsf(vals[j]));
    }
    // e4m3 scale: amax / 6.0 (the max e2m1 magnitude), stored as a raw byte
    const float scale = amax > 0.0f ? amax / 6.0f : 1.0f;
    const float inv = 1.0f / scale;
    // store the e4m3 scale byte (round-to-nearest via float-to-fp8)
    __half s_h = __float2half_rn(scale);
    __nv_fp8_e4m3 s_f8(s_h); // fp8 constructor from half
    sc[b] = *reinterpret_cast<const unsigned char*>(&s_f8);
    // quantize 16 values to 8 packed bytes (2 codes per byte)
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
      unsigned char c0 = nvfp4_code(vals[2 * j] * inv);
      unsigned char c1 = nvfp4_code(vals[2 * j + 1] * inv);
      codes[b * 8 + j] = (c0 & 0xF) | ((c1 & 0xF) << 4);
    }
  }
}

// Dequant: nvfp4 rows [start, start + count) -> bf16 scratch [count][lw].
extern "C" __global__ void tf_glm_lat_dequant_nvfp4(const unsigned char* cache, __nv_bfloat16* out,
                                                       int64_t start, int64_t count, int64_t lw) {
  const int64_t packed = lw / 2, scales = lw / 16;
  const int64_t row = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= count) return;
  const unsigned char* row_cache = cache + (start + row) * (packed + scales);
  const unsigned char* codes = row_cache;
  const unsigned char* sc = row_cache + packed;
  __nv_bfloat16* dst = out + row * lw;
  for (int b = 0; b < scales; ++b) {
    // dequant the e4m3 scale
    __nv_fp8_e4m3 s_f8;
    *reinterpret_cast<unsigned char*>(&s_f8) = sc[b];
    const float scale = __half2float(__half(s_f8));
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
      unsigned char byte = codes[b * 8 + j];
      dst[b * 16 + 2 * j] = __float2bfloat16_rn(nvfp4_val(byte & 0xF) * scale);
      dst[b * 16 + 2 * j + 1] = __float2bfloat16_rn(nvfp4_val((byte >> 4) & 0xF) * scale);
    }
  }
}
