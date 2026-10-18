// Offline check of tf_glm_kda_chain's branch path (stage 4b). Row 1 is made a no-op on the state
// (beta 0, decay 1), so a chain [0,1,2] and a branch run with row 2 parented to row 0 must agree on
// row 2: same state, same conv taps. Rows 0-1 must also equal the plain chain.
#include "glm_kda.cu"
#include <cstdio>
#include <cmath>
#include <cstring>
#include <vector>

static constexpr int H = 1, PS = 512, AS = 128, GS = 128;
static constexpr int C = 3 * H * DK;  // 384 conv channels
static constexpr int B_OFF = 3 * H * DK;
using bf16 = __nv_bfloat16;

static float val(int i, float f) { return 0.3f * std::sin((double)i * f); }
static bf16 B(float x) { return __float2bfloat16(x); }

struct In {
  std::vector<bf16> P, A, G, cs, norm_w;
  std::vector<float> cw, state_in, a_log, dt_bias;
};

static In makeInputs(int rows) {
  In in;
  in.P.resize(rows * PS);
  for (int r = 0; r < rows; ++r)
    for (int c2 = 0; c2 < PS; ++c2) in.P[r * PS + c2] = B(val(r * PS + c2, 0.013f));
  in.A.resize(rows * AS);
  for (int r = 0; r < rows; ++r)
    for (int i = 0; i < AS; ++i) in.A[r * AS + i] = B(0.1f * val(r * AS + i, 0.021f));
  in.G.resize(rows * GS);
  for (int r = 0; r < rows; ++r)
    for (int i = 0; i < GS; ++i) in.G[r * GS + i] = B(val(r * GS + i, 0.017f));
  in.cs.assign((TAPS - 1) * C, B(0.01f));
  in.cw.resize(C * TAPS);
  for (int c2 = 0; c2 < C; ++c2)
    for (int k = 0; k < TAPS; ++k) in.cw[c2 * TAPS + k] = 0.25f + 0.01f * (c2 % 7) * (k + 1);
  in.state_in.resize(H * DV * DK);
  for (size_t i = 0; i < in.state_in.size(); ++i) in.state_in[i] = 0.02f * std::cos((double)i * 0.0011);
  in.a_log.resize(H, 0.1f);
  in.dt_bias.resize(H * DK);
  for (int i = 0; i < H * DK; ++i) in.dt_bias[i] = 0.05f * val(i, 0.009f);
  in.norm_w.resize(DV);
  for (int i = 0; i < DV; ++i) in.norm_w[i] = B(0.8f + 0.05f * (i % 5));
  return in;
}

struct Dev {
  bf16 *P, *A, *G, *cs, *norm_w, *out;
  float *cw, *state_in, *a_log, *dt_bias;
  int* parents;
};

static Dev alloc(const In& in, int rows) {
  Dev d;
  auto up = [](const void* src, size_t bytes, void** dst) {
    cudaMalloc(dst, bytes);
    cudaMemcpy(*dst, src, bytes, cudaMemcpyHostToDevice);
  };
  up(in.P.data(), in.P.size() * 2, (void**)&d.P);
  up(in.A.data(), in.A.size() * 2, (void**)&d.A);
  up(in.G.data(), in.G.size() * 2, (void**)&d.G);
  up(in.cs.data(), in.cs.size() * 2, (void**)&d.cs);
  up(in.norm_w.data(), in.norm_w.size() * 2, (void**)&d.norm_w);
  up(in.cw.data(), in.cw.size() * 4, (void**)&d.cw);
  up(in.state_in.data(), in.state_in.size() * 4, (void**)&d.state_in);
  up(in.a_log.data(), in.a_log.size() * 4, (void**)&d.a_log);
  up(in.dt_bias.data(), in.dt_bias.size() * 4, (void**)&d.dt_bias);
  cudaMalloc((void**)&d.out, (size_t)rows * H * DV * 2);
  cudaMalloc((void**)&d.parents, (size_t)rows * 4);
  return d;
}

static void run(Dev& d, int rows, const int* parents) {
  if (parents) cudaMemcpy(d.parents, parents, (size_t)rows * 4, cudaMemcpyHostToDevice);
  tf_glm_kda_chain<<<H, 1024>>>(H, d.P, PS, B_OFF, d.A, AS, d.G, GS, d.cs, d.cw, d.state_in,
                                d.a_log, d.dt_bias, d.norm_w, 1e-5f, 0.1f, rows, d.out, nullptr,
                                nullptr, nullptr, nullptr, nullptr, parents ? d.parents : nullptr);
}

int main() {
  // Row 1 is a state no-op: beta 0 (b channel very negative) and decay 1 (A very negative).
  In a = makeInputs(3);
  for (int i = 0; i < AS; ++i) a.A[AS + i] = B(-60.0f);
  a.P[PS + B_OFF] = B(-80.0f);

  Dev dc = alloc(a, 3);
  run(dc, 3, nullptr);  // the chain: rows 0, 1 (no-op), 2
  if (cudaDeviceSynchronize() != cudaSuccess) { printf("chain run failed\n"); return 1; }
  std::vector<bf16> chain(3 * H * DV);
  cudaMemcpy(chain.data(), dc.out, chain.size() * 2, cudaMemcpyDeviceToHost);

  Dev db = alloc(a, 3);
  int par[3] = {-1, 0, 0};  // row 2 is row 0's second child
  run(db, 3, par);
  if (cudaDeviceSynchronize() != cudaSuccess) { printf("branch run failed\n"); return 1; }
  std::vector<bf16> branch(3 * H * DV);
  cudaMemcpy(branch.data(), db.out, branch.size() * 2, cudaMemcpyDeviceToHost);

  int d01 = 0, d2 = 0;
  for (int i = 0; i < 2 * H * DV; ++i)
    if (std::memcmp(&chain[i], &branch[i], 2) != 0) ++d01;
  for (int i = 0; i < H * DV; ++i)
    if (std::memcmp(&chain[2 * H * DV + i], &branch[2 * H * DV + i], 2) != 0) ++d2;
  printf("rows 0-1 vs chain: %d of %d differ\n", d01, 2 * H * DV);
  printf("sibling row 2 vs chain (no-op row 1): %d of %d differ\n", d2, H * DV);
  printf("%s\n", (d01 == 0 && d2 == 0) ? "BRANCH PATH OK" : "BRANCH PATH WRONG");
  return (d01 == 0 && d2 == 0) ? 0 : 1;
}
