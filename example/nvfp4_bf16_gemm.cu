// =====================================================================================
// blackhole sm120 NVFP4 x NVFP4 -> BF16 GEMM —— 可运行示例 / 正确性 + 性能测试
// -------------------------------------------------------------------------------------
// 用法:
//   ./nvfp4_bf16_gemm [M N K]          (默认 4096 4096 4096)
//
// 流程: 随机生成 NVFP4 输入 -> CPU 参考 -> 跑 blackhole kernel -> 比对 -> 计时报 TFLOPS。
//
// 编译 / 运行 / ncu 采样方法见同目录 README.md。
//
// ------------------------------- 实测数据 (2026-06-13) -------------------------------
// 测试环境: GeForce RTX 5060 Ti (36 SMs, sm_120a), CUDA 13.1, Driver 595.71.05
//
// 正确性: 256 个随机抽样点, max_rel=0.0000, mismatches=0 -> PASSED (M=N=K=4096)
//
// 性能 (M=N=K=4096, iters=50 warmup+timed):
//   实测: 281.6 TFLOP/s @ ~2.6 GHz SM clock
//
// 理论峰值 (RTX 5060 Ti NVFP4 dense, non-sparse):
//   基于 GB202 170 SM / 1700 TFLOPS FP4 等比推算: 36 SM × 10 TFLOPS/SM ≈ 360 TFLOPS
//   频率修正 (2.6/2.45 GHz): ≈ 382 TFLOPS
//   MFU (实际频率): 73.8%
//   MFU (绝对峰值): 62.1%  (max boost 3.09 GHz)
//
// NCU 核心指标 (ncu --set full, 第 2 次 launch):
//   Tensor Pipeline (SM Busy):   68.5%   (最高利用率流水线, 由 Tensor (FP) 子流水线主导)
//   SM Cycles Active:            97.12%  (SM 几乎一直有活干)
//   DRAM Throughput:             16.3%   (远未跑满 -> 算力受限, 非带宽受限)
//   Issue Slots Busy:            16.27%  (发射槽利用率偏低)
//   Active Warps / Scheduler:    1.99 / 12  (低 occupancy, smem ~90KB/99KB 限制)
//   Eligible Warps / Scheduler:  0.21    (低 eligible -> 延迟隐藏不足)
//
//   瓶颈分析: Occupancy 是主要制约——每个 SM 只能容纳 1 个 CTA (shared memory 峰值 ~90KB,
//   上限 99KB)。这导致 scheduler 只有 ~2 个 active warp (上限 12), 无法充分隐藏 TMA
//   和 scale factor 加载延迟。可调方向: 减少流水级数 Stages 或缩小 Tile 尺寸以降低
//   smem 压力, 换取更高 occupancy。
// =====================================================================================
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>

#include <cuda_runtime.h>

#include <blackhole/sm120/nvfp4_gemm.hpp>

using namespace blackhole::sm120;
using KT = Nvfp4GemmTraits;

#define CHECK_CUDA(call)                                                        \
  do {                                                                          \
    cudaError_t _e = (call);                                                    \
    if (_e != cudaSuccess) {                                                    \
      printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); \
      std::exit(1);                                                             \
    }                                                                           \
  } while (0)

// e2m1 量化：把 float 量化成最近的 NVFP4 可表示值，返回 (4bit 编码, 还原后的 float)
static void quantize_e2m1(float v, uint8_t& code, float& deq) {
  cutlass::float_e2m1_t q(v);
  code = q.storage & 0xf;
  deq  = float(q);
}
// ue4m3 量化（无符号缩放因子）
static void quantize_ue4m3(float v, uint8_t& code, float& deq) {
  cutlass::float_ue4m3_t q(v);
  code = q.storage;
  deq  = float(q);
}

int main(int argc, char** argv) {
  int M = 4096, N = 4096, K = 4096;
  if (argc == 4) { M = atoi(argv[1]); N = atoi(argv[2]); K = atoi(argv[3]); }
  printf("NVFP4 x NVFP4 -> BF16 GEMM   M=%d N=%d K=%d\n", M, N, K);

  if (M % KT::TileM || N % KT::TileN || K % (2 * KT::TileK)) {
    printf("This example requires M%%128==0, N%%128==0, K%%256==0\n");
    return 1;
  }

  const int Kg = K / KT::SFVecSize;  // 每行的 SF 个数

  // ---- 主机端缓冲 ----
  std::vector<uint8_t> hA(size_t(M) * K / 2);          // 打包的 fp4 (2/字节)
  std::vector<uint8_t> hB(size_t(N) * K / 2);
  std::vector<float>   refA(size_t(M) * K);            // 还原值，供 CPU 参考
  std::vector<float>   refB(size_t(N) * K);

  auto layout_sfa = make_layout_SFA(M, K);
  auto layout_sfb = make_layout_SFB(N, K);
  std::vector<uint8_t> hSFA(cute::cosize(layout_sfa)); // ue4m3 1 字节
  std::vector<uint8_t> hSFB(cute::cosize(layout_sfb));
  std::vector<float>   sfaVal(size_t(M) * Kg);
  std::vector<float>   sfbVal(size_t(N) * Kg);

  std::mt19937 rng(1234);
  std::uniform_int_distribution<int> pick(0, 6);                  // fp4 量级别太大，控制累加范围
  const float fp4_levels[7] = {-2.f, -1.f, -0.5f, 0.f, 0.5f, 1.f, 2.f};
  const float sf_levels[4]  = {0.5f, 1.f, 1.f, 2.f};             // ue4m3 缩放
  std::uniform_int_distribution<int> pick_sf(0, 3);

  // 填 A + SFA
  for (int m = 0; m < M; ++m) {
    for (int kg = 0; kg < Kg; ++kg) {
      uint8_t sc; float sd; quantize_ue4m3(sf_levels[pick_sf(rng)], sc, sd);
      hSFA[layout_sfa(cute::make_coord(m, kg * KT::SFVecSize))] = sc;
      sfaVal[size_t(m) * Kg + kg] = sd;
    }
    for (int k = 0; k < K; ++k) {
      uint8_t code; float deq; quantize_e2m1(fp4_levels[pick(rng)], code, deq);
      refA[size_t(m) * K + k] = deq;
      size_t byte = (size_t(m) * K + k) / 2;
      if (k & 1) hA[byte] = (hA[byte] & 0x0f) | (code << 4);
      else       hA[byte] = (hA[byte] & 0xf0) | code;
    }
  }
  // 填 B + SFB
  for (int n = 0; n < N; ++n) {
    for (int kg = 0; kg < Kg; ++kg) {
      uint8_t sc; float sd; quantize_ue4m3(sf_levels[pick_sf(rng)], sc, sd);
      hSFB[layout_sfb(cute::make_coord(n, kg * KT::SFVecSize))] = sc;
      sfbVal[size_t(n) * Kg + kg] = sd;
    }
    for (int k = 0; k < K; ++k) {
      uint8_t code; float deq; quantize_e2m1(fp4_levels[pick(rng)], code, deq);
      refB[size_t(n) * K + k] = deq;
      size_t byte = (size_t(n) * K + k) / 2;
      if (k & 1) hB[byte] = (hB[byte] & 0x0f) | (code << 4);
      else       hB[byte] = (hB[byte] & 0xf0) | code;
    }
  }

  // ---- 设备端缓冲 ----
  uint8_t *dA, *dB, *dSFA, *dSFB; cutlass::bfloat16_t* dD;
  CHECK_CUDA(cudaMalloc(&dA, hA.size()));
  CHECK_CUDA(cudaMalloc(&dB, hB.size()));
  CHECK_CUDA(cudaMalloc(&dSFA, hSFA.size()));
  CHECK_CUDA(cudaMalloc(&dSFB, hSFB.size()));
  CHECK_CUDA(cudaMalloc(&dD, size_t(M) * N * sizeof(cutlass::bfloat16_t)));
  CHECK_CUDA(cudaMemcpy(dA, hA.data(), hA.size(), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dB, hB.data(), hB.size(), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dSFA, hSFA.data(), hSFA.size(), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dSFB, hSFB.data(), hSFB.size(), cudaMemcpyHostToDevice));

  // ---- 跑 kernel（预热 + 校验） ----
  CHECK_CUDA(nvfp4_bf16_gemm(M, N, K, dA, dSFA, dB, dSFB, dD));
  CHECK_CUDA(cudaDeviceSynchronize());

  std::vector<cutlass::bfloat16_t> hD(size_t(M) * N);
  CHECK_CUDA(cudaMemcpy(hD.data(), dD, hD.size() * sizeof(cutlass::bfloat16_t), cudaMemcpyDeviceToHost));

  // ---- CPU 参考（抽样比对若干行，全量 4096^3 太慢） ----
  printf("Verifying (sampled rows)...\n");
  std::mt19937 rsel(7);
  std::uniform_int_distribution<int> rowsel(0, M - 1), colsel(0, N - 1);
  int checked = 0, bad = 0; double max_rel = 0;
  for (int s = 0; s < 256; ++s) {
    int m = rowsel(rsel), n = colsel(rsel);
    float acc = 0.f;
    for (int k = 0; k < K; ++k) {
      float a = refA[size_t(m) * K + k] * sfaVal[size_t(m) * Kg + k / KT::SFVecSize];
      float b = refB[size_t(n) * K + k] * sfbVal[size_t(n) * Kg + k / KT::SFVecSize];
      acc += a * b;
    }
    float got = float(hD[size_t(m) * N + n]);
    float ref = float(cutlass::bfloat16_t(acc));         // 参考也转 bf16 再比
    float rel = std::abs(got - ref) / (std::abs(ref) + 1e-3f);
    max_rel = std::max(max_rel, double(rel));
    ++checked;
    if (rel > 0.05f) { if (bad < 8) printf("  mismatch (%d,%d): got %.3f ref %.3f\n", m, n, got, ref); ++bad; }
  }
  printf("Checked %d elems, max_rel=%.4f, mismatches=%d -> %s\n",
         checked, max_rel, bad, bad == 0 ? "PASSED" : "FAILED");

  // ---- 计时 ----
  const int iters = 50;
  cudaEvent_t beg, end; CHECK_CUDA(cudaEventCreate(&beg)); CHECK_CUDA(cudaEventCreate(&end));
  CHECK_CUDA(nvfp4_bf16_gemm(M, N, K, dA, dSFA, dB, dSFB, dD));  // warmup
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaEventRecord(beg));
  for (int i = 0; i < iters; ++i) {
    CHECK_CUDA(nvfp4_bf16_gemm(M, N, K, dA, dSFA, dB, dSFB, dD));
  }
  CHECK_CUDA(cudaEventRecord(end));
  CHECK_CUDA(cudaEventSynchronize(end));
  float ms = 0; CHECK_CUDA(cudaEventElapsedTime(&ms, beg, end)); ms /= iters;
  double tflops = 2.0 * M * N * K / (ms * 1e-3) / 1e12;
  printf("Avg %.3f ms   %.1f TFLOP/s\n", ms, tflops);

  cudaFree(dA); cudaFree(dB); cudaFree(dSFA); cudaFree(dSFB); cudaFree(dD);
  return bad == 0 ? 0 : 1;
}
