#pragma once

// =====================================================================================
// blackhole / sm120 / NVFP4 (e2m1 + UE4M3 block-scale) x NVFP4 -> BF16 GEMM
// -------------------------------------------------------------------------------------
// 纯 CuTe 手写实现（不使用 CUTLASS 的 collective / device builder），目标架构 sm_120a
// （消费级 Blackwell, RTX 50xx）。计算 D = A x B：
//
//   * A: M x K，行主序（K 连续），元素 e2m1（4bit），每 16 个 K 元素共享一个 ue4m3 缩放因子 SFA
//   * B: N x K，K 连续（即 K x N 列主序），同样 e2m1 + ue4m3 SFB（NVFP4 格式）
//   * D: M x N，行主序，bfloat16；内部用 f32 累加
//
// 核心指令是 Blackwell 消费卡的块缩放 Tensor Core MMA：
//   mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col
//       .f32.e2m1.e2m1.f32.ue4m3
// 即每条指令完成 16x8x64 的 fp4 矩阵乘，硬件自动把 A/B 的每 16 个 K 元素乘上
// 各自的 ue4m3 缩放因子（SF 由寄存器提供，每线程一个 32bit 寄存器装 4 个 SF）。
//
// ------------------------------- 整体结构 -------------------------------------------
//
//   gmem --TMA(异步,128B swizzle)--> smem (5 级流水) --ldmatrix--> 寄存器 --MMA--> f32 累加
//                                                                              |
//   gmem <--TMA store-- smem (复用主循环的 smem) <--stmatrix-- bf16 转换 <------+
//
// * CTA tile: 128(M) x 128(N) x 128(K fp4)。每个 CTA 256 线程（8 个 warp），
//   warp 排布 4x2（M 方向 4 个、N 方向 2 个）。
// * 软件流水：Stages=5 级 smem 缓冲。每级一对 mbarrier：
//     - full[s]  : TMA 写完该级 smem 后翻相（transaction barrier，按字节数计数）
//     - empty[s] : 256 个线程都读完该级 smem 后翻相（每线程 arrive 一次）
//   没有专门的 producer warp——warp0 里用 elect_one_sync 选出 1 个线程负责发 TMA，
//   该线程同时也参与 MMA。sm120 没有 setmaxnreg，专门拆 producer warp 收益很小，
//   这种写法更简单，流水深度足够时性能相当。
// * K 维两级流水：一个 stage 内 K=128 拆成 2 个 k_block（每个 64，即一条 MMA 指令的 K）。
//   计算 k_block 的同时从 smem 预取 k_block+1 的寄存器（跨 stage 也预取），让
//   ldmatrix 和 MMA 重叠，Tensor Core 不空转。
// * epilogue：f32 累加器转 bf16，用 stmatrix 写入（与主循环复用的）smem，再整块
//   TMA store 回 gmem——写出是合并的、无 bank conflict。
//
// ------------------------------- 数据布局约定 ---------------------------------------
//
// fp4 打包：一个字节放 2 个 e2m1，K 偶数下标在低 4 位（与 cutlass 的子字节布局一致）。
//
// 缩放因子（SF）布局：与 CUTLASS 的 Sm1xxBlockScaledConfig 完全一致（不然没法和
// cuBLAS/CUTLASS 的数据互通）。逻辑上 SFA 是 (M, K/16) 的矩阵，物理上按
// 128 行 x 4 列 的块交错存放：块内偏移 = (m % 32) * 16 + (m / 32 % 4) * 4 + sf_k % 4，
// 块与块先沿 M 平铺、再沿 K 平铺。用本文件的 make_layout_SFA/SFB 生成布局即可，
// 宿主端按该 layout 填数据。
//
// 限制（为保持示例简洁，聚焦最常见形状）：M % 128 == 0, N % 128 == 0, K % 256 == 0。
// =====================================================================================

#include <cute/tensor.hpp>
#include <cute/arch/cluster_sm90.hpp>           // elect_one_sync
#include <cute/arch/copy_sm90_desc.hpp>         // mbarrier 辅助函数
#include <cute/arch/copy_sm90_tma.hpp>          // tma_store_fence/arrive/wait
#include <cute/atom/copy_traits_sm90_tma.hpp>   // make_tma_copy
#include <cute/atom/mma_atom.hpp>
#include <cute/atom/mma_traits_sm120.hpp>       // SM120 块缩放 MMA atom + traits
#include <cutlass/detail/sm100_blockscaled_layout.hpp>  // 规范 SF gmem 布局（与 CUTLASS 互通）

namespace blackhole::sm120 {

using namespace cute;

// =====================================================================================
// 1. 编译期配置：类型、TiledMMA、smem 布局、TMA —— 所有“选型”都集中在这里
// =====================================================================================
struct Nvfp4GemmTraits {
  // ---- 元素类型 ----
  using ElementA   = cutlass::float_e2m1_t;   // 4bit
  using ElementB   = cutlass::float_e2m1_t;
  using ElementSF  = cutlass::float_ue4m3_t;  // 无符号 e4m3 缩放因子
  using ElementD   = cutlass::bfloat16_t;
  using ElementAcc = float;

  static constexpr int SFVecSize = 16;        // NVFP4：每 16 个 K 元素一个 SF

  // ---- CTA tile 与流水级数 ----
  // 128x128x128(fp4)：A/B 各 8KB/级，SFA/SFB 各 1KB/级 => 18KB/级，5 级共 90KB，
  // 在 sm120 的 99KB smem 上限内（epilogue 的 32KB bf16 缓冲与主循环 smem 复用）。
  static constexpr int TileM  = 128;
  static constexpr int TileN  = 128;
  static constexpr int TileK  = 128;
  static constexpr int Stages = 5;
  using TileShape = Shape<Int<TileM>, Int<TileN>, Int<TileK>>;

  // ---- TiledMMA ----
  // atom：m16n8k64 块缩放 MMA（SF 向量长度 16，f32 累加）
  // atom 排布 4x2x1：M 方向 4 个 warp、N 方向 2 个 => 8 warp / 256 线程。
  // N 方向的置换 Layout<(8,2,2),(1,16,8)> 让一个 warp 拿到 SF 归约所需的全部列
  // （与 CUTLASS sm120 builder 的选择一致），TiledMMA tile = 128 x 32 x 64。
  using MmaAtom = SM120::BLOCKSCALED::SM120_16x8x64_TN_VS<
      ElementA, ElementB, ElementAcc, ElementSF, SFVecSize>;
  using AtomLayoutMNK = Layout<Shape<_4, _2, _1>>;
  using PermN = Layout<Shape<_8, _2, _2>, Stride<_1, _16, _8>>;
  using TiledMma = decltype(make_tiled_mma(
      MmaAtom{}, AtomLayoutMNK{}, Tile<Int<TileM>, PermN, _64>{}));

  static constexpr int NumThreads = 256;
  static_assert(size(TiledMma{}) == NumThreads);

  // MMA 的 K=64 内含 64/16 = 4 个 SF（对应 PTX 的 scale_vec::4X）
  static constexpr int MMA_NSF = 64 / SFVecSize;

  // ---- A/B 的 smem 布局 ----
  // smem 中 fp4 以 4bit 元素存放（uint4_t），K-major。一行 K=128 个 fp4 = 64 字节，
  // 选 64B swizzle（SW64）保证 ldmatrix 无 bank conflict。
  using SmemAtomAB   = UMMA::Layout_K_SW64_Atom<uint4_t>;
  using SmemLayoutA  = decltype(tile_to_shape(
      SmemAtomAB{}, Shape<Int<TileM>, Int<TileK>, Int<Stages>>{}));
  using SmemLayoutB  = decltype(tile_to_shape(
      SmemAtomAB{}, Shape<Int<TileN>, Int<TileK>, Int<Stages>>{}));

  // ---- SF 的 smem 布局 ----
  // 一个 tile 的 SFA 逻辑形状是 (M=128, K=128)，物理只存 128 x (128/16) = 1024 字节。
  // 布局公式来自 CUTLASS sm120 builder（与 gmem 的 128x4 块交错格式对应，TMA 直拷）：
  //   M 模式: (32,4,M/128)        步长 (16,4,512)
  //   K 模式: ((16,4),(1,K/64))   步长 ((0,1),(4,512*M/128))   [16 的步长为 0：同一
  //                                SF 在 16 个 K 元素上广播；用 filter_zeros 取实存]
  using SmemAtomSFA = Layout<
      Shape <Shape<_32, _4, Int<TileM / 128>>,
             Shape<Shape<Int<SFVecSize>, Int<MMA_NSF>>, Shape<_1, Int<TileK / 64>>>>,
      Stride<Stride<_16, _4, _512>,
             Stride<Stride<_0, _1>, Stride<_4, Int<TileM / 128 * 512>>>>>;
  using SmemAtomSFB = Layout<
      Shape <Shape<_32, _4, Int<TileN / 128>>,
             Shape<Shape<Int<SFVecSize>, Int<MMA_NSF>>, Shape<_1, Int<TileK / 64>>>>,
      Stride<Stride<_16, _4, _512>,
             Stride<Stride<_0, _1>, Stride<_4, Int<TileN / 128 * 512>>>>>;
  // 加上流水级模式（级间步长 = 单级实际存储量）
  using SmemLayoutSFA = decltype(make_layout(
      append(shape(SmemAtomSFA{}), Int<Stages>{}),
      append(stride(SmemAtomSFA{}), Int<size(filter_zeros(SmemAtomSFA{}))>{})));
  using SmemLayoutSFB = decltype(make_layout(
      append(shape(SmemAtomSFB{}), Int<Stages>{}),
      append(stride(SmemAtomSFB{}), Int<size(filter_zeros(SmemAtomSFB{}))>{})));

  // ---- smem -> 寄存器 copy atom ----
  // A/B 用 ldmatrix.x4（一次 4 个 8x8x16bit 矩阵）；SF 量很小，普通 LDS 即可。
  using SmemCopyAtomA  = Copy_Atom<SM75_U32x4_LDSM_N, uint4_t>;
  using SmemCopyAtomB  = Copy_Atom<SM75_U32x4_LDSM_N, uint4_t>;

  // ---- epilogue：bf16 输出经 smem 中转 ----
  // stmatrix 写 smem（SW128 swizzle，无 bank conflict），再整块 TMA store。
  using SmemLayoutD = decltype(tile_to_shape(
      UMMA::Layout_K_SW128_Atom<ElementD>{}, Shape<Int<TileM>, Int<TileN>>{}));
  using SmemCopyAtomD = Copy_Atom<SM90_U32x4_STSM_N, ElementD>;

  // ---- gmem 端 SF 布局（与 CUTLASS Sm1xxBlockScaledConfig 一致） ----
  using BlkScaledConfig = cutlass::detail::Sm1xxBlockScaledConfig<SFVecSize>;

  // ---- 每级 TMA 事务字节数（A + B + SFA + SFB），用于 full barrier 的 expect_tx ----
  static constexpr uint32_t TmaBytesPerStage = static_cast<uint32_t>(
      (TileM * TileK + TileN * TileK) / 2 +                 // fp4: 半字节/元素
      size(filter_zeros(SmemAtomSFA{})) +                   // SFA 实存字节
      size(filter_zeros(SmemAtomSFB{})));                   // SFB 实存字节

  // ---- smem 总体规划 ----
  // 主循环区与 epilogue 区复用同一块 smem（union）；barrier 放在 union 之外。
  struct SharedStorage {
    union {
      struct {
        alignas(1024) cute::ArrayEngine<uint4_t,   cute::cosize_v<SmemLayoutA>>   A;
        alignas(1024) cute::ArrayEngine<uint4_t,   cute::cosize_v<SmemLayoutB>>   B;
        alignas(128)  cute::ArrayEngine<ElementSF, cute::cosize_v<SmemLayoutSFA>> SFA;
        alignas(128)  cute::ArrayEngine<ElementSF, cute::cosize_v<SmemLayoutSFB>> SFB;
      } mainloop;
      struct {
        alignas(1024) cute::ArrayEngine<ElementD, cute::cosize_v<SmemLayoutD>> D;
      } epilogue;
    } tensors;
    alignas(8) uint64_t full_barrier[Stages];   // TMA 写完 -> 可读
    alignas(8) uint64_t empty_barrier[Stages];  // 全部线程读完 -> 可写
  };
  static_assert(sizeof(SharedStorage) <= 101376, "exceeds sm120 smem capacity (99KB)");
};

// =====================================================================================
// 2. 缩放因子的 TiledMMA 分块辅助函数
// -------------------------------------------------------------------------------------
// SF 不是 TiledMMA 的原生操作数（atom 的 SFALayout/SFBLayout 带 0 步长的广播模式），
// cute 没有现成的 partition_A 可用，这几个函数把 SF 张量按 MMA 的线程布局切开。
// 移植自 CUTLASS sm120 blockscaled collective（纯 cute 布局代数，BSD-3 许可）。
// =====================================================================================
namespace detail {

template <class SFATensor, class Atom, class TiledThr, class TiledPerm>
CUTE_HOST_DEVICE constexpr auto
thrfrg_SFA(SFATensor&& sfatensor, TiledMMA<Atom, TiledThr, TiledPerm>& mma) {
  CUTE_STATIC_ASSERT_V(rank(sfatensor) >= Int<2>{});
  using AtomShape_MNK    = typename Atom::Shape_MNK;
  using AtomLayoutSFA_TV = typename Atom::Traits::SFALayout;

  auto permutation_mnk = TiledPerm{};
  auto thr_layout_vmnk = mma.get_thr_layout_vmnk();

  // 先按 TiledMMA 的 (M,K) 置换重排，再切成 atom 大小的块，
  // 然后用 atom 的 SFA TV 布局把 (M,K) 变成 (线程, 值)，最后按线程网格切块。
  auto t_tile   = make_tile(get<0>(permutation_mnk), get<2>(permutation_mnk));
  auto t_tensor = logical_divide(sfatensor, t_tile);                  // (PermM,PermK)
  auto a_tile   = make_tile(make_layout(size<0>(AtomShape_MNK{})),
                            make_layout(size<2>(AtomShape_MNK{})));
  auto a_tensor = zipped_divide(t_tensor, a_tile);                    // ((AtomM,AtomK),(RestM,RestK))
  auto tv_tensor = a_tensor.compose(AtomLayoutSFA_TV{}, _);           // ((ThrV,FrgV),(RestM,RestK))
  auto thr_tile = make_tile(_, make_tile(make_layout(size<1>(thr_layout_vmnk)),
                                         make_layout(size<3>(thr_layout_vmnk))));
  auto thr_tensor = zipped_divide(tv_tensor, thr_tile);  // ((ThrV,(ThrM,ThrK)),(FrgV,(RestM,RestK)))
  return thr_tensor;
}

template <class SFBTensor, class Atom, class TiledThr, class TiledPerm>
CUTE_HOST_DEVICE constexpr auto
thrfrg_SFB(SFBTensor&& sfbtensor, TiledMMA<Atom, TiledThr, TiledPerm>& mma) {
  CUTE_STATIC_ASSERT_V(rank(sfbtensor) >= Int<2>{});
  using AtomShape_MNK    = typename Atom::Shape_MNK;
  using AtomLayoutSFB_TV = typename Atom::Traits::SFBLayout;

  auto permutation_mnk = TiledPerm{};
  auto thr_layout_vmnk = mma.get_thr_layout_vmnk();

  auto t_tile   = make_tile(get<1>(permutation_mnk), get<2>(permutation_mnk));
  auto t_tensor = logical_divide(sfbtensor, t_tile);                  // (PermN,PermK)
  auto b_tile   = make_tile(make_layout(size<1>(AtomShape_MNK{})),
                            make_layout(size<2>(AtomShape_MNK{})));
  auto b_tensor = zipped_divide(t_tensor, b_tile);                    // ((AtomN,AtomK),(RestN,RestK))
  auto tv_tensor = b_tensor.compose(AtomLayoutSFB_TV{}, _);           // ((ThrV,FrgV),(RestN,RestK))
  auto thr_tile = make_tile(_, make_tile(make_layout(size<2>(thr_layout_vmnk)),
                                         make_layout(size<3>(thr_layout_vmnk))));
  auto thr_tensor = zipped_divide(tv_tensor, thr_tile);  // ((ThrV,(ThrN,ThrK)),(FrgV,(RestN,RestK)))
  return thr_tensor;
}

// 为当前线程分配 SF 的寄存器 fragment（形状与 partition_A/B 的 (MMA,MMA_M,MMA_K) 对齐）
template <class SFATensor, class ThrMma>
CUTE_HOST_DEVICE constexpr auto
partition_fragment_SFA(SFATensor&& sfatensor, ThrMma& thread_mma) {
  using ValTypeSF = typename ThrMma::Atom::Traits::ValTypeSF;
  auto thr_tensor = make_tensor(static_cast<SFATensor&&>(sfatensor).data(),
                                thrfrg_SFA(sfatensor.layout(), thread_mma));
  auto thr_vmnk = thread_mma.thr_vmnk_;
  auto thr_vmk  = make_coord(get<0>(thr_vmnk), make_coord(get<1>(thr_vmnk), get<3>(thr_vmnk)));
  auto partition_SFA =
      thr_tensor(thr_vmk, make_coord(_, repeat<rank<1, 1>(thr_tensor)>(_)));
  return make_fragment_like<ValTypeSF>(partition_SFA);
}

template <class SFBTensor, class ThrMma>
CUTE_HOST_DEVICE constexpr auto
partition_fragment_SFB(SFBTensor&& sfbtensor, ThrMma& thread_mma) {
  using ValTypeSF = typename ThrMma::Atom::Traits::ValTypeSF;
  auto thr_tensor = make_tensor(static_cast<SFBTensor&&>(sfbtensor).data(),
                                thrfrg_SFB(sfbtensor.layout(), thread_mma));
  auto thr_vmnk = thread_mma.thr_vmnk_;
  auto thr_vnk  = make_coord(get<0>(thr_vmnk), make_coord(get<2>(thr_vmnk), get<3>(thr_vmnk)));
  auto partition_SFB =
      thr_tensor(thr_vnk, make_coord(_, repeat<rank<1, 1>(thr_tensor)>(_)));
  return make_fragment_like<ValTypeSF>(partition_SFB);
}

// (thr_idx, val) -> (M,K) / (N,K) 的 TV 布局，用来构造 SF 的 smem->寄存器 tiled copy
template <class TiledMma>
CUTE_HOST_DEVICE constexpr auto get_layoutSFA_TV(TiledMma& mma) {
  auto tile_shape_mnk = tile_shape(mma);
  auto ref_A = make_layout(make_shape(size<0>(tile_shape_mnk), size<2>(tile_shape_mnk)));
  auto thr_layout_vmnk = mma.get_thr_layout_vmnk();
  auto atile = make_tile(_, make_tile(make_layout(make_shape(size<1>(thr_layout_vmnk),
                                                             size<2>(thr_layout_vmnk)),
                                                  make_stride(Int<1>{}, Int<0>{})),
                                      _));
  auto thridx_2_thrid = right_inverse(thr_layout_vmnk);
  return thrfrg_SFA(ref_A, mma).compose(atile, _).compose(thridx_2_thrid, _);
}

template <class TiledMma>
CUTE_HOST_DEVICE constexpr auto get_layoutSFB_TV(TiledMma& mma) {
  auto tile_shape_mnk = tile_shape(mma);
  auto ref_B = make_layout(make_shape(size<1>(tile_shape_mnk), size<2>(tile_shape_mnk)));
  auto thr_layout_vmnk = mma.get_thr_layout_vmnk();
  auto btile = make_tile(_, make_tile(make_layout(make_shape(size<1>(thr_layout_vmnk),
                                                             size<2>(thr_layout_vmnk)),
                                                  make_stride(Int<0>{}, Int<1>{})),
                                      _));
  auto thridx_2_thrid = right_inverse(thr_layout_vmnk);
  return thrfrg_SFB(ref_B, mma).compose(btile, _).compose(thridx_2_thrid, _);
}

}  // namespace detail

// =====================================================================================
// 3. 宿主端布局辅助：构造 gmem 张量布局（示例程序填数据时也用它们）
// =====================================================================================
// SFA 布局：逻辑 (M, K)（含 16 广播模式），物理为 128x4 块交错。M/K 必须是 128/64 的倍数。
CUTE_HOST_DEVICE auto make_layout_SFA(int M, int K) {
  return Nvfp4GemmTraits::BlkScaledConfig::tile_atom_to_shape_SFA(make_shape(M, 0, K));
}
CUTE_HOST_DEVICE auto make_layout_SFB(int N, int K) {
  return Nvfp4GemmTraits::BlkScaledConfig::tile_atom_to_shape_SFB(make_shape(0, N, K));
}
// 一个 (M, K) 问题需要的 SF 存储字节数
CUTE_HOST_DEVICE size_t sf_storage_bytes(int MN, int K) {
  return size_t(MN) * (K / Nvfp4GemmTraits::SFVecSize);
}

#if defined(__CUDACC__)

// =====================================================================================
// 4. 设备端 kernel
// =====================================================================================
template <class KT>
struct Nvfp4GemmParams {
  // TMA copy 对象内含 TMA descriptor，必须以 __grid_constant__ 形参进 kernel
  using LayoutSFAGmem = decltype(make_layout_SFA(0, 0));
  using LayoutSFBGmem = decltype(make_layout_SFB(0, 0));

  using TmaA = decltype(make_tma_copy(
      SM90_TMA_LOAD{},
      make_tensor(recast_ptr<typename KT::ElementA>((void const*)nullptr),
                  make_layout(make_shape(0, 0), make_stride(0, _1{}))),
      typename KT::SmemLayoutA{}(_, _, Int<0>{})));
  using TmaB = decltype(make_tma_copy(
      SM90_TMA_LOAD{},
      make_tensor(recast_ptr<typename KT::ElementB>((void const*)nullptr),
                  make_layout(make_shape(0, 0), make_stride(0, _1{}))),
      typename KT::SmemLayoutB{}(_, _, Int<0>{})));
  using TmaSFA = decltype(make_tma_copy(
      SM90_TMA_LOAD{},
      make_tensor((typename KT::ElementSF const*)nullptr, LayoutSFAGmem{}),
      typename KT::SmemLayoutSFA{}(_, _, Int<0>{})));
  using TmaSFB = decltype(make_tma_copy(
      SM90_TMA_LOAD{},
      make_tensor((typename KT::ElementSF const*)nullptr, LayoutSFBGmem{}),
      typename KT::SmemLayoutSFB{}(_, _, Int<0>{})));
  using TmaD = decltype(make_tma_copy(
      SM90_TMA_STORE{},
      make_tensor((typename KT::ElementD*)nullptr,
                  make_layout(make_shape(0, 0), make_stride(0, _1{}))),
      typename KT::SmemLayoutD{}));

  TmaA   tma_a;
  TmaB   tma_b;
  TmaSFA tma_sfa;
  TmaSFB tma_sfb;
  TmaD   tma_d;
  LayoutSFAGmem layout_sfa;
  LayoutSFBGmem layout_sfb;
  int M, N, K;
  int tiles_m, tiles_n;
};

template <class KT>
__global__ void __launch_bounds__(KT::NumThreads, 1)
nvfp4_gemm_kernel(const __grid_constant__ Nvfp4GemmParams<KT> params) {
#if defined(CUTE_ARCH_MXF4NVF4_4X_UE4M3_MMA_ENABLED) || (__CUDA_ARCH__ >= 1200)
  constexpr int Stages = KT::Stages;
  using SharedStorage  = typename KT::SharedStorage;

  extern __shared__ char smem_raw[];
  SharedStorage& smem = *reinterpret_cast<SharedStorage*>(smem_raw);

  const int thread_idx = threadIdx.x;
  const int warp_idx   = thread_idx / 32;
  // warp0 中选举一个 lane 作为 TMA producer（它同时也照常参与 MMA）
  const bool is_producer = (warp_idx == 0) && cute::elect_one_sync();

  // ---------------- CTA -> 输出 tile 的映射（沿 M 每 8 个 CTA 一组的光栅化，提高 L2 命中） ----------------
  constexpr int RasterM = 8;
  int m_coord, n_coord;
  {
    int group_size = RasterM * params.tiles_n;
    int group      = blockIdx.x / group_size;
    int rem        = blockIdx.x % group_size;
    int gm         = min(RasterM, params.tiles_m - group * RasterM);  // 尾组
    m_coord = group * RasterM + rem % gm;
    n_coord = rem / gm;
  }

  const int k_tiles = params.K / KT::TileK;

  // ---------------- barrier 初始化 ----------------
  if (thread_idx == 0) {
    CUTE_UNROLL
    for (int s = 0; s < Stages; ++s) {
      cute::initialize_barrier(smem.full_barrier[s],  /*arrive_count=*/1);
      cute::initialize_barrier(smem.empty_barrier[s], /*arrive_count=*/KT::NumThreads);
    }
  }
  // mbarrier 初始化对 TMA（async proxy）可见
  asm volatile("fence.mbarrier_init.release.cluster;\n" ::);
  __syncthreads();

  // ---------------- gmem / smem 张量 ----------------
  Tensor mA   = params.tma_a.get_tma_tensor(make_shape(params.M, params.K));     // (M,K)
  Tensor mB   = params.tma_b.get_tma_tensor(make_shape(params.N, params.K));     // (N,K)
  Tensor mSFA = params.tma_sfa.get_tma_tensor(shape(params.layout_sfa));
  Tensor mSFB = params.tma_sfb.get_tma_tensor(shape(params.layout_sfb));

  auto cta_tiler = typename KT::TileShape{};
  Tensor gA   = local_tile(mA,   cta_tiler, make_coord(m_coord, _, _), Step<_1, X, _1>{})(_, _, 0, _);  // (TileM,TileK,k)
  Tensor gB   = local_tile(mB,   cta_tiler, make_coord(_, n_coord, _), Step<X, _1, _1>{})(_, _, 0, _);  // (TileN,TileK,k)
  Tensor gSFA = local_tile(mSFA, cta_tiler, make_coord(m_coord, _, _), Step<_1, X, _1>{})(_, _, 0, _);
  Tensor gSFB = local_tile(mSFB, cta_tiler, make_coord(_, n_coord, _), Step<X, _1, _1>{})(_, _, 0, _);

  Tensor sA   = make_tensor(make_smem_ptr(smem.tensors.mainloop.A.begin()),   typename KT::SmemLayoutA{});
  Tensor sB   = make_tensor(make_smem_ptr(smem.tensors.mainloop.B.begin()),   typename KT::SmemLayoutB{});
  Tensor sSFA = make_tensor(make_smem_ptr(smem.tensors.mainloop.SFA.begin()), typename KT::SmemLayoutSFA{});
  Tensor sSFB = make_tensor(make_smem_ptr(smem.tensors.mainloop.SFB.begin()), typename KT::SmemLayoutSFB{});

  // ---------------- TMA 分块（producer 视角） ----------------
  auto tma_slice_a   = params.tma_a.get_slice(_0{});
  auto tma_slice_b   = params.tma_b.get_slice(_0{});
  auto tma_slice_sfa = params.tma_sfa.get_slice(_0{});
  auto tma_slice_sfb = params.tma_sfb.get_slice(_0{});
  Tensor tAgA   = tma_slice_a.partition_S(gA);      // (TMA,TMA_M,TMA_K,k)
  Tensor tAsA   = tma_slice_a.partition_D(sA);      // (TMA,TMA_M,TMA_K,Stage)
  Tensor tBgB   = tma_slice_b.partition_S(gB);
  Tensor tBsB   = tma_slice_b.partition_D(sB);
  Tensor tAgSFA = tma_slice_sfa.partition_S(gSFA);
  Tensor tAsSFA = tma_slice_sfa.partition_D(sSFA);
  Tensor tBgSFB = tma_slice_sfb.partition_S(gSFB);
  Tensor tBsSFB = tma_slice_sfb.partition_D(sSFB);

  // 发起第 k_tile 块到第 stage 级缓冲的 4 路 TMA（单线程调用）
  auto issue_tma = [&](int k_tile, int stage) {
    cute::set_barrier_transaction_bytes(smem.full_barrier[stage], KT::TmaBytesPerStage);
    copy(params.tma_a.with(smem.full_barrier[stage]),   tAgA(_, _, _, k_tile),   tAsA(_, _, _, stage));
    copy(params.tma_b.with(smem.full_barrier[stage]),   tBgB(_, _, _, k_tile),   tBsB(_, _, _, stage));
    copy(params.tma_sfa.with(smem.full_barrier[stage]), tAgSFA(_, _, _, k_tile), tAsSFA(_, _, _, stage));
    copy(params.tma_sfb.with(smem.full_barrier[stage]), tBgSFB(_, _, _, k_tile), tBsSFB(_, _, _, stage));
  };

  // ---------------- 流水线预热：先填满全部 Stages 级 ----------------
  if (is_producer) {
    cute::prefetch_tma_descriptor(params.tma_a.get_tma_descriptor());
    cute::prefetch_tma_descriptor(params.tma_b.get_tma_descriptor());
    cute::prefetch_tma_descriptor(params.tma_sfa.get_tma_descriptor());
    cute::prefetch_tma_descriptor(params.tma_sfb.get_tma_descriptor());
    cute::prefetch_tma_descriptor(params.tma_d.get_tma_descriptor());
    for (int s = 0; s < min(Stages, k_tiles); ++s) {
      issue_tma(/*k_tile=*/s, /*stage=*/s);
    }
  }

  // ---------------- MMA 侧分块 ----------------
  typename KT::TiledMma tiled_mma;
  auto thread_mma = tiled_mma.get_thread_slice(thread_idx);

  // 寄存器 fragment：一次装下整个 stage 的 K（=2 个 k_block），相邻 k_block 双缓冲
  Tensor tCrA   = thread_mma.partition_fragment_A(sA(_, _, Int<0>{}));            // (MMA,MMA_M,K_BLK)
  Tensor tCrB   = thread_mma.partition_fragment_B(sB(_, _, Int<0>{}));            // (MMA,MMA_N,K_BLK)
  Tensor tCrSFA = detail::partition_fragment_SFA(sSFA(_, _, Int<0>{}), thread_mma);
  Tensor tCrSFB = detail::partition_fragment_SFB(sSFB(_, _, Int<0>{}), thread_mma);

  // A/B 的 smem->寄存器 tiled copy（ldmatrix）
  auto tiled_copy_A = make_tiled_copy_A(typename KT::SmemCopyAtomA{}, tiled_mma);
  auto thr_copy_A   = tiled_copy_A.get_thread_slice(thread_idx);
  Tensor tCsA       = thr_copy_A.partition_S(as_position_independent_swizzle_tensor(sA));
  Tensor tCrA_view  = thr_copy_A.retile_D(tCrA);

  auto tiled_copy_B = make_tiled_copy_B(typename KT::SmemCopyAtomB{}, tiled_mma);
  auto thr_copy_B   = tiled_copy_B.get_thread_slice(thread_idx);
  Tensor tCsB       = thr_copy_B.partition_S(as_position_independent_swizzle_tensor(sB));
  Tensor tCrB_view  = thr_copy_B.retile_D(tCrB);

  // SF 的 smem->寄存器 copy（量小，普通 LDS）
  auto tile_mnk = tile_shape(tiled_mma);
  auto tiled_copy_SFA = make_tiled_copy_impl(
      Copy_Atom<UniversalCopy<typename KT::ElementSF>, typename KT::ElementSF>{},
      detail::get_layoutSFA_TV(tiled_mma),
      make_shape(size<0>(tile_mnk), size<2>(tile_mnk)));
  auto thr_copy_SFA  = tiled_copy_SFA.get_thread_slice(thread_idx);
  Tensor tCsSFA      = thr_copy_SFA.partition_S(as_position_independent_swizzle_tensor(sSFA));
  Tensor tCrSFA_view = thr_copy_SFA.retile_D(tCrSFA);

  auto tiled_copy_SFB = make_tiled_copy_impl(
      Copy_Atom<UniversalCopy<typename KT::ElementSF>, typename KT::ElementSF>{},
      detail::get_layoutSFB_TV(tiled_mma),
      make_shape(size<1>(tile_mnk), size<2>(tile_mnk)));
  auto thr_copy_SFB  = tiled_copy_SFB.get_thread_slice(thread_idx);
  Tensor tCsSFB      = thr_copy_SFB.partition_S(as_position_independent_swizzle_tensor(sSFB));
  Tensor tCrSFB_view = thr_copy_SFB.retile_D(tCrSFB);

  // f32 累加器
  Tensor accum = partition_fragment_C(tiled_mma, take<0, 2>(cta_tiler));          // (MMA,MMA_M,MMA_N)
  clear(accum);

  constexpr int K_BLOCK_MAX = size<2>(tCrA);   // = TileK / 64 = 2

  // 从第 stage 级 smem 取第 k_block 段到寄存器
  auto copy_kblock = [&](auto k_block, int stage) {
    copy(tiled_copy_A, tCsA(_, _, k_block, stage), tCrA_view(_, _, k_block));
    copy(tiled_copy_B, tCsB(_, _, k_block, stage), tCrB_view(_, _, k_block));
    copy(tCsSFA(_, _, k_block, stage), tCrSFA_view(_, _, k_block));
    copy(tCsSFB(_, _, k_block, stage), tCrSFB_view(_, _, k_block));
    // 注：mxf4nvf4 双 fp4 路径不需要 f8f6f4 路径的 <<2 移位修正
  };
  // 用第 k_block 段做 MMA：A/B 与各自 SF zip 在一起喂给 cute::gemm，
  // mma_unpack 会拆开并填好 mma.block_scale 指令的全部操作数
  auto gemm_kblock = [&](auto k_block) {
    cute::gemm(tiled_mma,
               make_zip_tensor(tCrA(_, _, k_block), tCrSFA(_, _, k_block)),
               make_zip_tensor(tCrB(_, _, k_block), tCrSFB(_, _, k_block)),
               accum);
  };

  // ---------------- 主循环 ----------------
  // 相位说明：mbarrier 初相为 0，第一次翻相后 wait(parity=0) 通过。
  // 消费第 t 个 k_tile 用 stage = t % Stages、parity = (t/Stages) & 1。
  int read_stage = 0, read_phase = 0;

  cute::wait_barrier(smem.full_barrier[0], 0);   // 等第 0 级数据就绪
  copy_kblock(Int<0>{}, /*stage=*/0);            // 预取第一段寄存器

  CUTE_NO_UNROLL
  for (int k_tile = 0; k_tile < k_tiles; ++k_tile) {
    int next_stage = (read_stage + 1 == Stages) ? 0 : read_stage + 1;
    int next_phase = (read_stage + 1 == Stages) ? (read_phase ^ 1) : read_phase;
    bool has_next  = (k_tile + 1 < k_tiles);

    for_each(make_int_sequence<K_BLOCK_MAX>{}, [&](auto k_block) {
      if constexpr (k_block == K_BLOCK_MAX - 1) {
        // 本级 smem 在上一轮 copy_kblock 后已全部进寄存器：
        // 1) 跨 stage 预取下一级的第 0 段
        if (has_next) {
          cute::wait_barrier(smem.full_barrier[next_stage], next_phase);
          copy_kblock(Int<0>{}, next_stage);
        }
        // 2) 本线程宣布读完本级（释放给 producer 复用）
        cute::arrive_barrier(smem.empty_barrier[read_stage]);
        // 3) producer 等到全员读完后，立刻把本级重新填上 Stages 个 tile 之后的数据
        if (is_producer && (k_tile + Stages) < k_tiles) {
          cute::wait_barrier(smem.empty_barrier[read_stage], read_phase);
          issue_tma(k_tile + Stages, read_stage);
        }
      } else {
        copy_kblock(k_block + Int<1>{}, read_stage);   // 段内预取下一段
      }
      gemm_kblock(k_block);
    });

    read_stage = next_stage;
    read_phase = next_phase;
  }

  // ---------------- epilogue：f32 -> bf16，stmatrix 进 smem，TMA store 回 gmem ----------------
  // 主循环 smem 即将被 D 复用，先确保所有线程的 MMA 全部完成
  __syncthreads();

  Tensor sD = make_tensor(make_smem_ptr(smem.tensors.epilogue.D.begin()), typename KT::SmemLayoutD{});

  // f32 -> bf16（保持 fragment 布局逐元素转换）
  Tensor tDrD = make_fragment_like<typename KT::ElementD>(accum);
  CUTE_UNROLL
  for (int i = 0; i < size(accum); ++i) {
    tDrD(i) = static_cast<typename KT::ElementD>(accum(i));
  }

  auto tiled_copy_D = make_tiled_copy_C(typename KT::SmemCopyAtomD{}, tiled_mma);
  auto thr_copy_D   = tiled_copy_D.get_thread_slice(thread_idx);
  Tensor tDrD_view  = thr_copy_D.retile_S(tDrD);
  Tensor tDsD       = thr_copy_D.partition_D(as_position_independent_swizzle_tensor(sD));
  copy(tiled_copy_D, tDrD_view, tDsD);

  // stmatrix 的 smem 写对 TMA（async proxy）可见
  cute::tma_store_fence();
  __syncthreads();

  if (is_producer) {
    Tensor mD = params.tma_d.get_tma_tensor(make_shape(params.M, params.N));      // (M,N)
    Tensor gD = local_tile(mD, Shape<Int<KT::TileM>, Int<KT::TileN>>{}, make_coord(m_coord, n_coord));
    auto tma_slice_d = params.tma_d.get_slice(_0{});
    Tensor tDsD_tma = tma_slice_d.partition_S(sD);
    Tensor tDgD_tma = tma_slice_d.partition_D(gD);
    copy(params.tma_d, tDsD_tma, tDgD_tma);
    cute::tma_store_arrive();
    cute::tma_store_wait<0>();
  }
#endif
}

// =====================================================================================
// 5. 宿主端 launch
// =====================================================================================
//   A_packed : e2m1 打包数据（2 元素/字节，低 4 位在前），行主序 M x K
//   SFA      : ue4m3 缩放因子，布局 = make_layout_SFA(M, K)
//   B_packed : e2m1 打包数据，N x K（K 连续，即列主序的 K x N）
//   SFB      : ue4m3 缩放因子，布局 = make_layout_SFB(N, K)
//   D        : bf16 输出，行主序 M x N
inline cudaError_t nvfp4_bf16_gemm(int M, int N, int K,
                                   void const* A_packed, void const* SFA,
                                   void const* B_packed, void const* SFB,
                                   void* D, cudaStream_t stream = nullptr) {
  using KT = Nvfp4GemmTraits;

  if (M % KT::TileM != 0 || N % KT::TileN != 0 || K % (2 * KT::TileK) != 0) {
    return cudaErrorInvalidValue;   // 本示例实现仅支持整 tile 形状（K 还需偶数个 tile）
  }

  // gmem 张量（fp4 用子字节指针 + 元素粒度布局描述，make_tma_copy 内部会转成字节版 TMA）
  auto layout_sfa = make_layout_SFA(M, K);
  auto layout_sfb = make_layout_SFB(N, K);

  Tensor tensor_a = make_tensor(recast_ptr<KT::ElementA>(A_packed),
                                make_layout(make_shape(M, K), make_stride(K, _1{})));
  Tensor tensor_b = make_tensor(recast_ptr<KT::ElementB>(B_packed),
                                make_layout(make_shape(N, K), make_stride(K, _1{})));
  Tensor tensor_sfa = make_tensor(static_cast<KT::ElementSF const*>(SFA), layout_sfa);
  Tensor tensor_sfb = make_tensor(static_cast<KT::ElementSF const*>(SFB), layout_sfb);
  Tensor tensor_d = make_tensor(static_cast<KT::ElementD*>(D),
                                make_layout(make_shape(M, N), make_stride(N, _1{})));

  Nvfp4GemmParams<KT> params;
  params.tma_a   = make_tma_copy(SM90_TMA_LOAD{},  tensor_a,   KT::SmemLayoutA{}(_, _, Int<0>{}));
  params.tma_b   = make_tma_copy(SM90_TMA_LOAD{},  tensor_b,   KT::SmemLayoutB{}(_, _, Int<0>{}));
  params.tma_sfa = make_tma_copy(SM90_TMA_LOAD{}, tensor_sfa, KT::SmemLayoutSFA{}(_, _, Int<0>{}));
  params.tma_sfb = make_tma_copy(SM90_TMA_LOAD{}, tensor_sfb, KT::SmemLayoutSFB{}(_, _, Int<0>{}));
  params.tma_d   = make_tma_copy(SM90_TMA_STORE{}, tensor_d,   KT::SmemLayoutD{});
  params.layout_sfa = layout_sfa;
  params.layout_sfb = layout_sfb;
  params.M = M; params.N = N; params.K = K;
  params.tiles_m = M / KT::TileM;
  params.tiles_n = N / KT::TileN;

  constexpr int smem_bytes = int(sizeof(typename KT::SharedStorage));
  auto* kernel = &nvfp4_gemm_kernel<KT>;
  static bool attr_set = false;     // 动态 smem 超过 48KB 需要显式开权限（每进程一次）
  if (!attr_set) {
    cudaError_t err = cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
    if (err != cudaSuccess) { return err; }
    attr_set = true;
  }

  dim3 grid(params.tiles_m * params.tiles_n, 1, 1);
  dim3 block(KT::NumThreads, 1, 1);
  kernel<<<grid, block, smem_bytes, stream>>>(params);
  return cudaGetLastError();
}

#endif  // __CUDACC__

}  // namespace blackhole::sm120
