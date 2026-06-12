# NVFP4 × NVFP4 → BF16 GEMM 示例

演示 `blackhole::sm120::nvfp4_bf16_gemm`——一个纯 CuTe 手写的、面向消费级
Blackwell（sm_120，RTX 50 系）的块缩放 NVFP4 GEMM。算子实现见
[`include/blackhole/sm120/nvfp4_gemm.hpp`](../include/blackhole/sm120/nvfp4_gemm.hpp)，
里面有逐段中文注释。

计算：`D = A × B`
- A：`M×K`，行主序，NVFP4（`e2m1` 4bit + 每 16 个元素一个 `ue4m3` 缩放因子 SFA）
- B：`N×K`，K 连续，NVFP4（同上，SFB）
- D：`M×N`，行主序，`bfloat16`；内部 `f32` 累加

核心指令是 Blackwell 的块缩放 Tensor Core MMA
`mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64...`，
一条指令做 16×8×64 的 fp4 矩阵乘并自动套用缩放因子。

## 环境要求

- **GPU**：compute capability 12.0（sm_120，GeForce RTX 50 系）。该 MMA 指令
  仅这一代消费卡支持，**在其它架构上无法运行**。
- **CUDA Toolkit ≥ 12.8**（首个支持 sm_120 的版本；提供 `mma...mxf4nvf4` 指令）。
- 已初始化 CUTLASS 子模块：`git submodule update --init --recursive`。

## 编译

从仓库根目录：

```bash
nvcc -std=c++17 -O3 -arch=sm_120a \
     --expt-relaxed-constexpr \
     -I include \
     -I third_party/cutlass/include \
     -I third_party/cutlass/tools/util/include \
     example/nvfp4_bf16_gemm.cu -o nvfp4_bf16_gemm
```

- `-arch=sm_120a`：必须带 `a`（architecture-specific），块缩放 MMA 的 PTX 是
  arch-conditional 的，普通 `sm_120` 不会启用 `CUTE_ARCH_MXF4NVF4_4X_UE4M3_MMA_ENABLED`。
- 动态共享内存约 91 KB，已在 host 端用 `cudaFuncSetAttribute` 申请，无需额外操作。

## 运行

```bash
./nvfp4_bf16_gemm                 # 默认 4096 x 4096 x 4096
./nvfp4_bf16_gemm 8192 8192 4096  # 自定义 M N K（要求 M%128=0, N%128=0, K%256=0）
```

输出示例：

```
NVFP4 x NVFP4 -> BF16 GEMM   M=4096 N=4096 K=4096
Verifying (sampled rows)...
Checked 256 elems, max_rel=0.0123, mismatches=0 -> PASSED
Avg 0.142 ms   967.3 TFLOP/s
```

程序会随机生成 NVFP4 输入、抽样跑 CPU 参考做正确性校验，再计时报告 TFLOP/s。

## 用 ncu 采样 Tensor Core 利用率

[Nsight Compute](https://developer.nvidia.com/nsight-compute)（`ncu`）是看
Tensor Core 占用最直接的工具。

### 1. 快速看 Tensor (MMA) 流水线利用率

```bash
sudo ncu --launch-skip 1 --launch-count 1 \
  --metrics \
sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active,\
sm__cycles_active.avg.pct_of_peak_sustained_elapsed,\
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__throughput.avg.pct_of_peak_sustained_elapsed \
  ./nvfp4_bf16_gemm 4096 4096 4096
```

关键指标：

| 指标 | 含义 |
| --- | --- |
| `sm__pipe_tensor_op_hmma_cycles_active...pct_of_peak_sustained_active` | **Tensor Core 利用率**——这条 MMA 流水线活跃周期占峰值的百分比，越接近 100% 越好 |
| `sm__throughput...` | SM 整体计算吞吐占比 |
| `dram__throughput...` | 显存带宽占比；GEMM 算力受限时这一项应远低于 SM |

> `--launch-skip 1` 跳过第一次（预热）launch，采第 2 次更稳。
> 在共享/无 sudo 环境用 `ncu`，需要管理员开启 GPU 计数器权限
> （见 NVIDIA ERR_NVGPUCTRPERM 文档）。

### 2. 跑内置的 “Compute Workload Analysis” section（看各流水线占用条形图）

```bash
sudo ncu --set full --launch-skip 1 --launch-count 1 \
  -o nvfp4_gemm_report \
  ./nvfp4_bf16_gemm 4096 4096 4096
```

生成 `nvfp4_gemm_report.ncu-rep`，用 `ncu-ui` 打开，看
**Compute Workload Analysis**：其中 `Tensor (FP)` / `LSU` / `ALU` 等
pipe 的 utilization 条。Tensor 条应是最高的那根。

### 3. 只关心一个数（脚本里抓利用率）

```bash
sudo ncu --launch-skip 1 --launch-count 1 --csv \
  --metrics sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active \
  ./nvfp4_bf16_gemm 4096 4096 4096 | tail -1
```

## 调优提示（想把 Tensor Core 利用率打更高）

`include/blackhole/sm120/nvfp4_gemm.hpp` 里 `Nvfp4GemmTraits` 集中了所有可调项：

- **`Stages`**：流水级数。smem 够的话加级数能更好地把 TMA 延迟藏在计算后面。
- **`TileM/TileN/TileK`**：CTA tile。大 tile 提高算术强度（减少 SF/重复加载），
  但占更多 smem、降低 occupancy——需要按 M/N/K 实测权衡。
- **`AtomLayoutMNK`**（warp 排布）与 **光栅化顺序 `RasterM`**：影响 L2 命中和
  尾部 wave 均衡。
- 主循环里 K 维双缓冲（段内预取 + 跨 stage 预取）确保 `ldmatrix` 与 MMA 重叠，
  这是让 Tensor Core 不空转的关键路径。

> 本示例为单一最优形状（M、N 为 128 的倍数、K 为 256 的倍数）做了固定 tile，
> 旨在作为学习 / 起点；通用形状需要补尾块（predication）与多组 tile 启发式选择。
