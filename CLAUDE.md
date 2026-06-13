# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`blackhole` is a deliberately minimal, **header-first** collection of GPU kernels
for NVIDIA **Blackwell** GPUs, written **directly against CuTe** primitives
(layouts, tensors, copy/MMA atoms, TMA, mbarriers) rather than the higher-level
CUTLASS collective/device builder APIs (`CollectiveBuilder`, `GemmUniversal`).
The whole point is fine-grained control of the Blackwell datapath, so when adding
an operator, hand-write the CuTe mainloop/epilogue instead of reaching for the
builders. Keep the repo lean — add directories (tests/, benchmarks/, etc.) only
when something concrete needs them.

Two SM families are targeted and kept in separate subtrees because their tensor
cores differ materially:
- **`sm_100`** — datacenter Blackwell (B100/B200), `tcgen05` 5th-gen tensor cores
- **`sm_120`** — consumer Blackwell (RTX 50xx), warp-level block-scaled MMA
  (`mma.sync...mxf4nvf4` / `mxf8f6f4`), FP4/FP6/FP8 paths, **no TMA multicast**
  (cluster shape must be 1×1×1)

## Required setup

CUTLASS/CuTe is a git submodule and everything depends on it:

```bash
git submodule update --init --recursive   # populates third_party/cutlass
```

## Build & run

There is no compiled library — kernels are header-only, so the normal workflow
is compiling a `.cu` that includes the operator header directly with `nvcc`.
The top-level `CMakeLists.txt` only exposes the header-only `blackhole::blackhole`
INTERFACE target (include path = `include/` + the CUTLASS submodule headers).

Compile + run the NVFP4 example (from repo root):

```bash
nvcc -std=c++17 -O3 -arch=sm_120a --expt-relaxed-constexpr \
     -I include \
     -I third_party/cutlass/include \
     -I third_party/cutlass/tools/util/include \
     example/nvfp4_bf16_gemm.cu -o nvfp4_bf16_gemm

./nvfp4_bf16_gemm                 # default 4096^3; runs correctness check + TFLOP/s
./nvfp4_bf16_gemm 8192 8192 4096  # M%128==0, N%128==0, K%256==0
```

The trailing **`a`** in `-arch=sm_120a` (architecture-specific) is required — the
Blackwell block-scaled MMA PTX is arch-conditional and is **not** emitted for
plain `sm_120`. **CUDA Toolkit ≥ 12.8** is required for sm_120.

Profiling Tensor Core utilization with `ncu` is documented in
[`example/README.md`](example/README.md) (key metric:
`sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active`).

## Conventions

- **Include root is `include/`**; headers are included as
  `#include <blackhole/sm120/...>` / `#include <blackhole/common/...>`.
- **`include/blackhole/common/arch.hpp`** is the arch dispatch surface: `Arch`
  enum, `arch_traits<Arch>` (e.g. `has_tcgen05`), and the
  `BLACKHOLE_ARCH_SM100`/`BLACKHOLE_ARCH_SM120` macros derived from `__CUDA_ARCH__`.
  Gate arch-specific device code through these.
- New operators go under `include/blackhole/{sm100,sm120}/`, one self-contained
  header per operator.

## Kernel design reference: `sm120/nvfp4_gemm.hpp`

The NVFP4×NVFP4→BF16 GEMM is the worked example of the intended style; mirror its
structure for new sm_120 kernels. Key pieces:

- A single `Nvfp4GemmTraits` struct centralizes **all** tunables and derived
  types (element types, `TiledMMA`, CTA tile `128×128×128`, pipeline `Stages`,
  smem layouts, TMA byte counts, `SharedStorage`). Tune here.
- **Software-pipelined mainloop**: multi-stage smem buffers, a pair of mbarriers
  per stage (`full` = TMA-done, `empty` = all-threads-read-done). One elected
  lane (`elect_one_sync`) issues all TMAs and also participates in the MMA — there
  is no separate producer warp (sm_120 gains little from one). The K dimension is
  double-buffered (intra-stage + cross-stage register prefetch) so `ldmatrix`
  overlaps MMA and the tensor core never stalls.
- **Epilogue** reuses the mainloop smem (a `union` in `SharedStorage`): f32
  accumulators → bf16 → `stmatrix` to smem → one TMA store to gmem.
- The actual MMA is dispatched via `cute::gemm(tiled_mma, make_zip_tensor(A, SFA),
  make_zip_tensor(B, SFB), accum)` — operand data and scale factors are zipped
  together and `mma_unpack` (in cute's sm120 traits) fills the block-scale PTX.

### Block-scaled scale-factor layout (important)

Scale-factor (SF) tensors do **not** use a plain row/col-major layout. They follow
CUTLASS's `cutlass::detail::Sm1xxBlockScaledConfig` interleaved format (128×4
blocked; build via `tile_atom_to_shape_SFA/SFB`, exposed here as
`make_layout_SFA/SFB`). Any new block-scaled operator must produce/consume SF data
in this exact layout to stay interoperable with CUTLASS/cuBLAS reference data — do
not invent a custom SF layout.

## Verifying without a Blackwell GPU

Most CuTe bugs live in compile-time layout algebra (`make_tma_copy`, MMA
partitioning, smem-budget `static_assert`s), which is **fully host-evaluable**.
A layout/partition smoke test compiled with plain `g++` — replicating the kernel's
tensor-partition sequence, instantiating the TMA/MMA types but never executing
them — catches the majority of template/shape errors. (Running such code on host
prints `cast_smem_ptr_to_uint not supported` warnings; those are harmless
host-execution artifacts — only the `static_assert`s matter.) Device compilation
and numerical correctness still require an actual sm_120 GPU + CUDA ≥ 12.8.
