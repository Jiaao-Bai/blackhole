# Repository structure

This document describes how `blackhole` is organized and where new operators
should go.

## Principles

1. **Header-first.** Operators are implemented as templated headers under
   `include/blackhole/`. A consumer can use a kernel by including the header
   and linking the `blackhole::blackhole` INTERFACE target — no separate
   compiled library is required.
2. **Pure CuTe.** Kernels are built from CuTe primitives (layouts, tensors,
   `copy`/`gemm` atoms, TMA, `tcgen05` MMA). We deliberately avoid the
   higher-level CUTLASS collective/device builder APIs to keep full control of
   the Blackwell datapath.
3. **Arch separation.** `sm_100` and `sm_120` have materially different tensor
   cores (`tcgen05` vs. consumer Blackwell), so each gets its own subtree.
   Anything genuinely shared lives in `common/`.

## Where things go

| Directory                     | Contents                                             |
| ----------------------------- | ---------------------------------------------------- |
| `include/blackhole/common/`   | Arch traits, type helpers, tiling/scheduling utils   |
| `include/blackhole/sm100/`    | `sm_100` operator headers (gemm, attention, …)       |
| `include/blackhole/sm120/`    | `sm_120` operator headers                            |
| `src/<arch>/`                 | Optional `.cu` TUs / explicit template instantiations|
| `tests/<arch>/`               | Correctness tests (vs. reference)                    |
| `benchmarks/<arch>/`          | Performance benchmarks                               |
| `examples/`                   | Minimal standalone usage examples                    |
| `cmake/`                      | Reusable CMake helper modules                        |
| `scripts/`                    | Build / profiling / codegen helpers                  |
| `third_party/cutlass/`        | CUTLASS + CuTe (git submodule)                       |

## Include convention

Public headers are included as:

```cpp
#include <blackhole/common/arch.hpp>
#include <blackhole/sm100/gemm.hpp>
```

The `include/` directory is the single public include root.
