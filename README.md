# blackhole

A pure [CuTe](https://github.com/NVIDIA/cutlass/tree/main/include/cute)-based
collection of high-performance kernels for NVIDIA **Blackwell** GPUs
(`sm_100` and `sm_120`).

`blackhole` is a header-first operator library. Kernels are written directly
against CuTe primitives (layouts, tensors, copy/MMA atoms) rather than the
higher-level CUTLASS collective/device APIs, giving fine-grained control over
the Blackwell tensor-core (`tcgen05`) and TMA paths.

## Targets

| SM arch  | GPUs                          | Notes                                  |
| -------- | ----------------------------- | -------------------------------------- |
| `sm_100` | Datacenter Blackwell (B100/B200) | `tcgen05` MMA, 5th-gen tensor cores |
| `sm_120` | Consumer Blackwell (RTX 50xx) | Blackwell SM, FP4/FP8 paths            |

## Repository layout

```
blackhole/
├── include/blackhole/      # Public headers (header-only operators)
│   ├── common/             # Shared utilities (arch traits, tiling, types)
│   ├── sm100/              # sm_100 operator headers
│   └── sm120/              # sm_120 operator headers
├── CMakeLists.txt          # header-only blackhole::blackhole target
└── third_party/
    └── cutlass/            # CUTLASS/CuTe (git submodule)
```

Tests, benchmarks and examples are intentionally omitted; add them when needed.

## Getting started

Clone with submodules:

```bash
git clone --recursive https://github.com/jiaao-bai/blackhole.git
# or, if already cloned:
git submodule update --init --recursive
```

Build (CUDA 12.8+ recommended for full Blackwell support):

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release \
      -DBLACKHOLE_SM100=ON -DBLACKHOLE_SM120=ON
cmake --build build -j
```

## License

See [LICENSE](LICENSE). CUTLASS is distributed under its own license; see
`third_party/cutlass/LICENSE.txt`.
