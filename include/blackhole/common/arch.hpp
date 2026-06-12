#pragma once

// Architecture detection and traits for Blackwell targets.
//
// blackhole supports two Blackwell SM families:
//   * sm_100 : datacenter Blackwell (B100/B200), tcgen05 5th-gen tensor cores
//   * sm_120 : consumer Blackwell (RTX 50xx)
//
// Use these macros/traits to gate arch-specific code paths.

#include <cute/config.hpp>

namespace blackhole {

enum class Arch {
  kSm100,
  kSm120,
};

template <Arch A>
struct arch_traits;

template <>
struct arch_traits<Arch::kSm100> {
  static constexpr int sm           = 100;
  static constexpr bool has_tcgen05 = true;
};

template <>
struct arch_traits<Arch::kSm120> {
  static constexpr int sm           = 120;
  static constexpr bool has_tcgen05 = false;
};

// Compile-time current target convenience macros.
#if defined(__CUDA_ARCH__)
#  if (__CUDA_ARCH__ >= 1000) && (__CUDA_ARCH__ < 1200)
#    define BLACKHOLE_ARCH_SM100 1
#  elif (__CUDA_ARCH__ >= 1200) && (__CUDA_ARCH__ < 1300)
#    define BLACKHOLE_ARCH_SM120 1
#  endif
#endif

}  // namespace blackhole
