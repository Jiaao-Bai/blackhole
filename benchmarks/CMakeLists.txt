# Performance benchmarks. Add per-arch subdirectories as kernels land.
if(BLACKHOLE_SM100)
  add_subdirectory(sm100)
endif()
if(BLACKHOLE_SM120)
  add_subdirectory(sm120)
endif()
