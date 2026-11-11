cmake_minimum_required(VERSION 3.18)
# Build fragment for simd.
if(NOT TARGET pygo_simd)
  add_library(pygo_simd INTERFACE)
  target_include_directories(pygo_simd INTERFACE ${CMAKE_CURRENT_LIST_DIR} ${CMAKE_CURRENT_LIST_DIR}/../include)
  target_compile_features(pygo_simd INTERFACE cxx_std_17)
endif()
