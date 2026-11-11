cmake_minimum_required(VERSION 3.18)
# Build fragment for avx.
if(NOT TARGET pygo_avx)
  add_library(pygo_avx INTERFACE)
  target_include_directories(pygo_avx INTERFACE ${CMAKE_CURRENT_LIST_DIR} ${CMAKE_CURRENT_LIST_DIR}/../include)
  target_compile_features(pygo_avx INTERFACE cxx_std_17)
endif()
