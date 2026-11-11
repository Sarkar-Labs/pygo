cmake_minimum_required(VERSION 3.18)
# Build fragment for blas.
if(NOT TARGET pygo_blas)
  add_library(pygo_blas INTERFACE)
  target_include_directories(pygo_blas INTERFACE ${CMAKE_CURRENT_LIST_DIR} ${CMAKE_CURRENT_LIST_DIR}/../include)
  target_compile_features(pygo_blas INTERFACE cxx_std_17)
endif()
