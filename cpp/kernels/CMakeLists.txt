cmake_minimum_required(VERSION 3.18)
# Build fragment for kernels.
if(NOT TARGET pygo_kernels)
  add_library(pygo_kernels INTERFACE)
  target_include_directories(pygo_kernels INTERFACE ${CMAKE_CURRENT_LIST_DIR} ${CMAKE_CURRENT_LIST_DIR}/../include)
  target_compile_features(pygo_kernels INTERFACE cxx_std_17)
endif()
