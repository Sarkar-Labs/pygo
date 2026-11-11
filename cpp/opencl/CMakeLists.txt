cmake_minimum_required(VERSION 3.18)
# Build fragment for opencl.
if(NOT TARGET pygo_opencl)
  add_library(pygo_opencl INTERFACE)
  target_include_directories(pygo_opencl INTERFACE ${CMAKE_CURRENT_LIST_DIR} ${CMAKE_CURRENT_LIST_DIR}/../include)
  target_compile_features(pygo_opencl INTERFACE cxx_std_17)
endif()
