cmake_minimum_required(VERSION 3.18)
# Build fragment for cpu.
if(NOT TARGET pygo_cpu)
  add_library(pygo_cpu INTERFACE)
  target_include_directories(pygo_cpu INTERFACE ${CMAKE_CURRENT_LIST_DIR} ${CMAKE_CURRENT_LIST_DIR}/../include)
  target_compile_features(pygo_cpu INTERFACE cxx_std_17)
endif()
