cmake_minimum_required(VERSION 3.18)
# Build fragment for profiler.
if(NOT TARGET pygo_profiler)
  add_library(pygo_profiler INTERFACE)
  target_include_directories(pygo_profiler INTERFACE ${CMAKE_CURRENT_LIST_DIR} ${CMAKE_CURRENT_LIST_DIR}/../include)
  target_compile_features(pygo_profiler INTERFACE cxx_std_17)
endif()
