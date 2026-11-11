cmake_minimum_required(VERSION 3.18)
# Build fragment for metrics.
if(NOT TARGET pygo_metrics)
  add_library(pygo_metrics INTERFACE)
  target_include_directories(pygo_metrics INTERFACE ${CMAKE_CURRENT_LIST_DIR} ${CMAKE_CURRENT_LIST_DIR}/../include)
  target_compile_features(pygo_metrics INTERFACE cxx_std_17)
endif()
