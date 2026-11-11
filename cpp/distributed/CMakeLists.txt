cmake_minimum_required(VERSION 3.18)
# Build fragment for distributed.
if(NOT TARGET pygo_distributed)
  add_library(pygo_distributed INTERFACE)
  target_include_directories(pygo_distributed INTERFACE ${CMAKE_CURRENT_LIST_DIR} ${CMAKE_CURRENT_LIST_DIR}/../include)
  target_compile_features(pygo_distributed INTERFACE cxx_std_17)
endif()
