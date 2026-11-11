cmake_minimum_required(VERSION 3.18)
# Build fragment for core.
if(NOT TARGET pygo_core)
  add_library(pygo_core INTERFACE)
  target_include_directories(pygo_core INTERFACE ${CMAKE_CURRENT_LIST_DIR} ${CMAKE_CURRENT_LIST_DIR}/../include)
  target_compile_features(pygo_core INTERFACE cxx_std_17)
endif()
