cmake_minimum_required(VERSION 3.18)
# Build fragment for neon.
if(NOT TARGET pygo_neon)
  add_library(pygo_neon INTERFACE)
  target_include_directories(pygo_neon INTERFACE ${CMAKE_CURRENT_LIST_DIR} ${CMAKE_CURRENT_LIST_DIR}/../include)
  target_compile_features(pygo_neon INTERFACE cxx_std_17)
endif()
