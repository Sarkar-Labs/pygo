cmake_minimum_required(VERSION 3.18)
# Build fragment for src.
if(NOT TARGET pygo_src)
  add_library(pygo_src INTERFACE)
  target_include_directories(pygo_src INTERFACE ${CMAKE_CURRENT_LIST_DIR} ${CMAKE_CURRENT_LIST_DIR}/../include)
  target_compile_features(pygo_src INTERFACE cxx_std_17)
endif()
