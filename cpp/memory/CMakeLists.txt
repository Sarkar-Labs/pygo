cmake_minimum_required(VERSION 3.18)
# Build fragment for memory.
if(NOT TARGET pygo_memory)
  add_library(pygo_memory INTERFACE)
  target_include_directories(pygo_memory INTERFACE ${CMAKE_CURRENT_LIST_DIR} ${CMAKE_CURRENT_LIST_DIR}/../include)
  target_compile_features(pygo_memory INTERFACE cxx_std_17)
endif()
