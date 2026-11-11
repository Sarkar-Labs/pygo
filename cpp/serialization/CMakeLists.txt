cmake_minimum_required(VERSION 3.18)
# Build fragment for serialization.
if(NOT TARGET pygo_serialization)
  add_library(pygo_serialization INTERFACE)
  target_include_directories(pygo_serialization INTERFACE ${CMAKE_CURRENT_LIST_DIR} ${CMAKE_CURRENT_LIST_DIR}/../include)
  target_compile_features(pygo_serialization INTERFACE cxx_std_17)
endif()
