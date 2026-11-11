cmake_minimum_required(VERSION 3.18)
# Build fragment for inference.
if(NOT TARGET pygo_inference)
  add_library(pygo_inference INTERFACE)
  target_include_directories(pygo_inference INTERFACE ${CMAKE_CURRENT_LIST_DIR} ${CMAKE_CURRENT_LIST_DIR}/../include)
  target_compile_features(pygo_inference INTERFACE cxx_std_17)
endif()
