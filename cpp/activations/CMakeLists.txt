cmake_minimum_required(VERSION 3.18)
# Build fragment for activations.
if(NOT TARGET pygo_activations)
  add_library(pygo_activations INTERFACE)
  target_include_directories(pygo_activations INTERFACE ${CMAKE_CURRENT_LIST_DIR} ${CMAKE_CURRENT_LIST_DIR}/../include)
  target_compile_features(pygo_activations INTERFACE cxx_std_17)
endif()
