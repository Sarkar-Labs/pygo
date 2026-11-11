cmake_minimum_required(VERSION 3.18)
# Build fragment for layers.
if(NOT TARGET pygo_layers)
  add_library(pygo_layers INTERFACE)
  target_include_directories(pygo_layers INTERFACE ${CMAKE_CURRENT_LIST_DIR} ${CMAKE_CURRENT_LIST_DIR}/../include)
  target_compile_features(pygo_layers INTERFACE cxx_std_17)
endif()
