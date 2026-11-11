cmake_minimum_required(VERSION 3.18)
# Build fragment for autograd.
if(NOT TARGET pygo_autograd)
  add_library(pygo_autograd INTERFACE)
  target_include_directories(pygo_autograd INTERFACE ${CMAKE_CURRENT_LIST_DIR} ${CMAKE_CURRENT_LIST_DIR}/../include)
  target_compile_features(pygo_autograd INTERFACE cxx_std_17)
endif()
