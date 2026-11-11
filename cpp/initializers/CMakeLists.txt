cmake_minimum_required(VERSION 3.18)
# Build fragment for initializers.
if(NOT TARGET pygo_initializers)
  add_library(pygo_initializers INTERFACE)
  target_include_directories(pygo_initializers INTERFACE ${CMAKE_CURRENT_LIST_DIR} ${CMAKE_CURRENT_LIST_DIR}/../include)
  target_compile_features(pygo_initializers INTERFACE cxx_std_17)
endif()
