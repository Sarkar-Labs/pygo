cmake_minimum_required(VERSION 3.18)
# Build fragment for tensor.
if(NOT TARGET pygo_tensor)
  add_library(pygo_tensor INTERFACE)
  target_include_directories(pygo_tensor INTERFACE ${CMAKE_CURRENT_LIST_DIR} ${CMAKE_CURRENT_LIST_DIR}/../include)
  target_compile_features(pygo_tensor INTERFACE cxx_std_17)
endif()
