cmake_minimum_required(VERSION 3.18)
# Build fragment for nn.
if(NOT TARGET pygo_nn)
  add_library(pygo_nn INTERFACE)
  target_include_directories(pygo_nn INTERFACE ${CMAKE_CURRENT_LIST_DIR} ${CMAKE_CURRENT_LIST_DIR}/../include)
  target_compile_features(pygo_nn INTERFACE cxx_std_17)
endif()
