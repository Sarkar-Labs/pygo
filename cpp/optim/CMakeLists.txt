cmake_minimum_required(VERSION 3.18)
# Build fragment for optim.
if(NOT TARGET pygo_optim)
  add_library(pygo_optim INTERFACE)
  target_include_directories(pygo_optim INTERFACE ${CMAKE_CURRENT_LIST_DIR} ${CMAKE_CURRENT_LIST_DIR}/../include)
  target_compile_features(pygo_optim INTERFACE cxx_std_17)
endif()
