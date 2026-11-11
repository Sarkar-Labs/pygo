cmake_minimum_required(VERSION 3.18)
# Build fragment for runtime.
if(NOT TARGET pygo_runtime)
  add_library(pygo_runtime INTERFACE)
  target_include_directories(pygo_runtime INTERFACE ${CMAKE_CURRENT_LIST_DIR} ${CMAKE_CURRENT_LIST_DIR}/../include)
  target_compile_features(pygo_runtime INTERFACE cxx_std_17)
endif()
