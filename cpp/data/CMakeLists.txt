cmake_minimum_required(VERSION 3.18)
# Build fragment for data.
if(NOT TARGET pygo_data)
  add_library(pygo_data INTERFACE)
  target_include_directories(pygo_data INTERFACE ${CMAKE_CURRENT_LIST_DIR} ${CMAKE_CURRENT_LIST_DIR}/../include)
  target_compile_features(pygo_data INTERFACE cxx_std_17)
endif()
