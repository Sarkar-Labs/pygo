cmake_minimum_required(VERSION 3.18)
# Build fragment for metal.
if(NOT TARGET pygo_metal)
  add_library(pygo_metal INTERFACE)
  target_include_directories(pygo_metal INTERFACE ${CMAKE_CURRENT_LIST_DIR} ${CMAKE_CURRENT_LIST_DIR}/../include)
  target_compile_features(pygo_metal INTERFACE cxx_std_17)
endif()
