cmake_minimum_required(VERSION 3.18)
# Build fragment for transforms.
if(NOT TARGET pygo_transforms)
  add_library(pygo_transforms INTERFACE)
  target_include_directories(pygo_transforms INTERFACE ${CMAKE_CURRENT_LIST_DIR} ${CMAKE_CURRENT_LIST_DIR}/../include)
  target_compile_features(pygo_transforms INTERFACE cxx_std_17)
endif()
