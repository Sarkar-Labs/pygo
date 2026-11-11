cmake_minimum_required(VERSION 3.18)
# Build fragment for vulkan.
if(NOT TARGET pygo_vulkan)
  add_library(pygo_vulkan INTERFACE)
  target_include_directories(pygo_vulkan INTERFACE ${CMAKE_CURRENT_LIST_DIR} ${CMAKE_CURRENT_LIST_DIR}/../include)
  target_compile_features(pygo_vulkan INTERFACE cxx_std_17)
endif()
