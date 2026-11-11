cmake_minimum_required(VERSION 3.18)
# Build fragment for device.
if(NOT TARGET pygo_device)
  add_library(pygo_device INTERFACE)
  target_include_directories(pygo_device INTERFACE ${CMAKE_CURRENT_LIST_DIR} ${CMAKE_CURRENT_LIST_DIR}/../include)
  target_compile_features(pygo_device INTERFACE cxx_std_17)
endif()
