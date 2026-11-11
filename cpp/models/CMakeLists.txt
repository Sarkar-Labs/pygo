cmake_minimum_required(VERSION 3.18)
# Build fragment for models.
if(NOT TARGET pygo_models)
  add_library(pygo_models INTERFACE)
  target_include_directories(pygo_models INTERFACE ${CMAKE_CURRENT_LIST_DIR} ${CMAKE_CURRENT_LIST_DIR}/../include)
  target_compile_features(pygo_models INTERFACE cxx_std_17)
endif()
