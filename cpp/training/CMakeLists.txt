cmake_minimum_required(VERSION 3.18)
# Build fragment for training.
if(NOT TARGET pygo_training)
  add_library(pygo_training INTERFACE)
  target_include_directories(pygo_training INTERFACE ${CMAKE_CURRENT_LIST_DIR} ${CMAKE_CURRENT_LIST_DIR}/../include)
  target_compile_features(pygo_training INTERFACE cxx_std_17)
endif()
