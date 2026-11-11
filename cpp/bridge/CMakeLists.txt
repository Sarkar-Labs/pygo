cmake_minimum_required(VERSION 3.20)
add_library(pygo_c_api OBJECT c_api.cpp)
target_include_directories(pygo_c_api PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/../include ${CMAKE_CURRENT_SOURCE_DIR}/..)
target_compile_features(pygo_c_api PUBLIC cxx_std_17)
