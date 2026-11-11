cmake_minimum_required(VERSION 3.18)
project(pygo_native LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_library(pygo_generated_runtime INTERFACE)
target_include_directories(pygo_generated_runtime INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
# Optional object compilation can be enabled by consumers; the individual files are kept dependency-light.
