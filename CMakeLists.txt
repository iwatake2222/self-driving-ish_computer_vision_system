cmake_minimum_required(VERSION 3.0)

# Create project
set(ProjectName "detection_topview")
project(${ProjectName})

# Select build system and set compile options
include(${CMAKE_CURRENT_LIST_DIR}/common_helper/cmakes/build_setting.cmake)

# Include common module
include_directories(cvui)
add_subdirectory(common_helper)

# InferenceHelper
set(INFERENCE_HELPER_DIR ${CMAKE_CURRENT_LIST_DIR}/InferenceHelper/)
set(INFERENCE_HELPER_ENABLE_TFLITE ON CACHE BOOL "TFLITE")
set(INFERENCE_HELPER_ENABLE_TFLITE_DELEGATE_XNNPACK ON CACHE BOOL "TFLITE_XNNPACK")
add_subdirectory(${INFERENCE_HELPER_DIR}/inference_helper inference_helper)
include_directories(${INFERENCE_HELPER_DIR}/inference_helper)

# Project directories
add_subdirectory(pj_detection_topview)

# Copy resouce
file(COPY ${CMAKE_CURRENT_LIST_DIR}/resource DESTINATION ${CMAKE_BINARY_DIR}/)
add_definitions(-DRESOURCE_DIR="${CMAKE_BINARY_DIR}/resource/")
