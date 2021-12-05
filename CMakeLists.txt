cmake_minimum_required(VERSION 3.0)

# Create project
set(ProjectName "main")
project(${ProjectName})

# Select build system and set compile options
include(${PROJECT_SOURCE_DIR}/common_helper/cmakes/build_setting.cmake)

# Option
set(ENABLE_TENSORRT off CACHE BOOL "ENABLE_TENSORRT? [on/off]")
if(ENABLE_TENSORRT)
    set(INFERENCE_HELPER_ENABLE_TENSORRT ON CACHE BOOL "INFERENCE_HELPER_ENABLE_TENSORRT" FORCE)
    set(INFERENCE_HELPER_ENABLE_TFLITE OFF CACHE BOOL "INFERENCE_HELPER_ENABLE_TFLITE" FORCE)
    set(INFERENCE_HELPER_ENABLE_TFLITE_DELEGATE_XNNPACK OFF CACHE BOOL "INFERENCE_HELPER_ENABLE_TFLITE_DELEGATE_XNNPACK" FORCE)
    add_definitions(-DENABLE_TENSORRT)
else()
    set(INFERENCE_HELPER_ENABLE_TENSORRT OFF CACHE BOOL "INFERENCE_HELPER_ENABLE_TENSORRT" FORCE)
    set(INFERENCE_HELPER_ENABLE_TFLITE ON CACHE BOOL "INFERENCE_HELPER_ENABLE_TFLITE" FORCE)
    set(INFERENCE_HELPER_ENABLE_TFLITE_DELEGATE_XNNPACK ON CACHE BOOL "INFERENCE_HELPER_ENABLE_TFLITE_DELEGATE_XNNPACK" FORCE)
endif()

set(ENABLE_SEGMENTATION on CACHE BOOL "ENABLE_SEGMENTATION? [on/off]")
if(ENABLE_SEGMENTATION)
    add_definitions(-DENABLE_SEGMENTATION)
else()
    add_definitions(-UENABLE_SEGMENTATION)
endif()

set(ENABLE_DEPTH on CACHE BOOL "ENABLE_DEPTH? [on/off]")
if(ENABLE_DEPTH)
    add_definitions(-DENABLE_DEPTH)
else()
    add_definitions(-UENABLE_DEPTH)
endif()

# Copy resouce
file(COPY ${PROJECT_SOURCE_DIR}/resource DESTINATION ${CMAKE_BINARY_DIR}/)
add_definitions(-DRESOURCE_DIR="${CMAKE_BINARY_DIR}/resource/")

# Create executable file
add_executable(${ProjectName} main.cpp)

# Sub modules
## CommonHelper
add_subdirectory(${PROJECT_SOURCE_DIR}/common_helper)
target_include_directories(${ProjectName} PUBLIC ${PROJECT_SOURCE_DIR}/common_helper)
target_link_libraries(${ProjectName} CommonHelper)

## InferenceHelper
set(INFERENCE_HELPER_DIR ${PROJECT_SOURCE_DIR}/inference_helper)
add_subdirectory(${INFERENCE_HELPER_DIR}/inference_helper inference_helper)

## OpenCV
find_package(OpenCV REQUIRED)
target_include_directories(${ProjectName} PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(${ProjectName} ${OpenCV_LIBS})

## OpenMP
find_package(OpenMP REQUIRED)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

## cvui
target_include_directories(${ProjectName} PUBLIC ${PROJECT_SOURCE_DIR}/cvui)

## ImageProcessor
add_subdirectory(${PROJECT_SOURCE_DIR}/image_processor)
target_include_directories(${ProjectName} PUBLIC ${PROJECT_SOURCE_DIR}/image_processor)
target_link_libraries(${ProjectName} ImageProcessor)
