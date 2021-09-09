cmake_minimum_required(VERSION 3.12)

# Create project
set(ProjectName "DetectionTopView")
project(${ProjectName})

# Select build system and set compile options
include(${PROJECT_SOURCE_DIR}/common_helper/cmakes/build_setting.cmake)

# Option
set(ENABLE_TENSORRT off CACHE BOOL "ENABLE_TENSORRT? [on/off]")
if(ENABLE_TENSORRT)
    set(INFERENCE_HELPER_ENABLE_TENSORRT ON CACHE BOOL "INFERENCE_HELPER_ENABLE_TENSORRT" FORCE)
    add_compile_definitions(ENABLE_TENSORRT)
endif()

# Copy resouce
file(COPY ${PROJECT_SOURCE_DIR}/resource DESTINATION ${CMAKE_BINARY_DIR}/)
add_compile_definitions(RESOURCE_DIR="${CMAKE_BINARY_DIR}/resource/")

# Create executable file
add_executable(${ProjectName})
set_target_properties(
    ${ProjectName}
    PROPERTIES OUTPUT_NAME "main"
    SOURCES main.cpp
)

# Sub modules
## CommonHelper
add_subdirectory(${PROJECT_SOURCE_DIR}/common_helper)
target_include_directories(${ProjectName} PUBLIC ${PROJECT_SOURCE_DIR}/common_helper)
target_link_libraries(${ProjectName} CommonHelper)

## InferenceHelper
set(INFERENCE_HELPER_DIR ${PROJECT_SOURCE_DIR}/inference_helper)
set(INFERENCE_HELPER_ENABLE_TFLITE ON CACHE BOOL "TFLITE")
set(INFERENCE_HELPER_ENABLE_TFLITE_DELEGATE_XNNPACK ON CACHE BOOL "TFLITE_XNNPACK")
add_subdirectory(${INFERENCE_HELPER_DIR}/inference_helper inference_helper)

## OpenCV
find_package(OpenCV REQUIRED)
target_include_directories(${ProjectName} PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(${ProjectName} ${OpenCV_LIBS})

## cvui
target_include_directories(${ProjectName} PUBLIC ${PROJECT_SOURCE_DIR}/cvui)

## ImageProcessor
add_subdirectory(${PROJECT_SOURCE_DIR}/image_processor)
target_include_directories(${ProjectName} PUBLIC ${PROJECT_SOURCE_DIR}/image_processor)
target_link_libraries(${ProjectName} ImageProcessor)
