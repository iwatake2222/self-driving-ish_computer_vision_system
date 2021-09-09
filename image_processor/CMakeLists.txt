cmake_minimum_required(VERSION 3.0)

set(LibraryName "ImageProcessor")

# Create library
add_library (${LibraryName} image_processor.cpp image_processor.h image_processor_if.cpp image_processor_if.h
    detection_engine.cpp detection_engine.h
    lane_engine.cpp lane_engine.h
    semantic_segmentation_engine.cpp semantic_segmentation_engine.h
    depth_engine.cpp depth_engine.h
    lane_information.cpp lane_information.h
)

# Sub modules
## CommonHelper
target_include_directories(${LibraryName} PUBLIC ${PROJECT_SOURCE_DIR}/common_helper)
target_link_libraries(${LibraryName} CommonHelper)

## InferenceHelper
target_include_directories(${LibraryName} PUBLIC ${INFERENCE_HELPER_DIR}/inference_helper)
target_link_libraries(${LibraryName} InferenceHelper)

## OpenCV
find_package(OpenCV REQUIRED)
target_include_directories(${LibraryName} PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(${LibraryName} ${OpenCV_LIBS})
