cmake_minimum_required(VERSION 3.0)

set(LibraryName "detection_topview_core")

# Create library
add_library (${LibraryName} image_processor.cpp image_processor.h image_processor_if.cpp image_processor_if.h
    detection_engine.cpp detection_engine.h
    lane_engine.cpp lane_engine.h
    semantic_segmentation_engine.cpp semantic_segmentation_engine.h
    depth_engine.cpp depth_engine.h
)

# OpenCV
find_package(OpenCV REQUIRED)
target_include_directories(${LibraryName} PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(${LibraryName} ${OpenCV_LIBS})

# Link Common Module
target_link_libraries(${LibraryName} CommonHelper)
target_link_libraries(${LibraryName} InferenceHelper)
