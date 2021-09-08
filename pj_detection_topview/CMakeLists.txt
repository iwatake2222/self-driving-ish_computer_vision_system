cmake_minimum_required(VERSION 3.0)

set(ExecName "pj_detection_topview")

# Create executable file
add_executable(${ExecName})
set_target_properties(
    ${ExecName}
    PROPERTIES OUTPUT_NAME "main"
    SOURCES main.cpp
)

# OpenCV
find_package(OpenCV REQUIRED)
target_include_directories(${ExecName} PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(${ExecName} ${OpenCV_LIBS})

# Link ImageProcessor module
add_subdirectory(./image_processor image_processor)
target_include_directories(${ExecName} PUBLIC ./image_processor)
target_link_libraries(${ExecName} detection_topview_image_processor)
