cmake_minimum_required(VERSION 3.0)

set(LibraryName "CommonHelper")

set(COMMON_HELPER_WITH_OPENCV on CACHE BOOL "With OpenCV? [on/off]")


set(SRC
    common_helper.h common_helper.cpp
    bounding_box.h bounding_box.cpp
    simple_matrix.h
    hungarian_algorithm.h
    kalman_filter.h
    tracker.h tracker.cpp
    camera_model.h
)

if(COMMON_HELPER_WITH_OPENCV)
    set(SRC ${SRC} common_helper_cv.h common_helper_cv.cpp)
endif()

add_library(${LibraryName} ${SRC})

if(COMMON_HELPER_WITH_OPENCV)
    find_package(OpenCV REQUIRED)
    target_include_directories(${LibraryName} PUBLIC ${OpenCV_INCLUDE_DIRS})
    target_link_libraries(${LibraryName} ${OpenCV_LIBS})
endif()
