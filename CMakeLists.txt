cmake_minimum_required(VERSION 3.14)
project(Projet_OpenCV_CMake)


set(CMAKE_CXX_STANDARD 14)
find_package(OpenCV REQUIRED)

include_directories(include ${OpenCV_INCLUDE_DIRS})


add_executable(
        Projet_OpenCV_CMake
        src/main_test_opencv.cpp
        include/features.hpp src/features.cpp)

target_link_libraries(Projet_OpenCV_CMake ${OpenCV_LIBS})

