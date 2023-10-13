cmake_minimum_required(VERSION 3.5)

project(roadlinedetector LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project( roadlinedetector )
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable(roadlinedetector
    main.cpp
    ./resources/video.mp4
    helper.cpp)

set(OpenCV_LIBS "/usr/local/Cellar/opencv/4.8.1_1/lib")
target_link_libraries( roadlinedetector ${OpenCV_LIBS} )
