project(saliency)
cmake_minimum_required(VERSION 2.8)
find_package( OpenCV REQUIRED )

add_executable(${PROJECT_NAME} src/main.cpp src/GaussPyr.cpp)

target_link_libraries( saliency ${OpenCV_LIBS} )
