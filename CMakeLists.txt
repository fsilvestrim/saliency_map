cmake_minimum_required(VERSION 2.8)

add_definitions(-std=c++11)

set(PROJECT_NAME saliency)

project(${PROJECT_NAME})

find_package( OpenCV REQUIRED )

find_package(Boost 1.53 COMPONENTS filesystem regex system REQUIRED)
include_directories(${Boost_INCLUDE_DIR})
link_directories(${Boost_LIBRARY_DIR})

file(GLOB PROJ_SRC_FILES src/*.cpp)
file(GLOB PROJ_HEAD_FILES src/*.h)

add_executable(${PROJECT_NAME} ${PROJ_SRC_FILES})

target_link_libraries( saliency ${OpenCV_LIBS} ${Boost_LIBRARIES} ${PROJ_HEAD_FILES})
