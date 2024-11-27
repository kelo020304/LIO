cmake_minimum_required(VERSION 2.8.3)
project(eskf)

if( NOT CMAKE_BUILD_TYPE )
    set( CMAKE_BUILD_TYPE Release FORCE )
endif()


#set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++11 -march=native")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall")
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(catkin REQUIRED COMPONENTS roscpp std_msgs geometry_msgs nav_msgs tf cv_bridge image_transport)

find_package(OpenCV REQUIRED)
find_package(PCL REQUIRED)

include_directories(${catkin_INCLUDE_DIRS}
        ${PCL_INCLUDE_DIRS})

catkin_package()

add_executable(eskf
        src/eskf_estimator_node.cpp
        src/eskf_estimator.cpp
)

target_link_libraries(eskf ${catkin_LIBRARIES} ${PCL_LIBRARIES} )

add_executable(simulator
        simulation/data_generator_node.cpp
        simulation/data_generator.cpp
)

target_link_libraries(simulator ${catkin_LIBRARIES} ${OpenCV_LIBS} )