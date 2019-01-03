cmake_minimum_required (VERSION 3.2)
set (TARGET_NAME test_tf_cls)
project(${TARGET_NAME})

add_definitions(-std=c++11)

find_package(OpenCV)
find_package (Eigen3)

include_directories(include/third_party
	include/
	include/third_party/eigen3/
	${OpenCV_INCLUDE_DIRS}
	)

link_directories(
	./lib
	)

add_executable (${TARGET_NAME}
	main.cc
    )

target_link_libraries (${TARGET_NAME}
	${OpenCV_LIBS}
	libpthread.so
	libtensorflow_cc.so
	libtensorflow_framework.so
	)
