cmake_minimum_required(VERSION 3.2)
set(TARGET_NAME test_cpp)

SET(CMAKE_CXX_STANDARD 11)
SET(CMAKE_CXX_STANDARD_REQUIRED ON)

PROJECT(${TARGET_NAME})
SET(${TARGET_NAME}_VERSION "1.0.0.1")

# opencv minimum version 3.3 and open option WITH_VA_INTEL
FIND_PACKAGE(OpenCV)
FIND_PACKAGE(InferenceEngine)

include_directories (${InferenceEngine_INCLUDE_DIRS}
	${OpenCV_INCLUDE_DIRS}
	)

AUX_SOURCE_DIRECTORY(${PROJECT_SOURCE_DIR} SRC_FILES)

LINK_DIRECTORIES(
	${PROJECT_SOURCE_DIR}
)

ADD_EXECUTABLE (${TARGET_NAME} ${SRC_FILES})

TARGET_LINK_LIBRARIES(${TARGET_NAME}
	${OpenCV_LIBS}
	${InferenceEngine_LIBRARIES}
	IE::ie_cpu_extension
	libdl.so			# linux dll
	pthread
)
