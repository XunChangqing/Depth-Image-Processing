cmake_minimum_required(VERSION 2.8)
project(dip)

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/modules/" ${CMAKE_MODULE_PATH})

set(BIN_INSTALL_DIR "bin")
set(LIB_INSTALL_DIR "lib")
set(INCLUDE_INSTALL_DIR "include/${PROJECT_NAME}")

if(UNIX)
  set(BUILD_SHARED_LIBS true)
endif()

# CUDA
#find_package(CUDA REQUIRED)
# Eigen
#find_package(Eigen REQUIRED)
# GLUT
find_package(GLUT REQUIRED)
# HDF5
#find_package(HDF5 REQUIRED)
# OpenCV
set(OpenCV_DIR "C:/Users/xcq/gitwork/opencv2410/build")
find_package(OpenCV REQUIRED)
# OpenGL
find_package(OpenGL REQUIRED)
# OpenNI
find_package(OpenNI2 REQUIRED)
# OpenMP
find_package(OpenMP)
if (OPENMP_FOUND)
  set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

set(INCLUDE_DIRS
		${GLUT_INCLUDE_DIR}
		${OpenCV_INCLUDE_DIR}
		${OPENNI2_INCLUDE_DIR}
		"${CMAKE_SOURCE_DIR}/common/include/"
		"${CMAKE_SOURCE_DIR}/cameras/include/"
		"${CMAKE_SOURCE_DIR}/segmentation/include/"
   )

include_directories(${INCLUDE_DIRS})

set(SRCS
    "${CMAKE_SOURCE_DIR}/applications/face_detection/face_detection.cpp"
    "${CMAKE_SOURCE_DIR}/common/src/distance.cpp"
    "${CMAKE_SOURCE_DIR}/cameras/src/primesense.cpp"
    "${CMAKE_SOURCE_DIR}/segmentation/src/facemasker.cpp"
    )

set(LIBS
    ${GLUT_LIBRARIES}
    ${OpenCV_LIBRARIES}
    ${OPENGL_LIBRARIES}
    ${OPENNI2_LIBRARY}
    )

add_executable(face_detector ${SRCS} ${INCS})
target_link_libraries(face_detector ${LIBS})

