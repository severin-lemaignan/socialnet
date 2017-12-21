cmake_minimum_required(VERSION 3.2)

project(socialnet VERSION 0.1.0 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 11)

add_definitions(-DSOCIALNET_VERSION=${PROJECT_VERSION})


######################################################################
######################################################################
########## Dependencies

find_package(BLAS REQUIRED)
message(STATUS "blas: ${BLAS_LIBRARIES}")

find_package(Boost REQUIRED COMPONENTS program_options)

find_package(yaml-cpp REQUIRED)

# attention: we need to find OpenCV *before* Caffe to make sure Caffe
# uses the same version of OpenCV as the rest of the app
find_package(OpenCV 3.0.0 COMPONENTS core imgproc calib3d highgui REQUIRED)
message(STATUS "Found OpenCV ${OpenCV_VERSION}")

find_package(Caffe REQUIRED)
message(STATUS "Found Caffe at ${Caffe_DIR}")
if(Caffe_HAVE_CUDA)
    message(STATUS "Caffe is CUDA-accelerated. Fine.")
    if(Caffe_HAVE_CUDNN)
        message(STATUS "Caffe is CUDNN-accelerated. Very Fine.")
    else()
        message(WARNING "Caffe has CUDA but no CUDNN support enabled!")
    endif()
else()
    message(WARNING "Caffe has not CUDA support enabled!")
endif()

find_package(catkin REQUIRED COMPONENTS 
    roscpp
    rosbag
    tf2
    std_msgs
    visualization_msgs
    message_filters
    sensor_msgs
    cv_bridge
    image_geometry
    audio_common_msgs
    )


catkin_package(
    CATKIN_DEPENDS rosbag
    LIBRARIES 
)

######################################################################
######################################################################
#########  Targets

add_executable(train src/train.cpp)

add_executable(socialnet::train ALIAS train)

target_include_directories(train
    PRIVATE ${BLAS_INCLUDE_DIRS}
    PRIVATE ${YAML_CPP_INCLUDE_DIRS}
    PRIVATE ${Caffe_INCLUDE_DIRS}
    PRIVATE ${OpenCV_INCLUDE_DIRS}
    PRIVATE ${catkin_INCLUDE_DIRS}
    )

target_link_libraries(train
    PUBLIC ${BLAS_LIBRARIES}
    PUBLIC ${Caffe_LIBRARIES}
    PUBLIC ${OpenCV_LIBRARIES}
    PUBLIC ${catkin_LIBRARIES}
    )


