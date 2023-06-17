#cmake_minimum_required(VERSION 3.24)
#project(ncnn_example)
#
#set(CMAKE_CXX_STANDARD 17)
## Where to find CMake modules and OpenCV
#
#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -lstdc++")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -lstdc++")
#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp ")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp ")
##add_definitions(-D_GLIBCXX_USE_CXX14_ABI=0)
#
##配置opencv
##set(OpenCV_DIR "D:\\sofeware\\mingw_b\\install")
#set(OpenCV_DIR "D:\\sofeware\\OpenCV\\opencv\\mingw_build")
#find_package(OpenCV REQUIRED)
#add_definitions(-D_GLIBCXX_USE_CXX14_ABI=0)
#include_directories(${OpenCV_INCLUDE_DIRS})
#
##打印一下目录情况
##get_property(dirs DIRECTORY ${CMAKE_SOURCE_DIR} PROPERTY INCLUDE_DIRECTORIES)
##message(">>> include_dirs=${dirs}")
#
##导入生成的ncnn install目录
#set(NCNN_DIR D:/code_repositorie/ncnn-master/ncnn-master/build/install)
#include_directories(${NCNN_DIR}/include)
#include_directories(${NCNN_DIR}/lib)
#
#
##导入ncnn
#add_library(libncnn STATIC IMPORTED)
#set_target_properties(libncnn PROPERTIES IMPORTED_LOCATION ${NCNN_DIR}/lib/libncnn.a)
#
#
##添加执行程序
#add_executable(ncnn_example squeezenet.cpp)
#
##连接库
#set(OpenCV_LIBS opencv_core opencv_imgproc opencv_highgui opencv_imgcodecs)
#target_link_libraries(ncnn_example ${OpenCV_LIBS} libncnn)

cmake_minimum_required(VERSION 3.24)
project(VisionEngine)

Option(MIRROR_OPENMP "openmp support" ON)
Option(MIRROR_VULKAN "vulkan compute used" OFF)
Option(MIRROR_INSTALL_SDK "install mirror library and headers" OFF)
Option(MIRROR_BUILD_FACE "build face test" ON)
option(MIRROR_BUILD_OBJECT "build object test" OFF)

set(CMAKE_PREFIX_PATH
        ${CMAKE_PREFIX_PATH}
        D:\\Software\\tools\\protobuf\\protobuf-cpp-3.21.12\\protobuf-3.21.12\\install)

find_package(Protobuf REQUIRED)

#add_subdirectory(ncnn-20230517-full-source)

add_subdirectory(src)
