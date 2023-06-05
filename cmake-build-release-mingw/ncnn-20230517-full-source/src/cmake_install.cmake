# Install script for directory: D:/coding/git_repositories/ncnn_cv/ncnn-20230517-full-source/src

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "C:/Program Files (x86)/VisionEngine")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set default install directory permissions.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "D:/Software/tools/MinGW/mingw64/bin/objdump.exe")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "D:/coding/git_repositories/ncnn_cv/cmake-build-release-mingw/ncnn-20230517-full-source/src/libncnn.a")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/ncnn" TYPE FILE FILES
    "D:/coding/git_repositories/ncnn_cv/ncnn-20230517-full-source/src/allocator.h"
    "D:/coding/git_repositories/ncnn_cv/ncnn-20230517-full-source/src/benchmark.h"
    "D:/coding/git_repositories/ncnn_cv/ncnn-20230517-full-source/src/blob.h"
    "D:/coding/git_repositories/ncnn_cv/ncnn-20230517-full-source/src/c_api.h"
    "D:/coding/git_repositories/ncnn_cv/ncnn-20230517-full-source/src/command.h"
    "D:/coding/git_repositories/ncnn_cv/ncnn-20230517-full-source/src/cpu.h"
    "D:/coding/git_repositories/ncnn_cv/ncnn-20230517-full-source/src/datareader.h"
    "D:/coding/git_repositories/ncnn_cv/ncnn-20230517-full-source/src/gpu.h"
    "D:/coding/git_repositories/ncnn_cv/ncnn-20230517-full-source/src/layer.h"
    "D:/coding/git_repositories/ncnn_cv/ncnn-20230517-full-source/src/layer_shader_type.h"
    "D:/coding/git_repositories/ncnn_cv/ncnn-20230517-full-source/src/layer_type.h"
    "D:/coding/git_repositories/ncnn_cv/ncnn-20230517-full-source/src/mat.h"
    "D:/coding/git_repositories/ncnn_cv/ncnn-20230517-full-source/src/modelbin.h"
    "D:/coding/git_repositories/ncnn_cv/ncnn-20230517-full-source/src/net.h"
    "D:/coding/git_repositories/ncnn_cv/ncnn-20230517-full-source/src/option.h"
    "D:/coding/git_repositories/ncnn_cv/ncnn-20230517-full-source/src/paramdict.h"
    "D:/coding/git_repositories/ncnn_cv/ncnn-20230517-full-source/src/pipeline.h"
    "D:/coding/git_repositories/ncnn_cv/ncnn-20230517-full-source/src/pipelinecache.h"
    "D:/coding/git_repositories/ncnn_cv/ncnn-20230517-full-source/src/simpleocv.h"
    "D:/coding/git_repositories/ncnn_cv/ncnn-20230517-full-source/src/simpleomp.h"
    "D:/coding/git_repositories/ncnn_cv/ncnn-20230517-full-source/src/simplestl.h"
    "D:/coding/git_repositories/ncnn_cv/ncnn-20230517-full-source/src/vulkan_header_fix.h"
    "D:/coding/git_repositories/ncnn_cv/cmake-build-release-mingw/ncnn-20230517-full-source/src/ncnn_export.h"
    "D:/coding/git_repositories/ncnn_cv/cmake-build-release-mingw/ncnn-20230517-full-source/src/layer_shader_type_enum.h"
    "D:/coding/git_repositories/ncnn_cv/cmake-build-release-mingw/ncnn-20230517-full-source/src/layer_type_enum.h"
    "D:/coding/git_repositories/ncnn_cv/cmake-build-release-mingw/ncnn-20230517-full-source/src/platform.h"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/ncnn/ncnn.cmake")
    file(DIFFERENT _cmake_export_file_changed FILES
         "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/ncnn/ncnn.cmake"
         "D:/coding/git_repositories/ncnn_cv/cmake-build-release-mingw/ncnn-20230517-full-source/src/CMakeFiles/Export/790e04ecad7490f293fc4a38f0c73eb1/ncnn.cmake")
    if(_cmake_export_file_changed)
      file(GLOB _cmake_old_config_files "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/ncnn/ncnn-*.cmake")
      if(_cmake_old_config_files)
        string(REPLACE ";" ", " _cmake_old_config_files_text "${_cmake_old_config_files}")
        message(STATUS "Old export file \"$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/ncnn/ncnn.cmake\" will be replaced.  Removing files [${_cmake_old_config_files_text}].")
        unset(_cmake_old_config_files_text)
        file(REMOVE ${_cmake_old_config_files})
      endif()
      unset(_cmake_old_config_files)
    endif()
    unset(_cmake_export_file_changed)
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/ncnn" TYPE FILE FILES "D:/coding/git_repositories/ncnn_cv/cmake-build-release-mingw/ncnn-20230517-full-source/src/CMakeFiles/Export/790e04ecad7490f293fc4a38f0c73eb1/ncnn.cmake")
  if(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/ncnn" TYPE FILE FILES "D:/coding/git_repositories/ncnn_cv/cmake-build-release-mingw/ncnn-20230517-full-source/src/CMakeFiles/Export/790e04ecad7490f293fc4a38f0c73eb1/ncnn-release.cmake")
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/ncnn" TYPE FILE FILES "D:/coding/git_repositories/ncnn_cv/cmake-build-release-mingw/ncnn-20230517-full-source/src/ncnnConfig.cmake")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/pkgconfig" TYPE FILE FILES "D:/coding/git_repositories/ncnn_cv/cmake-build-release-mingw/ncnn-20230517-full-source/src/ncnn.pc")
endif()

