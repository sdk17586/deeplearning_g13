cmake_minimum_required(VERSION 3.16)

project(trt_int8_calibrator
  VERSION 0.1.0
  DESCRIPTION "TensorRT INT8 engine calibration tool"
  LANGUAGES CXX
)

include(GNUInstallDirs)
include(FindPackageHandleStandardArgs)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(CUDAToolkit REQUIRED)

find_path(TENSORRT_INCLUDE_DIR
  NAMES NvInfer.h
  HINTS ENV TENSORRT_ROOT
  PATH_SUFFIXES include
  PATHS
    /usr/include
    /usr/include/x86_64-linux-gnu
)

find_library(TENSORRT_NVINFER_LIBRARY
  NAMES nvinfer
  HINTS ENV TENSORRT_ROOT
  PATH_SUFFIXES lib lib64 lib/x86_64-linux-gnu
  PATHS
    /usr/lib
    /usr/lib/x86_64-linux-gnu
)

find_library(TENSORRT_NVONNXPARSER_LIBRARY
  NAMES nvonnxparser
  HINTS ENV TENSORRT_ROOT
  PATH_SUFFIXES lib lib64 lib/x86_64-linux-gnu
  PATHS
    /usr/lib
    /usr/lib/x86_64-linux-gnu
)

find_package_handle_standard_args(TensorRT
  REQUIRED_VARS
    TENSORRT_INCLUDE_DIR
    TENSORRT_NVINFER_LIBRARY
    TENSORRT_NVONNXPARSER_LIBRARY
)

add_library(trt_int8_calibrator_core STATIC
  calibrator_options.cpp
  file_utils.cpp
  trt_logger.cpp
  trt_utils.cpp
  binary_tensor_calibrator.cpp
  int8_engine_builder.cpp
)

add_library(trt_int8_calibrator::core ALIAS trt_int8_calibrator_core)

target_include_directories(trt_int8_calibrator_core
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    ${TENSORRT_INCLUDE_DIR}
)

target_link_libraries(trt_int8_calibrator_core
  PUBLIC
    CUDA::cudart
    ${TENSORRT_NVINFER_LIBRARY}
    ${TENSORRT_NVONNXPARSER_LIBRARY}
)

add_executable(trt_int8_calibrator main.cpp)
target_link_libraries(trt_int8_calibrator PRIVATE trt_int8_calibrator::core)

install(TARGETS trt_int8_calibrator trt_int8_calibrator_core
  EXPORT trt_int8_calibrator_targets
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

install(FILES
  binary_tensor_calibrator.hpp
  calibrator_options.hpp
  file_utils.hpp
  int8_engine_builder.hpp
  trt_logger.hpp
  trt_utils.hpp
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/trt_int8_calibrator
)
