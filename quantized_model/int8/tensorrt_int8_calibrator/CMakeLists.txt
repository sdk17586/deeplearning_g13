cmake_minimum_required(VERSION 3.16)

project(trt_int8_calibrator LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(CUDAToolkit REQUIRED)

add_executable(trt_int8_calibrator
  main.cpp
  calibrator_options.cpp
  file_utils.cpp
  trt_logger.cpp
  trt_utils.cpp
  binary_tensor_calibrator.cpp
  int8_engine_builder.cpp
)

target_include_directories(trt_int8_calibrator PRIVATE
  /usr/include/x86_64-linux-gnu
)

target_link_libraries(trt_int8_calibrator PRIVATE
  nvinfer
  nvonnxparser
  CUDA::cudart
)
