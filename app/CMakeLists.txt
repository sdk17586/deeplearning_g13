cmake_minimum_required(VERSION 3.16)

project(deepstream_action_predict LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(DEEPSTREAM_DIR "/opt/nvidia/deepstream/deepstream" CACHE PATH "DeepStream install directory")
set(DEEPSTREAM_LIB_DIR "${DEEPSTREAM_DIR}/lib" CACHE PATH "DeepStream library directory")

find_package(PkgConfig REQUIRED)
pkg_check_modules(GSTREAMER REQUIRED gstreamer-1.0)
pkg_check_modules(GLIB REQUIRED glib-2.0)

add_executable(deepstream_action_predict
  src/main.cpp
  src/config/app_config.cpp
  src/video_input/video_input_source_bin.cpp
  src/prediction/prediction_result_handler.cpp
  src/action_pipeline/action_recognition_pipeline.cpp
  src/validation/validation_runner.cpp
)

target_include_directories(deepstream_action_predict PRIVATE
  ${GSTREAMER_INCLUDE_DIRS}
  ${GLIB_INCLUDE_DIRS}
  ${DEEPSTREAM_DIR}/sources/includes
  ${DEEPSTREAM_DIR}/sources/gst-plugins/gst-nvdspreprocess/include
  ${CMAKE_CURRENT_SOURCE_DIR}/src/config
  ${CMAKE_CURRENT_SOURCE_DIR}/src/video_input
  ${CMAKE_CURRENT_SOURCE_DIR}/src/prediction
  ${CMAKE_CURRENT_SOURCE_DIR}/src/action_pipeline
  ${CMAKE_CURRENT_SOURCE_DIR}/src/validation
)

target_compile_options(deepstream_action_predict PRIVATE
  -Wall
  -Wextra
  ${GSTREAMER_CFLAGS_OTHER}
  ${GLIB_CFLAGS_OTHER}
)

target_link_directories(deepstream_action_predict PRIVATE
  ${DEEPSTREAM_LIB_DIR}
)

target_link_libraries(deepstream_action_predict PRIVATE
  ${GSTREAMER_LIBRARIES}
  ${GLIB_LIBRARIES}
  nvdsgst_meta
  nvds_meta
)

set_target_properties(deepstream_action_predict PROPERTIES
  BUILD_RPATH "${DEEPSTREAM_LIB_DIR}"
)
