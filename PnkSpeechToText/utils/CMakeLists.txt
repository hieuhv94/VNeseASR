cmake_minimum_required(VERSION 3.5.1)
message(STATUS "Looking for -------------------- ${wav2letter-inference_SOURCE_DIR}")
find_library(GLOG_LIBRARY glog)
add_library(interface_utils
  ${CMAKE_CURRENT_LIST_DIR}/DataType.cpp
  ${CMAKE_CURRENT_LIST_DIR}/DefaultMemoryManager.cpp
  ${CMAKE_CURRENT_LIST_DIR}/Functions.cpp
  ${CMAKE_CURRENT_LIST_DIR}/IOBuffer.cpp
)
set_property(TARGET interface_utils PROPERTY POSITION_INDEPENDENT_CODE ON)
add_dependencies(interface_utils cereal)
target_link_libraries(
  interface_utils
  PUBLIC
    ${GLOG_LIBRARY}
)
target_include_directories(
  interface_utils
  PUBLIC
    ${cereal_INCLUDE_DIRS}
    # going up one directory to use #include "inference/*/*.h"
    ${CMAKE_CURRENT_LIST_DIR}/..
)
