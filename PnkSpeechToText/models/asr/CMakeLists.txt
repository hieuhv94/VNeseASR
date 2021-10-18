cmake_minimum_required(VERSION 3.5.1)

# Creates interface library: interface_models_asr
include(${CMAKE_CURRENT_LIST_DIR}/nn/CMakeLists.txt)
include(${CMAKE_CURRENT_LIST_DIR}/feature/CMakeLists.txt)
include(${CMAKE_CURRENT_LIST_DIR}/decoder/CMakeLists.txt)

add_library(interface_models_asr INTERFACE)

target_sources(interface_models_asr
  INTERFACE
  ${CMAKE_CURRENT_LIST_DIR}/InferenceModule.cpp
  ${CMAKE_CURRENT_LIST_DIR}/ModuleParameter.cpp
  ${CMAKE_CURRENT_LIST_DIR}/ModuleProcessingState.cpp
)

add_dependencies(interface_models_asr interface_models_asr_nn)
add_dependencies(interface_models_asr cereal)

target_link_libraries(
    interface_models_asr
    INTERFACE
    ${cereal_LIBRARIES}
    interface_models_asr_nn
    interface_models_asr_feature
    interface_models_asr_decoder
)

target_include_directories(
  interface_models_asr
  INTERFACE
  # going up one directory to use #include "inference/*/*.h"
  ${wav2letter-inference_SOURCE_DIR}
  ${cereal_INCLUDE_DIRS}
  ${nn_backend_INCLUDE_DIRS}
)
