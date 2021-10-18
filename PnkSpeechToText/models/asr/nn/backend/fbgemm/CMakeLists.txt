cmake_minimum_required(VERSION 3.5.1)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/cmake)

include(BuildFbGemm)

add_library(interface_models_asr_nn_backend
  ${CMAKE_CURRENT_LIST_DIR}/Conv1dFbGemm.cpp
  ${CMAKE_CURRENT_LIST_DIR}/LinearFbGemm.cpp
  ${CMAKE_CURRENT_LIST_DIR}/PackedGemmMatrixFP16.cpp
)
set_property(TARGET interface_models_asr_nn_backend PROPERTY POSITION_INDEPENDENT_CODE ON)
set_target_properties(
  interface_models_asr_nn_backend
  PROPERTIES
    LINKER_LANGUAGE CXX
    CXX_STANDARD 11
)

add_dependencies(interface_models_asr_nn_backend fbgemm)
add_dependencies(interface_models_asr_nn_backend cereal)

target_link_libraries(interface_models_asr_nn_backend
  PUBLIC
    interface_models_asr
    ${fbgemm_LIBRARIES}
)

target_include_directories(interface_models_asr_nn_backend
  PUBLIC
    ${fbgemm_INCLUDE_DIRS}
    ${cereal_INCLUDE_DIRS}
    ${wav2letter-inference_SOURCE_DIR}/..
)

set(BACKEND_FOUND true)
