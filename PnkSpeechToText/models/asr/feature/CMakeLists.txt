add_library(interface_models_asr_feature
  ${CMAKE_CURRENT_LIST_DIR}/LogMelFeature.cpp
)
set_property(TARGET interface_models_asr_feature PROPERTY POSITION_INDEPENDENT_CODE ON)
add_dependencies(interface_models_asr_feature cereal)

target_link_libraries(
  interface_models_asr_feature
  PUBLIC
    ${cereal_LIBRARIES}
    interface_utils
)

target_include_directories(
  interface_models_asr_feature
  PUBLIC
    ${cereal_INCLUDE_DIRS}
    # going up one directory to use #include "inference/*/*.h"
    /root/wav2letter/src
    ${wav2letter-inference_SOURCE_DIR}
    ${wav2letter++_SOURCE_DIR}/src
)
