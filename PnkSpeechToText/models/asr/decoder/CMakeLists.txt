add_library(interface_models_asr_decoder INTERFACE)

find_package(kenlm REQUIRED)

target_sources(interface_models_asr_decoder
  INTERFACE
    ${CMAKE_CURRENT_LIST_DIR}/Decoder.cpp
)

target_link_libraries(
  interface_models_asr_decoder
  INTERFACE
    ${KENLM_LIBRARIES}
    interface_utils
    flashlight-app-asr
)

target_compile_definitions(
  interface_models_asr_decoder
  INTERFACE
   KENLM_MAX_ORDER=${KENLM_MAX_ORDER}
)

target_include_directories(
  interface_models_asr_decoder
  INTERFACE
    ${KENLM_INCLUDE_DIRS}
)
