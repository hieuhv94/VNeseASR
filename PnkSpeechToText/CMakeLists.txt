cmake_minimum_required(VERSION 3.5.1)

set(KENLM_MAX_ORDER 6 CACHE
    STRING
    "Maximum ngram order for KenLM"
)

# Inference backend names MUST match paths along
# inference/module/nn/backend/{backend_name}
set(AVAILABLE_INFERENCE_BACKENDS fbgemm)
set(W2L_INFERENCE_BACKEND fbgemm CACHE
  STRING
  "Inference backend library"
)
set_property(
  CACHE
  W2L_INFERENCE_BACKEND
  PROPERTY STRINGS
  ${AVAILABLE_INFERENCE_BACKENDS}
)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(pnks2t)

add_library(pnks2t SHARED PnkS2t.cpp Modules.cpp Session.cpp)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/cmake)

#--- Find GFlags ---
find_package(GFLAGS REQUIRED)
if (GFLAGS_FOUND)
  message(STATUS "GFLAGS found")
else()
  message(FATAL_ERROR "GFLAGS not found")
endif()

#--- Download/build Cereal ---
set(CEREAL_INSTALL_PATH ${CMAKE_CURRENT_LIST_DIR}/cereal)
include(Buildcereal)
# Move cereal headers
install(DIRECTORY ${CEREAL_SOURCE_DIR}/include/cereal
  DESTINATION ${CEREAL_INSTALL_PATH}
  COMPONENT cereal
  FILES_MATCHING
  PATTERN "*.hpp"
  PATTERN "*.h"
  PATTERN ".git" EXCLUDE
)
install(DIRECTORY ${CEREAL_SOURCE_DIR}/include/cereal
  DESTINATION include
  COMPONENT cereal
  FILES_MATCHING
  PATTERN "*.hpp"
  PATTERN "*.h"
)
install(
  FILES
    ${CEREAL_SOURCE_DIR}/LICENSE
    ${CEREAL_SOURCE_DIR}/README.md
  DESTINATION
    ${CEREAL_INSTALL_PATH}
)

# ------------------ add subdirectories --------------------

# Creates interface library: interface_utils
add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/utils)

# Creates interface library: interface_plugins
add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/plugins)

# Creates interface library: interface_models_asr
add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/models/asr)

# Creates interface library: interface_models_text_correction
add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/models/text_correction)
set_target_properties(
  pnks2t
  PROPERTIES
  LINKER_LANGUAGE CXX
  CXX_STANDARD 17
)
target_link_libraries(
  pnks2t
  PUBLIC
  interface_models_asr
  interface_models_text_correction
  interface_utils
  interface_plugins
  pthread
)

install (
  TARGETS
  pnks2t
  DESTINATION lib
)

install (
  DIRECTORY
  ${CMAKE_CURRENT_LIST_DIR}
  DESTINATION include
  FILES_MATCHING
  PATTERN "*.hpp"
  PATTERN "*.h"
)
