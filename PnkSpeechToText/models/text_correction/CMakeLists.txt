add_library(interface_models_text_correction INTERFACE)
find_package(kenlm REQUIRED)
find_library(BOOST_LOCALE_LIBRARY boost_locale)
target_sources(interface_models_text_correction
  INTERFACE
    ${CMAKE_CURRENT_LIST_DIR}/Corrector.cpp
    ${CMAKE_CURRENT_LIST_DIR}/WordToNum.cpp
    ${CMAKE_CURRENT_LIST_DIR}/KenLm.cpp
)
add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/punctner)

target_link_libraries(interface_models_text_correction
  INTERFACE
    punctner
    ${BOOST_LOCALE_LIBRARY}
    ${KENLM_LIBRARIES})

target_include_directories(
  interface_models_text_correction
  INTERFACE
    ${KENLM_INCLUDE_DIRS}
  )