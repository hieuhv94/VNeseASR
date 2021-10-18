add_library(punctner INTERFACE)

target_sources(punctner
  INTERFACE
    ${CMAKE_CURRENT_LIST_DIR}/tf.c
    ${CMAKE_CURRENT_LIST_DIR}/PunctNer.cpp
)
target_link_libraries(punctner
  INTERFACE
  tensorflow
  jsoncpp
)
