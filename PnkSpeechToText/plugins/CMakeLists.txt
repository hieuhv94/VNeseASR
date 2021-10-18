cmake_minimum_required(VERSION 3.5.1)

find_library(ASOUND_LIBRARY asound)
find_library(AVCODEC_LIBRARY avcodec)
find_library(AVUTIL_LIBRARY avutil)
find_library(AVFORMAT_LIBRARY avformat)
find_library(SWRESAMPLE_LIBRARY swresample)
find_library(SSL_LIBRARY ssl)
find_library(CRYPTO_LIBRARY crypto)

add_library(interface_plugins INTERFACE)
target_sources(interface_plugins
    INTERFACE
    ${CMAKE_CURRENT_LIST_DIR}/files/FilePlugin.cpp
    ${CMAKE_CURRENT_LIST_DIR}/websocket/WebSocket.cpp
    ${CMAKE_CURRENT_LIST_DIR}/websocket/WebSocketPlugin.cpp
    ${CMAKE_CURRENT_LIST_DIR}/microphone/MicroPlugin.cpp
    ${CMAKE_CURRENT_LIST_DIR}/socket/SocketPlugin.cpp
)

target_link_libraries(
    interface_plugins
    INTERFACE
        ${ASOUND_LIBRARY}
        ${AVCODEC_LIBRARY}
        ${AVUTIL_LIBRARY}
        ${AVFORMAT_LIBRARY}
        ${SWRESAMPLE_LIBRARY}
        ${SSL_LIBRARY}
        ${CRYPTO_LIBRARY}
        interface_utils
)

target_include_directories(
    interface_plugins
  INTERFACE
  ${CMAKE_CURRENT_LIST_DIR}/..
)