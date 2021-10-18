cmake_minimum_required(VERSION 3.5.1)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# File examples
add_executable(audioFileExample ${CMAKE_CURRENT_LIST_DIR}/audioFileExample.cpp)
target_compile_definitions(audioFileExample PUBLIC KENLM_MAX_ORDER=6)
target_link_libraries(
    audioFileExample
    PUBLIC
        pnks2t
        fbgemm
        glog
        ssl
        crypto
)

target_include_directories(
    audioFileExample
    PUBLIC
    /opt/kenlm/include/kenlm/
    /usr/local/include/PnkSpeechToText
)

# Microphone examples
add_executable(microphoneExample ${CMAKE_CURRENT_LIST_DIR}/microphoneExample.cpp)
target_compile_definitions(microphoneExample PUBLIC KENLM_MAX_ORDER=6)
target_link_libraries(
    microphoneExample
    PUBLIC
        pnks2t
        fbgemm
        glog
        ssl
        crypto
)

target_include_directories(
    microphoneExample
    PUBLIC
    /opt/kenlm/include/kenlm/
    /usr/local/include/PnkSpeechToText
)

# Websocket examples
add_executable(websocketExample ${CMAKE_CURRENT_LIST_DIR}/websocketExample.cpp)
target_compile_definitions(websocketExample PUBLIC KENLM_MAX_ORDER=6)
target_link_libraries(
    websocketExample
    PUBLIC
        pnks2t
        fbgemm
        glog
        ssl
        crypto
)

target_include_directories(
    websocketExample
    PUBLIC
    /opt/kenlm/include/kenlm/
    /usr/local/include/PnkSpeechToText
)

# Socket examples
add_executable(socketExample ${CMAKE_CURRENT_LIST_DIR}/socketExample.cpp)
target_compile_definitions(socketExample PUBLIC KENLM_MAX_ORDER=6)
target_link_libraries(
    socketExample
    PUBLIC
        pnks2t
        fbgemm
        glog
        ssl
        crypto
)

target_include_directories(
    socketExample
    PUBLIC
    /opt/kenlm/include/kenlm/
    /usr/local/include/PnkSpeechToText
)
