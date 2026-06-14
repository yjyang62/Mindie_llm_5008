cmake_minimum_required(VERSION 3.18)
project("mindie_llm_log")
set(CMAKE_CXX_STANDARD 17)

file(GLOB_RECURSE SOURCE_FILES "${CMAKE_CURRENT_LIST_DIR}/*.cpp")

set(LIB_NAME "mindie_llm_log")
# atb需要单独编译，修改so名称
if("${CMAKE_CURRENT_BINARY_DIR}" MATCHES "atb_models")
    set(LIB_NAME "atb_log")
endif()

add_library(${LIB_NAME} SHARED ${SOURCE_FILES})

set(SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../..")
include_directories(
        ${SOURCE_DIR}/src/include/utils
        ${SOURCE_DIR}/src/utils/log
        ${SOURCE_DIR}/third_party/spdlog/include
        ${SOURCE_DIR}/third_party/nlohmannJson/include
        ${SOURCE_DIR}/examples/atb_models/third_party/spdlog/include
        ${SOURCE_DIR}/examples/atb_models/third_party/nlohmannJson/include
)

target_link_libraries(${LIB_NAME} PRIVATE stdc++fs)

set_target_properties(${LIB_NAME} PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/lib")

install(TARGETS ${LIB_NAME} DESTINATION lib)
