cmake_minimum_required(VERSION 3.18)
set(CMAKE_CXX_STANDARD 17)

string(REPLACE "-Werror" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

execute_process(
    COMMAND bash -c "python3 -m pybind11 --cmakedir"
    OUTPUT_VARIABLE command_output
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
set(pybind11_DIR "${command_output}")
find_package(pybind11 REQUIRED)

#设置安全编译选项
set(CXX_FLAGS -ftrapv -D_FORTIFY_SOURCE=2 -fstack-protector-all)
string(REPLACE ";" " " CMAKE_CXX_FLAGS "${CXX_FLAGS} ${CMAKE_CXX_FLAGS}")

if (ENABLE_COVERAGE STREQUAL "ON")
    add_compile_options(--coverage)
    add_link_options(--coverage)
endif()

if(ENABLE_XSAN STREQUAL "ON")
    add_compile_options(-fsanitize=address -fno-omit-frame-pointer)
    add_link_options(-fsanitize=address)
endif()

include_directories(
        ${Python3_INCLUDE_DIRS}
        ${THIRD_PARTY_OUTPUT_DIR}/http/include
        ${THIRD_PARTY_OUTPUT_DIR}/openssl/include
        ${THIRD_PARTY_OUTPUT_DIR}/boost/include
        ${THIRD_PARTY_OUTPUT_DIR}/spdlog/include
        ${CMAKE_CURRENT_SOURCE_DIR}/common
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/config_manager
        ${CMAKE_CURRENT_SOURCE_DIR}/infer_instances
        ${CMAKE_CURRENT_SOURCE_DIR}/endpoint
        ${CMAKE_CURRENT_SOURCE_DIR}/endpoint/http_wrapper
        ${CMAKE_CURRENT_SOURCE_DIR}/endpoint/single_llm_req_handler
        ${CMAKE_CURRENT_SOURCE_DIR}/../config_manager
        ${CMAKE_CURRENT_SOURCE_DIR}/endpoint/utils
        ${CMAKE_CURRENT_SOURCE_DIR}/../include/request_response
        ${PROJECT_SOURCE_DIR}/include
        $ENV{ASCEND_HOME_PATH}/include
        ${pybind11_INCLUDE_DIRS}
)

link_directories(
        ${THIRD_PARTY_OUTPUT_DIR}/openssl/lib
        ${THIRD_PARTY_OUTPUT_DIR}/spdlog/lib
        ${THIRD_PARTY_OUTPUT_DIR}/boost/lib
        ${THIRD_PARTY_OUTPUT_DIR}/prometheus-cpp/lib
)

# If `msServiceProfiler/msServiceProfiler.h` exists, define macro `WITH_PROF`
if (EXISTS $ENV{ASCEND_HOME_PATH}/include/msServiceProfiler/Tracer.h AND
EXISTS $ENV{ASCEND_HOME_PATH}/include/msServiceProfiler/msServiceProfiler.h)
    add_definitions(-DWITH_PROF)
endif()

add_compile_definitions(CPPHTTPLIB_USE_POLL)

add_subdirectory(infer_instances)
add_subdirectory(endpoint)
add_subdirectory(tokenizer)
add_subdirectory(daemon)

# Add security compile options
include(options.cmake)

# Install config file
install(
    FILES ${CMAKE_CURRENT_SOURCE_DIR}/conf/config.json
    DESTINATION conf
    PERMISSIONS OWNER_READ OWNER_WRITE GROUP_READ
)
