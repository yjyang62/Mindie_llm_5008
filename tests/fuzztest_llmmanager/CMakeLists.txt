# 
# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# AscendTransformerBoost is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# 
set(BOOST_DIR ${PROJECT_SOURCE_DIR}/lib)
find_library(BOOST_CHRONO_LIBRARY libboost_chrono.so.1.82.0 PATHS ${BOOST_DIR} NO_DEFAULT_PATH)
find_library(BOOST_THREAD_LIBRARY libboost_thread.so.1.82.0 PATHS ${BOOST_DIR} NO_DEFAULT_PATH)
file(GLOB_RECURSE SOURCE_FILES "${CMAKE_CURRENT_LIST_DIR}/*.cpp")
set(pybind11_DIR ${pybind11_DIR})
find_package(pybind11 CONFIG)

add_executable(mindie_llmmanager_fuzztest ${SOURCE_FILES})
target_compile_options(mindie_llmmanager_fuzztest PRIVATE "-Wno-float-equal" "-Wno-unused-function" "-Wno-unused-variable" "-Wno-unused-parameter" "-Wno-attributes")
target_include_directories(mindie_llmmanager_fuzztest PRIVATE ${THIRD_PARTY_OUTPUT_DIR}/googletest/include)
target_link_directories(mindie_llmmanager_fuzztest PRIVATE ${THIRD_PARTY_OUTPUT_DIR}/googletest/lib)
target_include_directories(mindie_llmmanager_fuzztest PRIVATE ${THIRD_PARTY_OUTPUT_DIR}/secodefuzz/include)
target_link_directories(mindie_llmmanager_fuzztest PRIVATE ${THIRD_PARTY_OUTPUT_DIR}/secodefuzz/lib)
target_include_directories(mindie_llmmanager_fuzztest PRIVATE ${THIRD_PARTY_OUTPUT_DIR}/mockcpp/include)
target_link_directories(mindie_llmmanager_fuzztest PRIVATE ${THIRD_PARTY_OUTPUT_DIR}/mockcpp/lib)

target_include_directories(mindie_llmmanager_fuzztest PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/src/include/request_response
    ${PROJECT_SOURCE_DIR}/src/llm_manager_v2/include
    ${PROJECT_SOURCE_DIR}/src/utils/include
    ${CMAKE_CURRENT_LIST_DIR}/helper
)

file(GLOB GRPC_LIBS "${PROJECT_SOURCE_DIR}/lib/grpc/*")
target_link_directories(mindie_llmmanager_fuzztest PRIVATE 
    ${PROJECT_SOURCE_DIR}/lib
    ${PROJECT_SOURCE_DIR}/lib/grpc)

target_link_libraries(mindie_llmmanager_fuzztest PRIVATE
    boundscheck
    ssl
    crypto
    ${BOOST_CHRONO_LIBRARY}
    ${BOOST_THREAD_LIBRARY}
    -Wl,--start-group
    -lpthread
    -Wl,--end-group
    ${GRPC_LIBS} 
    mindie_llm_manager   
)

target_link_libraries(mindie_llmmanager_fuzztest PRIVATE mindie_llm_utils cpu_logits_handler memory_bridge pybind11::embed ascendcl -lSecodefuzz -lgtest -lgtest_main -lmockcpp -lpthread -lm)
install(TARGETS mindie_llmmanager_fuzztest DESTINATION bin)