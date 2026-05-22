# find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
# set(ENV{PYTHONPATH} "${PROJECT_DIR}:$ENV{PYTHONPATH}")
# find_package(pybind11 REQUIRED)
# message("**********************${Python3_LIBRARIES}****************")
# set(ENV{PYTHONPATH} "/home/guan/code/05_mindIe/branch/MindIE-LLM-huangyao/build/tests/dlt/ut/server/tokenizer")
# export  PYTHONPATH=/home/guan/code/05_mindIe/branch/MindIE-LLM-huangyao/build/tests/dlt/ut/server/tokenizer
execute_process(
    COMMAND bash -c "python3 -m pybind11 --cmakedir"
    OUTPUT_VARIABLE command_output
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

message(${command_output})
set(pybind11_DIR "${command_output}")
find_package(pybind11 REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

include_directories(
        ${PROJECT_SOURCE_DIR}/src/config_manager)

build_test(server_tokenizer ut "mindieservice_tokenizer;config_manager;pybind11::embed;pybind11::pybind11;${Python3_LIBRARIES};boundscheck" "${PROJECT_SOURCE_DIR}/src/server/tokenizer;${PROJECT_SOURCE_DIR}/src/server/include;${PROJECT_SOURCE_DIR}/src/server/common;${Python_INCLUDE_DIRS};${PROJECT_SOURCE_DIR}/src;${Python3_INCLUDE_DIRS}")