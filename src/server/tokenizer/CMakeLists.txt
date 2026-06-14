file(GLOB TOKENIZER_SRCS ${CMAKE_CURRENT_LIST_DIR}/*.cpp)

execute_process(
    COMMAND bash -c "python3 -m pybind11 --cmakedir"
    OUTPUT_VARIABLE command_output
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
set(pybind11_DIR "${command_output}")
find_package(pybind11 REQUIRED)

add_compile_options(-fPIC)

add_library(mindieservice_tokenizer SHARED ${TOKENIZER_SRCS})

target_link_libraries(mindieservice_tokenizer PUBLIC config_manager mindie_llm_utils ${RE2_SO_FILES} pybind11::embed)

install(TARGETS mindieservice_tokenizer LIBRARY DESTINATION lib)
