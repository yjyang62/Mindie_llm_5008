file(GLOB_RECURSE SOURCE_FILES "${CMAKE_CURRENT_LIST_DIR}/*.cpp")

add_library(sequence STATIC ${SOURCE_FILES})

target_link_libraries(sequence PUBLIC mindie_llm_log)
install(TARGETS sequence DESTINATION lib)
