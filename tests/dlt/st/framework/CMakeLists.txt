file(GLOB_RECURSE SOURCE_FILES "${CMAKE_CURRENT_LIST_DIR}/*.cpp")

add_library(dlt_framework STATIC ${SOURCE_FILES})

install(TARGETS dlt_framework DESTINATION lib)
