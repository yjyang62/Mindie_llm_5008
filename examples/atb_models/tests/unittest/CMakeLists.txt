file(GLOB_RECURSE SOURCE_FILES "${CMAKE_CURRENT_LIST_DIR}/*.cpp")

add_library(op_stub SHARED ../stub/aclnn_stub.cpp ../stub/atb_stub.cpp)
target_compile_options(op_stub PRIVATE "-Wno-float-equal" "-Wno-unused-function" "-Wno-unused-variable" "-Wno-unused-parameter" "-Wno-attributes")
target_link_libraries(op_stub PRIVATE atb opapi)
install(TARGETS op_stub DESTINATION lib)

add_executable(speed_unittest ${SOURCE_FILES} ../stub/aclnn_stub.cpp ../stub/atb_stub.cpp)

target_compile_options(speed_unittest PRIVATE "-Wno-float-equal" "-Wno-unused-function" "-Wno-unused-variable" "-Wno-unused-parameter" "-Wno-attributes" -fno-access-control)
target_include_directories(speed_unittest PRIVATE ${PROJECT_SOURCE_DIR}/third_party/googletest/include)
target_link_directories(speed_unittest PRIVATE ${PROJECT_SOURCE_DIR}/third_party/googletest/lib)
target_include_directories(speed_unittest PRIVATE ${PROJECT_SOURCE_DIR}/third_party/mockcpp/include)
target_link_directories(speed_unittest PRIVATE ${PROJECT_SOURCE_DIR}/third_party/mockcpp/lib)

target_link_libraries(speed_unittest PRIVATE atb_log atb atb_speed_torch atb_speed_core atb_speed_models atb_speed_operations ascendcl -lgtest -lgtest_main -lpthread -lm -lmockcpp)

install(TARGETS speed_unittest DESTINATION bin)