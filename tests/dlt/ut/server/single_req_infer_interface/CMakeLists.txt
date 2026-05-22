include_directories(
        ${THIRD_PARTY_OUTPUT_DIR}/prometheus-cpp/include
        ../utils
        ${PROJECT_SOURCE_DIR}/src/config_manager)
link_directories(${THIRD_PARTY_OUTPUT_DIR}/prometheus-cpp/lib)
target_link_libraries(mindieservice_endpoint_static
        prometheus-cpp::core
        prometheus-cpp::pull
)
build_test(server_infer_interface ut "mindieservice_endpoint_static;pd_grpc_proto_file;" "${PROJECT_SOURCE_DIR}/src/server/endpoint/single_req_infer_interface;${PROJECT_SOURCE_DIR}/src/server/endpoint/single_llm_req_handler;${PROJECT_SOURCE_DIR}/src/server/endpoint;${PROJECT_SOURCE_DIR}/src/server/endpoint/utils;${PROJECT_SOURCE_DIR}/src/server/include;${PROJECT_SOURCE_DIR}/src/include/request_response;${PROJECT_SOURCE_DIR}/src/include/utils;${PROJECT_SOURCE_DIR}/src/server/endpoint/grpc_proto;${PROJECT_SOURCE_DIR}/src/server/endpoint/grpc_wrapper;${PROJECT_SOURCE_DIR}/src/server/endpoint/http_wrapper;${CMAKE_CURRENT_SOURCE_DIR}/../utils")