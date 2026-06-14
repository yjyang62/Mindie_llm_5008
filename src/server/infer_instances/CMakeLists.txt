file(GLOB_RECURSE BACKEND_MGR_SRCS ${CMAKE_CURRENT_LIST_DIR}/*.cc ${CMAKE_CURRENT_LIST_DIR}/*.cpp)

add_library(mindieservice_infer_instances SHARED ${BACKEND_MGR_SRCS})

target_link_libraries(mindieservice_infer_instances
        PUBLIC
        config_manager
        mindie_llm_utils
        mindie_llm_manager_v2
)

install(TARGETS mindieservice_infer_instances
        LIBRARY DESTINATION lib
)
