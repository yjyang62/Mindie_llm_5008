include(${CMAKE_CURRENT_LIST_DIR}/../utils.cmake)

file(READ "${DEPENDENCY_JSON_FILE}" DEP_JSON_STRING)
download_open_source("${OPENSOURCE_COMPONENT_NAME}" "${FILE_GLOB_PATTERN}" "${THIRD_PARTY_CACHE_DIR}")

set(THREAD_NUM "${THREAD_NUM}") # clean warning
list(JOIN THIRD_PARTY_CXX_FLAGS " " THIRD_PARTY_CXX_FLAGS_STR)

if(EXISTS "${PROMETHEUS_OUTPUT_DIR}/include/prometheus/exposer.h")
    message(STATUS "${OPENSOURCE_COMPONENT_NAME} already built, skipping.")
    return()
endif()

set(PKG_DOWNLOAD_DIR "${THIRD_PARTY_SRC_DIR}/${OPENSOURCE_COMPONENT_NAME}")
set(PROMETHEUS_BUILD_DIR "${PKG_DOWNLOAD_DIR}/build")
file(MAKE_DIRECTORY "${PROMETHEUS_BUILD_DIR}")
execute_process(COMMAND sed -i "1i add_link_options(-s)" ${PKG_DOWNLOAD_DIR}/CMakeLists.txt)
execute_process(COMMAND sed -i "1i add_link_options(-Wl,-z,relro,-z,now)" ${PKG_DOWNLOAD_DIR}/CMakeLists.txt)

execute_process(
    COMMAND cmake 
        -DCMAKE_BUILD_TYPE=Release
        -DBUILD_SHARED_LIBS=ON
        -DENABLE_PUSH=OFF
        -DENABLE_COMPRESSION=OFF
        -DENABLE_TESTING=OFF
        -DCMAKE_CXX_FLAGS=${THIRD_PARTY_CXX_FLAGS_STR}
        -DCMAKE_INSTALL_PREFIX=${PROMETHEUS_OUTPUT_DIR}
        ${PKG_DOWNLOAD_DIR}
    WORKING_DIRECTORY ${PROMETHEUS_BUILD_DIR}
    OUTPUT_QUIET
)
execute_process(
    COMMAND make clean
    WORKING_DIRECTORY ${PROMETHEUS_BUILD_DIR}
    OUTPUT_QUIET
)
execute_process(
    COMMAND make -j${THREAD_NUM}
    WORKING_DIRECTORY ${PROMETHEUS_BUILD_DIR}
    OUTPUT_QUIET
)
execute_process(
    COMMAND make install
    WORKING_DIRECTORY ${PROMETHEUS_BUILD_DIR}
    OUTPUT_QUIET
)
message(STATUS "${OPENSOURCE_COMPONENT_NAME} has been successfully built and installed to ${PROMETHEUS_OUTPUT_DIR}")
