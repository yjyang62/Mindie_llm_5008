include(${CMAKE_CURRENT_LIST_DIR}/../utils.cmake)

file(READ "${DEPENDENCY_JSON_FILE}" DEP_JSON_STRING)
download_open_source("${OPENSOURCE_COMPONENT_NAME}" "${FILE_GLOB_PATTERN}" "${THIRD_PARTY_CACHE_DIR}")

set(THREAD_NUM "${THREAD_NUM}") # clean warning
list(JOIN " " THIRD_PARTY_CXX_FLAGS_STR ${THIRD_PARTY_CXX_FLAGS})

if(EXISTS "${OPENSSL_OUTPUT_DIR}/include/openssl/aes.h")
    message(STATUS "${OPENSOURCE_COMPONENT_NAME} already built, skipping.")
    return()
endif()

set(PKG_DOWNLOAD_DIR "${THIRD_PARTY_SRC_DIR}/${OPENSOURCE_COMPONENT_NAME}")
execute_process(COMMAND chmod +x config WORKING_DIRECTORY ${PKG_DOWNLOAD_DIR})
execute_process(COMMAND dos2unix config WORKING_DIRECTORY ${PKG_DOWNLOAD_DIR})
execute_process(COMMAND dos2unix Configure WORKING_DIRECTORY ${PKG_DOWNLOAD_DIR})
execute_process(COMMAND chmod +x Configure WORKING_DIRECTORY ${PKG_DOWNLOAD_DIR})

file(MAKE_DIRECTORY "${OPENSSL_OUTPUT_DIR}")
set(LINK_FLAGS_STR "-Wl,-z,now -s")
execute_process(
    COMMAND ./config --prefix=${OPENSSL_OUTPUT_DIR}
                    --libdir=${OPENSSL_OUTPUT_DIR}/lib
                    no-unit-test
                    no-tests
                    no-external-tests
                    CXXFLAGS=${THIRD_PARTY_CXX_FLAGS_STR}
                    LDFLAGS=${LINK_FLAGS_STR}
                    AR=/usr/bin/ar
                    RANLIB=/usr/bin/ranlib
    WORKING_DIRECTORY ${PKG_DOWNLOAD_DIR}
    OUTPUT_QUIET
)

execute_process(COMMAND make clean WORKING_DIRECTORY ${PKG_DOWNLOAD_DIR} OUTPUT_QUIET)
execute_process(COMMAND make -j${THREAD_NUM} WORKING_DIRECTORY ${PKG_DOWNLOAD_DIR} OUTPUT_QUIET)
execute_process(COMMAND make install_sw WORKING_DIRECTORY ${PKG_DOWNLOAD_DIR} OUTPUT_QUIET)
message(STATUS "${OPENSOURCE_COMPONENT_NAME} has been successfully built and installed to ${OPENSSL_OUTPUT_DIR}")
