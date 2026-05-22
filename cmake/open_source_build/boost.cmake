include(${CMAKE_CURRENT_LIST_DIR}/../utils.cmake)

file(READ "${DEPENDENCY_JSON_FILE}" DEP_JSON_STRING)
download_open_source("${OPENSOURCE_COMPONENT_NAME}" "${FILE_GLOB_PATTERN}" "${THIRD_PARTY_CACHE_DIR}")
list(JOIN THIRD_PARTY_CXX_FLAGS " " THIRD_PARTY_CXX_FLAGS_STR)

set(THREAD_NUM "${THREAD_NUM}") # clean warning
if(EXISTS "${BOOST_OUTPUT_DIR}/lib")
    message(STATUS "${OPENSOURCE_COMPONENT_NAME} already built, skipping.")
    return()
endif()

file(MAKE_DIRECTORY "${BOOST_OUTPUT_DIR}")
set(PKG_DOWNLOAD_DIR "${THIRD_PARTY_SRC_DIR}/${OPENSOURCE_COMPONENT_NAME}")
execute_process(
    COMMAND chmod +x ./bootstrap.sh
    COMMAND chmod +x ./tools/build/src/engine/build.sh
    WORKING_DIRECTORY "${PKG_DOWNLOAD_DIR}"
    RESULT_VARIABLE chmod_result
)
if(NOT chmod_result EQUAL 0)
    message(FATAL_ERROR "Failed to set executable permissions for Boost scripts.")
endif()

execute_process(
    COMMAND ./bootstrap.sh
    WORKING_DIRECTORY ${PKG_DOWNLOAD_DIR}
    RESULT_VARIABLE bootstrap_result OUTPUT_QUIET
)
if(NOT bootstrap_result EQUAL 0)
    message(FATAL_ERROR "Failed to run bootstrap.sh for Boost.")
endif()

set(BOOST_LINK_FLAGS_STR "-Wl,-z,now -s")
execute_process(
    COMMAND ./b2 toolset=gcc
            -j${THREAD_NUM}
            --disable-icu --with-thread --with-regex --with-log
            --with-filesystem --with-date_time --with-chrono --with-system
            cxxflags=${THIRD_PARTY_CXX_FLAGS_STR}
            linkflags=${BOOST_LINK_FLAGS_STR}
            link=shared
            threading=multi variant=release stage
            --prefix=${BOOST_OUTPUT_DIR} install
    WORKING_DIRECTORY ${PKG_DOWNLOAD_DIR}
    RESULT_VARIABLE b2_result
    OUTPUT_QUIET
)
if(NOT b2_result EQUAL 0)
    message(FATAL_ERROR "Failed to build and install Boost.${b2_result}")
endif()
message(STATUS "${OPENSOURCE_COMPONENT_NAME} has been successfully built and installed to ${BOOST_OUTPUT_DIR}")
