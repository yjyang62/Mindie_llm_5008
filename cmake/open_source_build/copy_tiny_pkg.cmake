include(${CMAKE_CURRENT_LIST_DIR}/../utils.cmake)
function(download_copy_pkg PKG_NAME CORE_FILE OUTPUT_FILE COPY_SRC COPY_DST)
    file(READ "${DEPENDENCY_JSON_FILE}" DEP_JSON_STRING)
    download_open_source("${PKG_NAME}" "${CORE_FILE}" "${THIRD_PARTY_CACHE_DIR}")

    if(EXISTS "${OUTPUT_FILE}")
        message(STATUS "${PKG_NAME} already copy, skipping.")
        return()
    endif()

    file(COPY "${COPY_SRC}" DESTINATION "${COPY_DST}")
    message(STATUS "Successfully copied ${PKG_NAME} content.")
endfunction()


function(copy_gloo_from_pytorch)
    set(GLOO_SRC_DIR "${THIRD_PARTY_SRC_DIR}/pytorch/third_party/gloo")
    set(GLOO_OUT_DIR  "${THIRD_PARTY_OUTPUT_DIR}/gloo")

    if(EXISTS "${GLOO_OUT_DIR}/gloo/algorithm.h")
        message(STATUS "gloo already installed (pytorch), skipping.")
        return()
    endif()

    if(NOT EXISTS "${GLOO_SRC_DIR}/gloo/algorithm.h")
        message(FATAL_ERROR "pytorch gloo not found at ${GLOO_SRC_DIR}")
    endif()

    file(COPY "${GLOO_SRC_DIR}" DESTINATION "${THIRD_PARTY_OUTPUT_DIR}")
    message(STATUS "Pytorch gloo copied successfully.")
endfunction()


set(THREAD_NUM "${THREAD_NUM}") # clean warning
set(FILE_GLOB_PATTERN "${FILE_GLOB_PATTERN}") # clean warning
set(OPENSOURCE_COMPONENT_NAME "${OPENSOURCE_COMPONENT_NAME}") # clean warning

download_copy_pkg("spdlog" "include/spdlog/spdlog.h" "${THIRD_PARTY_OUTPUT_DIR}/spdlog/include/spdlog/spdlog.h"
    "${THIRD_PARTY_SRC_DIR}/spdlog/include" "${THIRD_PARTY_OUTPUT_DIR}/spdlog"
)

download_copy_pkg("nlohmannJson" "include/nlohmann/json.hpp" "${THIRD_PARTY_OUTPUT_DIR}/nlohmannJson/include/nlohmann/json.hpp"
    "${THIRD_PARTY_SRC_DIR}/nlohmannJson/include" "${THIRD_PARTY_OUTPUT_DIR}/nlohmannJson"
)

download_copy_pkg("makeself" "makeself.sh" "${THIRD_PARTY_OUTPUT_DIR}/makeself/makeself.sh"
    "${THIRD_PARTY_SRC_DIR}/makeself" "${THIRD_PARTY_OUTPUT_DIR}"
)

download_copy_pkg("http" "httplib.h" "${THIRD_PARTY_OUTPUT_DIR}/http/httplib.h"
    "${THIRD_PARTY_SRC_DIR}/http" "${THIRD_PARTY_OUTPUT_DIR}"
)

if(NOT (DEFINED ENV{BUILD_ZONE} AND "$ENV{BUILD_ZONE}" STREQUAL "yellow"))
    download_copy_pkg("hseceasy_${PLATFORM}" "lib/libsecurec.so"
        "${THIRD_PARTY_OUTPUT_DIR}/hseceasy_${PLATFORM}/lib/libsecurec.so"
        "${THIRD_PARTY_SRC_DIR}/hseceasy_${PLATFORM}" "${THIRD_PARTY_OUTPUT_DIR}"
    )
    download_copy_pkg("gloo" "gloo/algorithm.h" "${THIRD_PARTY_OUTPUT_DIR}/gloo/gloo/algorithm.h"
        "${THIRD_PARTY_SRC_DIR}/gloo" "${THIRD_PARTY_OUTPUT_DIR}"
    )
else()
    copy_gloo_from_pytorch()
endif()
