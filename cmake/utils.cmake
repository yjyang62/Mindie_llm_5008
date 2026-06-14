include(FetchContent)
set(PLATFORM "${CMAKE_SYSTEM_PROCESSOR}")


function(_download_file_with_retry url cache_file do_verify expected_sha256)
    set(max_retries 5)
    set(attempt 1)
    while(attempt LESS_EQUAL max_retries)
        message(STATUS "Downloading (${attempt}/${max_retries}): ${url}")
        file(DOWNLOAD "${url}" "${cache_file}"
            TLS_VERIFY OFF
            STATUS st
        )
        list(GET st 0 code)

        if(code EQUAL 0)
            if(do_verify)
                file(SHA256 "${cache_file}" current_sha256)
                if(current_sha256 STREQUAL expected_sha256)
                    message(STATUS "Download OK and SHA256 verified: ${cache_file}")
                    return()
                else()
                    message(WARNING "SHA256 mismatch after download, retrying...")
                    file(REMOVE "${cache_file}")
                endif()
            else()
                message(STATUS "Download OK and SHA256 check disabled: ${cache_file}")
                return()
            endif()
        else()
            message(WARNING "Download failed with code ${code}, retrying...")
            file(REMOVE "${cache_file}")
        endif()
        execute_process(COMMAND ${CMAKE_COMMAND} -E sleep 0.001)
        math(EXPR attempt "${attempt} + 1")
    endwhile()
    message(FATAL_ERROR "Failed to download file after ${max_retries} attempts: ${url}")
endfunction()


function(_download_file url cache_file expected_sha256)
    set(do_verify TRUE)
    if(NOT expected_sha256)
        set(do_verify FALSE)
    endif()

    if(EXISTS "${cache_file}" AND do_verify)
        file(SHA256 "${cache_file}" current_sha256)
        if(current_sha256 STREQUAL expected_sha256)
            message(STATUS "File already exists and passed SHA256 check: ${cache_file}")
            return()
        else()
            message(WARNING "SHA256 mismatch, redownloading: ${cache_file}")
            file(REMOVE "${cache_file}")
        endif()
    elseif(EXISTS "${cache_file}" AND NOT do_verify)
        message(STATUS "File exists and SHA256 check disabled: ${cache_file}")
        return()
    endif()
    _download_file_with_retry("${url}" "${cache_file}" "${do_verify}" "${expected_sha256}")
endfunction()


function(_is_complete_source is_complete dir file_pattern)
    if(NOT EXISTS "${dir}")
        set(${is_complete} FALSE PARENT_SCOPE)
        return()
    endif()

    file(GLOB entries "${dir}/*")
    if(NOT entries)
        set(${is_complete} FALSE PARENT_SCOPE)
        return()
    endif()

    file(GLOB core "${dir}/${file_pattern}")
    if(NOT core)
        message(STATUS "Missing core file: ${file_pattern}")
        set(${is_complete} FALSE PARENT_SCOPE)
        return()
    endif()
    set(${is_complete} TRUE PARENT_SCOPE)
endfunction()


function(_download_open_source_with_path opensource_name src_path cache_dir)
    string(JSON DEPENDENCIES_COUNT LENGTH "${DEP_JSON_STRING}")
    math(EXPR DEPENDENCIES_COUNT "${DEPENDENCIES_COUNT} - 1")

    if(src_path AND NOT src_path STREQUAL "")
        set(src_dir "${src_path}")
    else()
        message(FATAL_ERROR 
            "Dependency '${opensource_name}' requires a valid src_path, "
            "but none was provided. Please specify a non-empty source path."
        )
    endif()

    set(found_dep FALSE)
    foreach(INDEX RANGE "${DEPENDENCIES_COUNT}")
        string(JSON DEP_NAME MEMBER "${DEP_JSON_STRING}" "${INDEX}")

        if(NOT "${DEP_NAME}" STREQUAL "${opensource_name}")
            continue()
        endif()
        set(found_dep TRUE)
        # Get 3rd package type.
        string(JSON DEP_TYPE GET "${DEP_JSON_STRING}" "${DEP_NAME}" "type")
        # file type
        if("${DEP_TYPE}" STREQUAL "file")
            string(JSON DEP_URL GET "${DEP_JSON_STRING}" "${DEP_NAME}" "url")
            string(JSON CACHE_FILE_NAME GET "${DEP_JSON_STRING}" "${DEP_NAME}" "fileName")

            if(NOT "${DEP_URL}" STREQUAL "")
                string(CONFIGURE "${DEP_URL}" download_url)
                message(STATUS "Begin to download ${DEP_NAME} from ${download_url}")
                string(JSON SHA256 GET "${DEP_JSON_STRING}" "${DEP_NAME}" "sha256")
                _download_file("${download_url}" "${cache_dir}/${CACHE_FILE_NAME}" "${SHA256}")
                FetchContent_Declare(
                    "${DEP_NAME}"
                    URL file://${cache_dir}/${CACHE_FILE_NAME}
                    SOURCE_DIR "${src_dir}"
                )
                FetchContent_Populate(${DEP_NAME})
            endif()
        endif()
        # git type
        if("${DEP_TYPE}" STREQUAL "git")
            string(JSON DEP_REPO GET "${DEP_JSON_STRING}" "${DEP_NAME}" "url")
            string(JSON DEP_TAG GET "${DEP_JSON_STRING}" "${DEP_NAME}" "tag")

            if(NOT "${DEP_REPO}" STREQUAL "")
                message(STATUS "Begin to download ${DEP_NAME} from ${DEP_REPO}")
                FetchContent_Declare(
                    "${DEP_NAME}"
                    SOURCE_DIR "${src_dir}"
                    GIT_REPOSITORY "${DEP_REPO}"
                    GIT_TAG "${DEP_TAG}"
                )
                FetchContent_Populate(${DEP_NAME})
            endif()
        endif()
        break()
        message(FATAL_ERROR "Unknown dependency type: ${DEP_TYPE}")
    endforeach()

    if(NOT found_dep)
        message(FATAL_ERROR "Dependency not found: ${opensource_name}")
    endif()
    message(STATUS "Successfully downloaded ${opensource_name}")
endfunction()


function(download_open_source pkg_name file_pattern output_dir)
    set(PKG_DIR "${THIRD_PARTY_SRC_DIR}/${pkg_name}")
    _is_complete_source(complete "${PKG_DIR}" "${file_pattern}")
    if(complete)
        message(STATUS "Download completed: ${pkg_name}")
        return()
    endif()

    if(EXISTS "${PKG_DIR}")
        message(STATUS "Cleaning ${PKG_DIR}")
        file(REMOVE_RECURSE "${PKG_DIR}")
    endif()
    if(NOT EXISTS "${output_dir}")
        file(MAKE_DIRECTORY "${output_dir}")
    endif()
    _download_open_source_with_path("${pkg_name}" "${PKG_DIR}" "${output_dir}")
endfunction()


function(apply_patches COMPONENT_SRC_DIR FILE_GLOB_PATTERN)
    file(GLOB PKG_LIST "${COMPONENT_SRC_DIR}/SOURCE/${FILE_GLOB_PATTERN}")
    list(GET PKG_LIST 0 PKG_DIR)

    if(NOT PKG_DIR)
        message(FATAL_ERROR "No package directory found in ${COMPONENT_SRC_DIR}/SOURCE")
    endif()

    file(GLOB SPEC_FILE "${COMPONENT_SRC_DIR}/*.spec")
    if(NOT SPEC_FILE)
        message(FATAL_ERROR "Spec file not found in ${COMPONENT_SRC_DIR}")
    endif()

    file(STRINGS "${SPEC_FILE}" PATCH_LINES REGEX "^Patch[0-9]*:")
    foreach(LINE ${PATCH_LINES})
        string(REGEX REPLACE "^Patch[0-9]*:[ \t]*" "" PATCH_FILE "${LINE}")
        set(PATCH_PATH "${COMPONENT_SRC_DIR}/${PATCH_FILE}")
        if(EXISTS "${PATCH_PATH}")
            execute_process(
                COMMAND patch -p1
                WORKING_DIRECTORY ${PKG_DIR}
                INPUT_FILE ${PATCH_PATH}
                OUTPUT_QUIET
            )
        else()
            message(WARNING "Patch file not found: ${PATCH_PATH}")
        endif()
    endforeach()
endfunction()


function(get_ABI_option_value)
    if(DEFINED USE_CXX11_ABI)
        if(${USE_CXX11_ABI} STREQUAL "0" OR ${USE_CXX11_ABI} STREQUAL "1")
message("
==================================
       Using ABI = ${USE_CXX11_ABI}
==================================
")
        return()
        endif()
    endif()

    execute_process(
        COMMAND python3 ${CMAKE_CURRENT_LIST_DIR}/../scripts/get_cxx11_abi_flag.py -f torch
        OUTPUT_VARIABLE _CXX_ABI_FLAG
        RESULT_VARIABLE _status
        ERROR_QUIET
    )
    if(NOT _status EQUAL 0)
        message(WARNING "Failed to get ABI flag from torch, using default = 0")
        set(USE_CXX11_ABI 0 PARENT_SCOPE)
        return()
    endif()

    string(STRIP "${_CXX_ABI_FLAG}" _CXX_ABI_FLAG)
    if(NOT _CXX_ABI_FLAG MATCHES "^[01]$")
        message(WARNING "Invalid ABI flag '${_CXX_ABI_FLAG}', using default = 0")
        set(USE_CXX11_ABI 0 PARENT_SCOPE)
        return()
    endif()
message("
==================================
       Using ABI = ${_CXX_ABI_FLAG}
==================================
")
    set(USE_CXX11_ABI ${_CXX_ABI_FLAG} PARENT_SCOPE)
endfunction()


function(ensure_aslr_level target_level)
    execute_process(
        COMMAND cat /proc/sys/kernel/randomize_va_space
        OUTPUT_VARIABLE current_level
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )

    if(NOT current_level STREQUAL "${target_level}")
        message(STATUS "Current ASLR level is ${current_level}, updating to ${target_level}")
        execute_process(
            COMMAND sudo sh -c "echo ${target_level} > /proc/sys/kernel/randomize_va_space"
            RESULT_VARIABLE result
        )
        if(NOT result EQUAL 0)
            message(WARNING "Failed to set ASLR level. You may need sudo permission.")
        else()
            message(STATUS "ASLR level successfully updated to ${target_level}")
        endif()
    else()
        message(STATUS "ASLR level is already ${target_level}, no change needed.")
    endif()
endfunction()


function(get_architecture ARCH)
    execute_process(COMMAND arch COMMAND tr -d "\n" OUTPUT_VARIABLE ARCHITECTURE)
    if (ARCHITECTURE STREQUAL "x86_64")
        message("Compiling PS lib for architecture: ${ARCHITECTURE}")
        add_compile_options(-mavx)
    elseif (ARCHITECTURE STREQUAL "aarch64")
        message("Compiling PS lib for architecture: ${ARCHITECTURE}")
    else()
        message(FATAL_ERROR "The target arch is not supported: ${ARCHITECTURE}")
    endif()
    set(${ARCH} "${ARCHITECTURE}" PARENT_SCOPE)
endfunction()


function(find_pytorch OUT_VAR)
    execute_process(
        COMMAND python3 -c "import torch, os; print(os.path.dirname(os.path.abspath(torch.__file__)))"
        OUTPUT_VARIABLE TORCH_PATH
        ERROR_VARIABLE PYTHON_ERROR
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    string(FIND "${PYTHON_ERROR}" "ModuleNotFoundError" TORCH_ERROR_FOUND)
    if(NOT TORCH_ERROR_FOUND EQUAL -1)
        message(FATAL_ERROR "Python 'torch' module not found!")
    endif()
    set(${OUT_VAR} "${TORCH_PATH}" PARENT_SCOPE)
    message(STATUS "Found Python torch at: ${TORCH_PATH}")
endfunction()
