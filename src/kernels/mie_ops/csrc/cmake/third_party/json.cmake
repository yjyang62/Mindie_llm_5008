# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

include_guard(GLOBAL)

if(json_FOUND)
    return()
endif()

unset(json_FOUND CACHE)
unset(JSON_INCLUDE CACHE)

set(JSON_INSTALL_PATH ${MINDIE_THIRD_PARTY_DIR}/nlohmannJson/)

find_path(JSON_INCLUDE
        NAMES nlohmann/json.hpp
        NO_CMAKE_SYSTEM_PATH
        NO_CMAKE_FIND_ROOT_PATH
        PATHS ${JSON_INSTALL_PATH}/include)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(json
        FOUND_VAR
        json_FOUND
        REQUIRED_VARS
        JSON_INCLUDE
        )

if(json_FOUND AND NOT FORCE_REBUILD_CANN_3RD)
    message("json found in ${JSON_INSTALL_PATH}, and not force rebuild cann third_party")
    set(JSON_INCLUDE_DIR ${JSON_INSTALL_PATH}/include)
    add_library(json INTERFACE IMPORTED)
else()
    message(FATAL_ERROR "json not found in ${JSON_INSTALL_PATH}, please build mindie 3rd first!")
endif()
