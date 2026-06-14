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

if(catlass_FOUND)
    return()
endif()

unset(catlass_FOUND CACHE)
unset(CATLASS_INCLUDE CACHE)

set(CATLASS_INSTALL_PATH ${MINDIE_THIRD_PARTY_DIR}/catlass)
set(CATLASS_INCLUDE_DIR ${CATLASS_INSTALL_PATH}/include)

find_path(CATLASS_INCLUDE
        NAMES catlass/catlass.hpp
        NO_CMAKE_SYSTEM_PATH
        NO_CMAKE_FIND_ROOT_PATH
        PATHS ${CATLASS_INSTALL_PATH}/include)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(catlass
        FOUND_VAR
        catlass_FOUND
        REQUIRED_VARS
        CATLASS_INCLUDE
        )

if(catlass_FOUND AND NOT FORCE_REBUILD_CANN_3RD)
    message("catlass found in ${CATLASS_INSTALL_PATH}, and not force rebuild cann third_party")
    add_library(catlass INTERFACE IMPORTED)
    target_include_directories(catlass INTERFACE ${CATLASS_INCLUDE_DIR})
else()
    message(FATAL_ERROR "catlass not found in ${CATLASS_INSTALL_PATH}, please build mindie 3rd first!")
endif()
