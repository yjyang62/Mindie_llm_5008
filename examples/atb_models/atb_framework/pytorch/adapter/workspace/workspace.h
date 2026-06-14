/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
 * MindIE is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

#ifndef ATB_SPEED_UTILS_WORKSPACE_H
#define ATB_SPEED_UTILS_WORKSPACE_H
#include <cstdint>
#include <memory>
#include <map>
#include "buffer_base.h"

namespace atb_speed {

class Workspace {
public:
    Workspace();
    ~Workspace();
    void *GetWorkspaceBuffer(uint64_t bufferSize, uint32_t bufferKey = 0);
    void ClearCache();
    int32_t GetCachedNum();

private:
    std::map<uint32_t, std::unique_ptr<BufferBase>> workspaceBuffer_;
};
} // namespace atb_speed
#endif