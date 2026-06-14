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

#include "workspace.h"
#include "atb_speed/log.h"
#include "buffer_device.h"

namespace atb_speed {

Workspace::Workspace()
{
    workspaceBuffer_[0].reset(new BufferDevice(629145600)); // 629145600 初始化空间大小
}

Workspace::~Workspace() {}

void Workspace::ClearCache()
{
    for (auto& buff : std::as_const(workspaceBuffer_)) {
        buff.second->ClearBuffer();
    }
}

int32_t Workspace::GetCachedNum()
{
    int32_t rt = 0;
    for (auto& buff : std::as_const(workspaceBuffer_)) {
        rt = rt + buff.second->GetCachedNum();
    }
    return rt;
}

void *Workspace::GetWorkspaceBuffer(uint64_t bufferSize, uint32_t bufferKey)
{
    if (workspaceBuffer_.count(bufferKey) == 0) {
        workspaceBuffer_[bufferKey].reset(new BufferDevice(bufferSize));
    }
    return workspaceBuffer_[bufferKey]->GetBuffer(bufferSize);
}

} // namespace atb_speed