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

#include <string>
#include <iostream>
#include <thread>
#include "utils.h"
#include "atb_speed/log.h"
#include "config.h"

namespace atb_torch {
constexpr uint64_t  DEFAULT_WORKSPACE_SIZE = 1024 * 1024 * 600;

Config &Config::Instance()
{
    static Config instance;
    return instance;
}

Config::Config()
{
    isUseTilingCopyStream_ = IsEnable("ATB_USE_TILING_COPY_STREAM");
    const char *taskQueueEnv = std::getenv("TASK_QUEUE_ENABLE");
    const char *blockingEnv = std::getenv("ASCEND_LAUNCH_BLOCKING");
    isTaskQueueEnable_ = !((taskQueueEnv != nullptr && std::string(taskQueueEnv) == "0") ||
                           (blockingEnv != nullptr && std::string(blockingEnv) == "1"));
    SetGlobalWorkspaceSize(DEFAULT_WORKSPACE_SIZE);

    ATB_SPEED_LOG_DEBUG("Config [IsUseTilingCopyStream:" << isUseTilingCopyStream_
                        << ", GlobalWorkspaceSize:" << GetGlobalWorkspaceSize()
                        << ", IsTaskQueueEnable:" << isTaskQueueEnable_ << "]");
}

Config::~Config() {}

bool Config::IsEnable(const char *env, bool enable) const
{
    const char *saveTensor = std::getenv(env);
    if (!saveTensor) {
        return enable;
    }
    return std::string(saveTensor) == "1";
}

bool Config::IsUseTilingCopyStream() const { return isUseTilingCopyStream_; }

bool Config::IsTaskQueueEnable() const { return isTaskQueueEnable_; }

uint64_t Config::GetGlobalWorkspaceSize() const { return globalWorkspaceSize_; }

void Config::SetGlobalWorkspaceSize(uint64_t size)
{
    if (size == globalWorkspaceSize_) {
        return;
    }

    globalWorkspaceSize_ = size;

    if (size == 0) {
        globalWorkspaceTensor_ = at::Tensor();
        return;
    }

    atb::TensorDesc tensorDesc;
    tensorDesc.dtype = ACL_UINT8;
    tensorDesc.format = ACL_FORMAT_ND;

    constexpr int KB_1 = 1024;
    tensorDesc.shape.dimNum = 2; // 2 dims， KB base
    tensorDesc.shape.dims[0] = KB_1;
    tensorDesc.shape.dims[1] = size / KB_1 + 1;

    globalWorkspaceTensor_ = Utils::CreateAtTensorFromTensorDesc(tensorDesc);
}

torch::Tensor &Config::GetGlobalWorkspaceTensor() { return globalWorkspaceTensor_; }
} // namespace atb_torch