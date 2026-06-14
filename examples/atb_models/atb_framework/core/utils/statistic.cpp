/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 * MindIE is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */
#include "atb_speed/utils/statistic.h"

namespace atb_speed {
thread_local Statistic g_statistic;

std::string Statistic::ToString() const
{
    return "totalTime:" + std::to_string(totalTime) + ", createTensorTime:" + std::to_string(createTensorTime) +
           ", planSetupTime:" + std::to_string(planSetupTime) + ", planAsyncTime:" + std::to_string(planAsyncTime) +
           ", planExecuteTime:" + std::to_string(planExecuteTime) +
           ", streamSyncTime:" + std::to_string(streamSyncTime) +
           ", tillingCopyTime:" + std::to_string(tillingCopyTime) +
           ", getBestKernelTime:" + std::to_string(getBestKernelTime) +
           ", kernelExecuteTime:" + std::to_string(kernelExecuteTime) +
           ", kernelCacheHitCount:" + std::to_string(kernelCacheHitCount) +
           ", kernelCacheMissCount:" + std::to_string(kernelCacheMissCount) +
           ", mallocTorchTensorSize:" + std::to_string(mallocTorchTensorSize);
}

void Statistic::Reset()
{
    totalTime = 0;
    createTensorTime = 0;
    planSetupTime = 0;
    planAsyncTime = 0;
    planExecuteTime = 0;
    streamSyncTime = 0;
    tillingCopyTime = 0;
    getBestKernelTime = 0;
    kernelExecuteTime = 0;
    kernelCacheHitCount = 0;
    kernelCacheMissCount = 0;
    mallocTorchTensorSize = 0;
}

Statistic &GetStatistic() { return g_statistic; }
} // namespace atb_speed