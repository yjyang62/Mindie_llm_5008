/*
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

#ifndef ATB_SPEED_UTILS_STATISTIC_H
#define ATB_SPEED_UTILS_STATISTIC_H
#include <string>
#include <cstdint>

namespace atb_speed {
struct Statistic {
    uint64_t totalTime = 0;
    uint64_t createTensorTime = 0;
    uint64_t planSetupTime = 0;
    uint64_t planAsyncTime = 0;
    uint64_t planExecuteTime = 0;
    uint64_t streamSyncTime = 0;
    uint64_t tillingCopyTime = 0;
    uint64_t getBestKernelTime = 0;
    uint64_t kernelExecuteTime = 0;
    uint64_t kernelCacheHitCount = 0;
    uint64_t kernelCacheMissCount = 0;
    uint64_t mallocTorchTensorSize = 0;

    std::string ToString() const;
    void Reset();
};

Statistic &GetStatistic();
} // namespace atb_speed
#endif