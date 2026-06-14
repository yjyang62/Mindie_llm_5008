/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * MindIE is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

#ifndef ATB_SPEED_MODELS_ALL2ALL_MATMUL_OPERATION_H
#define ATB_SPEED_MODELS_ALL2ALL_MATMUL_OPERATION_H
#include <atb/atb_infer.h>
#include "atb_speed/utils/operation_util.h"
#include "atb_speed/log.h"
#include "operations/fusion/linear/linear.h"
#include "operations/fusion/linear/linear_parallel.h"
#include "operations/fusion/norm/norm_linear.h"

namespace atb_speed {
namespace common {
constexpr int DEFAULT_TOPK_SCALE = -1;

struct All2AllMatmulParam {
    int32_t topk = 2;
    uint32_t numOfExperts = 8;
    uint32_t numOfDeviceExperts = 8;
    bool gateUpTransposeB = false;
    bool downTransposeB = false;
    bool enableExpertCumSumOutput = false;
    int32_t scaledTopk = -1;
    int moeEpRank = 0;
    int moeEpSize = 1;
    std::string lcclMoeEpDomain = "";
    HcclComm lcclMoeEpHcclComm = nullptr;
};

atb::Status CreateAll2AllMatmulOperation(const All2AllMatmulParam &param, atb::Operation **operation);
}
} // namespace atb_speed
#endif