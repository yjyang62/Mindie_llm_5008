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

#ifndef ATB_SPEED_MODELS_EXPERT_FILTER_OPERATION_H
#define ATB_SPEED_MODELS_EXPERT_FILTER_OPERATION_H
#include <atb/atb_infer.h>
#include "atb_speed/utils/operation_util.h"
#include "atb_speed/log.h"

namespace atb_speed {
namespace common {
struct ExpertFilterParam {
    bool shiftedTopK = true;
    bool isBF16 = false;
    bool enableGatingDp = false;
    long unsigned int numOfExperts = 8;
    std::vector<int32_t> deviceExpert = {0, 1, 2, 3, 4, 5, 6, 7};
};

atb::Status CreateExpertFilterOperation(const ExpertFilterParam &param, atb::Operation **operation);
}
} // namespace atb_speed
#endif