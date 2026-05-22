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

#ifndef ATB_SPEED_MODELS_ALL_TO_ALL_DISPATCH_OPERATION_H
#define ATB_SPEED_MODELS_ALL_TO_ALL_DISPATCH_OPERATION_H
#include <atb/atb_infer.h>
#include <atb/comm.h>
#include "atb_speed/utils/operation_util.h"
#include "atb_speed/log.h"

namespace atb_speed {
namespace common {
struct AllToAllDispatchParam {
    int topk = 1;
    int numOfExperts = 8;
    std::string backend = "hccl";
    HcclComm hcclComm = nullptr;
    bool hasMoeEp = false;
    int moeEpRank = 0;
    int moeEpSize = 1;
    std::string moeEpDomain = "";
    std::string moeEpRankTableFile = "";

    bool hasMlpTp = false;
    int mlpTpRank = 0;
    int mlpTpSize = 1;
    std::string mlpTpDomain = "";
    std::string mlpTpRankTableFile = "";
};

atb::Status CreateAllToAllDispatchOperation(const AllToAllDispatchParam &param, atb::Operation **operation);
}
} // namespace atb_speed
#endif