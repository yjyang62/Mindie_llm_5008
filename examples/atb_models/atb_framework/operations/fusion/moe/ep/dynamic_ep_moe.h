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

#ifndef ATB_SPEED_MODELS_DYNAMIC_EP_MOE_OPERATION_H
#define ATB_SPEED_MODELS_DYNAMIC_EP_MOE_OPERATION_H
#include <atb/atb_infer.h>
#include <atb/comm.h>
#include "atb_speed/utils/operation_util.h"
#include "atb_speed/log.h"
#include "operations/fusion/linear/linear.h"
#include "operations/fusion/linear/linear_parallel.h"
#include "operations/fusion/norm/norm_linear.h"

namespace atb_speed {
namespace common {
struct DynamicEpMoEParam {
    bool transpose = true;
    bool supportSwiGLU = true;
    int32_t topk = 2;
    int32_t scaledTopk = -1; /// 非deepseek模型默认不启用scaledTopk特性
    bool enableInitRoutingCutoff = false;  /// A flag indicating whether to use scaled topk option
    int gmmQuantType = 0;
    uint32_t numOfExperts = 8;
    uint32_t numOfDeviceExperts = 8;
    std::vector<int> moeLinearQuantType = {};
    bool hasBias = false;
    bool isBF16 = false;
    bool gateUpTransposeB = false;
    bool downTransposeB = false;
    bool isDynamicEp = false;
    int packQuantType = atb_speed::common::PackQuantType::ALL_FP;
    int denseQuantType = atb_speed::common::PackQuantType::PACK_QUANT_UNDEFINED;
    int quantGroupSize = 0; /// Group size of per-group quantization

    std::vector<int32_t> deviceExpert = {0, 1, 2, 3, 4, 5, 6, 7};
    int expertParallelDegree = 0;
    bool enableFusedRouting = false;
    bool enableGMMSwigluQuant = false;
    bool enableInitQuant = false;
    bool enableSwigluQuant = false;
    bool enableAtlasGMMFused = false;
    bool enableFusedTopk = false;
    bool enableDispatchCombineV2 = false; /// A flag indicating whether to use dispatch_v2 and combine_v2

    bool enableCVOverlap = false; /// A flag indicating whether the model use cube and vector parallel
    bool enableMoeDistribute = false;
    bool enableExpertCumSumOutput = false;
    bool enableGatingDp = false;
    int64_t numDanglingSharedExperts = 0;
    uint32_t numOfRedundantExpert = 0;

    int maxDecodeDpTokenSize = 0;
    std::string routingMethod = "";

    bool enableNodeBaseAll2All = false;
    HcclComm dispatchAndCombineHcclComm = nullptr;
    std::string dispatchAndCombinecommDomain = "";

    bool hasMoeEp = false;
    atb_speed::common::ParallelInfo moeEpParallelInfo;
    atb_speed::common::ParallelInfo mlpTpParallelInfo;
    atb_speed::common::ParallelInfo moeEpInterNodeParallelInfo;
    atb_speed::common::ParallelInfo moeEpIntraNodeParallelInfo;
    
    bool enableLcocAll2All = false;
    std::string lcclMoeEpDomain = "";
    HcclComm lcclMoeEpHcclComm = nullptr;

    bool mixSharedRouting = false;
    bool enableEPWB = false;
};

atb::Status CreateDynamicEpMoEOperation(const DynamicEpMoEParam &param, atb::Operation **operation);
}
} // namespace atb_speed
#endif