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

#ifndef ATB_SPEED_MODELS_QWEN_MOE_DECODER_LAYER_H
#define ATB_SPEED_MODELS_QWEN_MOE_DECODER_LAYER_H

#include <vector>
#include "nlohmann/json.hpp"

#include "atb/atb_infer.h"
#include "atb_speed/base/hosttensor_binder.h"
#include "atb_speed/log.h"
#include "models/moe/layer/decoder_layer.h"

namespace atb_speed {
namespace qwen {
class MoeDecoderLayerParam : public atb_speed::moe::MoeLayerParam {
public:
    bool isPack = true;
    int quantType = 0;
    int maskStartIdx = 0;
    int layerId = 0;
    int rank = 0;
    int worldSize = 1;
    bool hasSharedExpert = true;
    bool hasSharedExpertGate = true;
    bool hasMoe = true;
    int attnDeviceNum = 1;
    int ffnDeviceNum = 1;
    bool attnAllreduce = false;
    bool attnReduceScatter = false;
    bool attnAllGather = false;
    bool ffnAllreduce = false;
    bool ffnReduceScatter = false;
    bool ffnAllGather = false;
    bool hasAttnComm = false;
    bool hasFfnComm = false;
    bool isAntiOutlier = false;
    bool isDynamicEp = false;
    bool isLastLayer = false;
    bool enableAllToAllMC2 = false;
    bool enableExpertCumSumOutput = false;
    bool isDenseLayer = false;
    bool enableLoadBalance = false;
    std::string backend = "hccl";
    std::string rankTableFile = "";
    std::vector<int> seqLen;
    std::vector<int> tokenOffset;
    std::vector<int> attnLinearQuantType = {};
    std::vector<int> attnLinearTransposeType = {};
    HcclComm dispatchAndCombineHcclComm;
    std::string dispatchAndCombinecommDomain = "";
    bool enableEPWB = false;
    uint32_t numOfRedundantExpert = 0;
};

void SetFusionAttentionAclNNIncreAttentionParam(
    const MoeDecoderLayerParam &param,
    atb_speed::common::FusionAttentionParam<atb::infer::RmsNormParam> &fusionAttentionParam);

atb::Status MoeDecoderLayer(MoeDecoderLayerParam &param, atb::Operation **operation);

class MoeDecoderLayer : public HostTensorBinder {
public:
    MoeDecoderLayer();
    ~MoeDecoderLayer() override;

private:
    std::vector<int> tokenOffset_;
    std::vector<int> seqLen_;
    int32_t layerId_ = 0;
};
}  // namespace qwen
}  // namespace atb_speed
#endif