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

#ifndef ATB_SPEED_MODELS_HUNYUAN_DECODER_LAYER_H
#define ATB_SPEED_MODELS_HUNYUAN_DECODER_LAYER_H

#include <vector>
#include <nlohmann/json.hpp>

#include "atb/atb_infer.h"
#include "atb_speed/log.h"
#include "atb_speed/utils/operation_util.h"
#include "models/moe/layer/decoder_layer.h"

namespace atb_speed {
namespace hunyuan {

constexpr int WEIGHT_COUNT_PER_LAYER = 64;

struct DecoderLayerParam : public atb_speed::moe::MoeLayerParam {
    void PrintParam() override;
    // cla
    bool isCrossLayer = false;
    float softmaxScale = 0;
    // moe
    bool hasSharedExpert = true;
    bool hasSharedExpertGate = false;
    int numOfSharedExperts = 1;
    int layerId = 0;
    // quant
    std::vector<int> attnLinearQuantType = {};
    std::vector<int> attnLinearTransposeType = {};
    /// moe router experts pack quantize type
    int moePackQuantType = atb_speed::common::PackQuantType::PACK_QUANT_UNDEFINED;
};

class DecoderLayer : public atb_speed::moe::MoeDecoderLayer<atb::infer::RmsNormParam> {
    using NormType = atb::infer::RmsNormParam;
    using BaseMoeLayer = atb_speed::moe::MoeDecoderLayer<atb::infer::RmsNormParam>;
public:
    explicit DecoderLayer(const DecoderLayerParam &param);
    ~DecoderLayer() override = default;

    atb::Status BuildGraph(atb::Operation **operation) final;
private:
    void ConstructInTensorMap() final;
    void ConstructInternalTensorMap() final;
    void ConstructOutTensorMap();
    atb::Status AddSharedExpert();
    atb::Status AddExpertAdd() override;
    atb::Status AddMoeAllReduce() final;
    atb::Status AddCrossLayerAttention();
    void SetSparseMoeParam(atb_speed::common::SparseMoeParam &sparseMoeParam) override;

private:
    DecoderLayerParam param;
};

}  // namespace hunyuan
}  // namespace atb_speed
#endif
