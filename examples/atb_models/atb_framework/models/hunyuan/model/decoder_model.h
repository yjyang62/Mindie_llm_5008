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

#ifndef ATB_SPEED_MODELS_HUNYUAN_DECODER_MODEL_H
#define ATB_SPEED_MODELS_HUNYUAN_DECODER_MODEL_H

#include "atb_speed/base/model.h"
#include "atb_speed/utils/model_factory.h"
#include "models/hunyuan/layer/decoder_layer.h"
#include "models/moe/model/decoder_model.h"

namespace atb_speed {
namespace hunyuan {
class HunyuanModelParam : public atb_speed::moe::MoeModelParam {
public:
    void PrintParam() override;

    int claShareFactor = 2;
    float softmaxScale = 0.0F;
    /// moe router experts pack quantize type
    int moePackQuantType = atb_speed::common::PackQuantType::PACK_QUANT_UNDEFINED;

    std::vector<std::vector<int>> attnLinearQuantType = {};
    std::vector<std::vector<int>> attnLinearTransposeType = {};

protected:
    void ParseParam(const nlohmann::json &paramJson) override;
};

class DecoderModel : public atb_speed::moe::MoeDecoderModel {
public:
    explicit DecoderModel(const std::string &param);

protected:
    atb::Status CreateLayerOperation(atb::Operation **op, uint32_t layerId) override;
    void SetLayerParam(DecoderLayerParam &layerParam, uint32_t layerId);
    void ConstructInternalTensorMap() override;
    void SetLayerNodeInput(atb_speed::Model::Node &layerNode, uint32_t layerId) override;
    void SetLayerNodeOutput(atb_speed::Model::Node &layerNode);
    atb::Status AddLayer() override;

    HunyuanModelParam param;
    
private:
    bool isCrossLayer_{false};
};


REGISTER_MODEL(hunyuan, DecoderModel);

}  // namespace hunyuan
}  // namespace atb_speed
#endif
