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

#ifndef ATB_SPEED_MODELS_BAICHUAN2_13B_PA_QUANT_MODEL_H
#define ATB_SPEED_MODELS_BAICHUAN2_13B_PA_QUANT_MODEL_H

#include "atb_speed/base/model.h"
#include "atb_speed/utils/model_factory.h"
#include "models/baichuan2/13b/layer/paged_attention_quant_layer.h"
#include "models/base/model/decoder_model.h"
#include "models/base/param/model_param.h"

namespace atb_speed {
namespace baichuan2_13b {

class BaichuanModelParam : public atb_speed::base::ModelParam {
public:
    void PrintParam() override;

    // 是否开启alibi mask free
    bool enableAlibiMaskFree = false;

protected:
    void ParseParam(const nlohmann::json &paramJson) override;
};

class PagedAttentionQuantModel : public atb_speed::base::DecoderModel {
public:
    explicit PagedAttentionQuantModel(const std::string &param);

protected:
    void ConstructInTensorMap() override;
    atb::Status CreateLayerOperation(atb::Operation **op, uint32_t layerId) override;
    void SetLayerParam(BaichuanLayerParam &layerParam, uint32_t layerId);
    void SetLayerNodeInput(atb_speed::Model::Node &layerNode, uint32_t layerId) override;
    void SetFinalNormParam(atb::infer::RmsNormParam &normParam) override;
    atb::Status BindParamHostTensor(uint32_t nodeId) override;

    BaichuanModelParam param;
};

} // namespace baichuan2_13b
} // namespace atb_speed
#endif
