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

#ifndef ATB_SPEED_MODELS_BAICHUAN2_13B_PA_QUANT_LAYER_H
#define ATB_SPEED_MODELS_BAICHUAN2_13B_PA_QUANT_LAYER_H

#include <atb/atb_infer.h>
#include <nlohmann/json.hpp>

#include <map>
#include <string>

#include "models/base/layer/decoder_layer.h"
#include "models/base/param/layer_param.h"

namespace atb_speed {
namespace baichuan2_13b {

class BaichuanLayerParam : public atb_speed::base::LayerParam {
public:
    void PrintParam() override;

    // 是否开启alibi mask free
    bool enableAlibiMaskFree = false;
};

class PagedAttentionQuantLayer : public atb_speed::base::DecoderLayer<atb::infer::RmsNormParam> {
public:
    explicit PagedAttentionQuantLayer(const BaichuanLayerParam &param);

protected:
    void ConstructInTensorMap() override;
    std::map<unsigned int, std::vector<std::string>> GetAttentionIntensor() override;
    void SetFusionAttentionNormParam(
        atb_speed::common::FusionAttentionParam<atb::infer::RmsNormParam> &fusionAttentionParam) override;
    void SetFusionAttentionATBSelfAttentionParam(
        atb_speed::common::FusionAttentionParam<atb::infer::RmsNormParam> &fusionAttentionParam) override;
    void SetMlpNormParam(atb_speed::common::MlpParam<atb::infer::RmsNormParam> &mlpParam) override;

    BaichuanLayerParam param;
};

} // namespace baichuan2_13b
} // namespace atb_speed
#endif
