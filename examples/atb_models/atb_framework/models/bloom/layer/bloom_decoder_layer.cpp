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
#include "models/bloom/layer/bloom_decoder_layer.h"

namespace atb_speed {
namespace bloom {

BloomDecoderLayer::BloomDecoderLayer(
    const atb_speed::base::LayerParam &param) : atb_speed::base::DecoderLayer<atb::infer::LayerNormParam>(
        static_cast<atb_speed::base::LayerParam>(param))
{
    this->param = param;
    this->param.CheckParam();
};


void BloomDecoderLayer::SetFusionAttentionLinearParam(
    atb_speed::common::FusionAttentionParam<atb::infer::LayerNormParam> &fusionAttentionParam)
{
    DecoderLayer<atb::infer::LayerNormParam>::SetFusionAttentionLinearParam(fusionAttentionParam);
    fusionAttentionParam.qkvHasBias = true;
    fusionAttentionParam.splitWithStride = true;
}


void BloomDecoderLayer::SetMlpParam(atb_speed::common::MlpParam<atb::infer::LayerNormParam> &mlpParam)
{
    DecoderLayer<atb::infer::LayerNormParam>::SetMlpParam(mlpParam);
    mlpParam.normHasBias = true;
    mlpParam.mlpPackType = atb_speed::common::GetMlpPackType(this->param.packQuantType.at(1), true);
    mlpParam.activationParam.geluMode = atb::infer::ActivationParam::TANH_MODE;
    mlpParam.activationParam.activationType = atb::infer::ActivationType::ACTIVATION_GELU;
}

} // namespace bloom
} // namespace atb_speed