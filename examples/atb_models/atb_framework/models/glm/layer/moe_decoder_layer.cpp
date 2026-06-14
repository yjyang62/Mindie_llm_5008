/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * MindIE is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */
#include "operations/fusion/linear/linear.h"
#include "operations/fusion/norm/norm_linear.h"
#include "models/moe/layer/decoder_layer.h"
#include "models/glm/layer/moe_decoder_layer.h"

namespace atb_speed {
namespace glm {

template <typename NormType>
MoeDecoderLayer<NormType>::MoeDecoderLayer(
    const atb_speed::moe::MoeLayerParam &param) : atb_speed::moe::MoeDecoderLayer<NormType>(param) {};

template <typename NormType>
void MoeDecoderLayer<NormType>::SetFusionAttentionParam(
    atb_speed::common::FusionAttentionParam<NormType> &fusionAttentionParam)
{
    atb_speed::moe::MoeDecoderLayer<NormType>::SetFusionAttentionParam(fusionAttentionParam);
    fusionAttentionParam.rotaryType = atb_speed::common::RotaryType::HALF_ROTARY;
    fusionAttentionParam.ropeParam.rotaryCoeff = 2; // 2: rotary coeff
}

template class MoeDecoderLayer<atb::infer::RmsNormParam>;
template class MoeDecoderLayer<atb::infer::LayerNormParam>;

} // namespace glm
} // namespace atb_speed