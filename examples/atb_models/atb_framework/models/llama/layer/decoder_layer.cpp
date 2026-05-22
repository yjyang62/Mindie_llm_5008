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
#include "operations/fusion/linear/linear.h"
#include "operations/fusion/norm/norm_linear.h"
#include "models/llama/layer/decoder_layer.h"

namespace atb_speed {
namespace llama {

void LlamaLayerParam::PrintParam()
{
    LayerParam::PrintParam();
    std::stringstream ss;
    ss << "Llama Layer Param:splitWithStride: " << this->splitWithStride;
    ATB_SPEED_LOG_INFO(ss.str());
}

LlamaDecoderLayer::LlamaDecoderLayer(
    const LlamaLayerParam &param) : atb_speed::base::DecoderLayer<atb::infer::RmsNormParam>(param)
{
    this->param = param;
    this->param.CheckParam();
};

void LlamaDecoderLayer::SetFusionAttentionLinearParam(
    atb_speed::common::FusionAttentionParam<atb::infer::RmsNormParam> &fusionAttentionParam)
{
    DecoderLayer<atb::infer::RmsNormParam>::SetFusionAttentionLinearParam(fusionAttentionParam);
    fusionAttentionParam.splitWithStride = this->param.splitWithStride;
}

} // namespace llama
} // namespace atb_speed