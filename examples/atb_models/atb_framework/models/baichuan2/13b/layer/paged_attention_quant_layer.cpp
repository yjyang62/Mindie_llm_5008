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
#include "models/baichuan2/13b/layer/paged_attention_quant_layer.h"

namespace atb_speed {
namespace baichuan2_13b {

void BaichuanLayerParam::PrintParam()
{
    LayerParam::PrintParam();
    ATB_SPEED_LOG_DEBUG("BaichuanLayerParam: enableAlibiMaskFree: " << this->enableAlibiMaskFree);
}

PagedAttentionQuantLayer::PagedAttentionQuantLayer(
    const BaichuanLayerParam &param) : atb_speed::base::DecoderLayer<atb::infer::RmsNormParam>(param)
{
    this->param = param;
    this->param.CheckParam();
    this->param.PrintParam();
    this->inTensorCandidates["alibi_mask_compress"] = {"in_slopes"};
}

void PagedAttentionQuantLayer::ConstructInTensorMap()
{
    atb_speed::base::DecoderLayer<atb::infer::RmsNormParam>::ConstructInTensorMap();
    atb_speed::common::AddTensorToList(this->inTensorCandidates, "alibi_mask_compress", this->inTensorList);
}

std::map<unsigned int, std::vector<std::string>> PagedAttentionQuantLayer::GetAttentionIntensor()
{
    std::map<unsigned int, std::vector<std::string>> attnInTensor = \
        DecoderLayer<atb::infer::RmsNormParam>::GetAttentionIntensor();
    if (this->param.enableAlibiMaskFree) {
        attnInTensor[common::AttnInTensorCategory::ATTN_ALIBI_MASK_COMPRESS] = \
                    this->inTensorCandidates["alibi_mask_compress"];
    }
    return attnInTensor;
}

void PagedAttentionQuantLayer::SetFusionAttentionNormParam(
    atb_speed::common::FusionAttentionParam<atb::infer::RmsNormParam> &fusionAttentionParam)
{
    atb_speed::base::DecoderLayer<atb::infer::RmsNormParam>::SetFusionAttentionNormParam(fusionAttentionParam);
    fusionAttentionParam.normParamType.normParam.precisionMode = atb::infer::RmsNormParam::HIGH_PERFORMANCE_MODE;
}

void PagedAttentionQuantLayer::SetFusionAttentionATBSelfAttentionParam(
    atb_speed::common::FusionAttentionParam<atb::infer::RmsNormParam> &fusionAttentionParam)
{
    atb_speed::base::DecoderLayer<atb::infer::RmsNormParam>::SetFusionAttentionATBSelfAttentionParam(
        fusionAttentionParam);
    fusionAttentionParam.selfAttentionParam.kernelType = atb::infer::SelfAttentionParam::KERNELTYPE_HIGH_PRECISION;
    if (this->param.enableAlibiMaskFree) {
        fusionAttentionParam.selfAttentionParam.maskType = \
        atb::infer::SelfAttentionParam::MaskType::MASK_TYPE_ALIBI_COMPRESS_LEFT_ALIGN;
    } else {
        fusionAttentionParam.selfAttentionParam.maskType = atb::infer::SelfAttentionParam::MaskType::MASK_TYPE_ALIBI;
    }
}

void PagedAttentionQuantLayer::SetMlpNormParam(
    atb_speed::common::MlpParam<atb::infer::RmsNormParam> &mlpParam)
{
    atb_speed::base::DecoderLayer<atb::infer::RmsNormParam>::SetMlpNormParam(mlpParam);
    mlpParam.normParamType.normParam.precisionMode = atb::infer::RmsNormParam::HIGH_PERFORMANCE_MODE;
}

} // namespace baichuan2_13b
} // namespace atb_speed
