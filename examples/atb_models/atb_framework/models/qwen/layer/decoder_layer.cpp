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
#include "models/qwen/layer/decoder_layer.h"

namespace atb_speed {
namespace qwen {

void QwenLayerParam::PrintParam()
{
    LayerParam::PrintParam();
    std::stringstream ss;
    ss << " Layer Param: " << "enableLogN: " << this->enableLogN << ", isEmbedding: " << this->isEmbedding
       << ", enableQScale: " << this->enableQScale;
    ATB_SPEED_LOG_DEBUG(ss.str());
}

QwenDecoderLayer::QwenDecoderLayer(const QwenLayerParam &param) : base::DecoderLayer<atb::infer::RmsNormParam>(param)
{
    this->param = param;
    this->param.PrintParam();
    this->inTensorCandidates["logn_enable"] = {"kv_cache_idx"};
};

void QwenDecoderLayer::ConstructInTensorMap()
{
    DecoderLayer<atb::infer::RmsNormParam>::ConstructInTensorMap();

    // 添加logn特性的Tensor
    if (this->param.enableLogN) {
        atb_speed::common::AddTensorToList(this->inTensorCandidates, "logn_enable", this->inTensorList);
    }
}

void QwenDecoderLayer::SetFusionAttentionParam(
    atb_speed::common::FusionAttentionParam<atb::infer::RmsNormParam> &fusionAttentionParam)
{
    DecoderLayer<atb::infer::RmsNormParam>::SetFusionAttentionParam(fusionAttentionParam);
    if (this->param.enableLogN) {
        fusionAttentionParam.pageAttentionParam.scaleType = atb::infer::PagedAttentionParam::SCALE_TYPE_LOGN;
    }
    if (this->param.isEmbedding) {
        fusionAttentionParam.selfAttentionParam.maskType =
            atb::infer::SelfAttentionParam::MaskType::MASK_TYPE_UNDEFINED;
    } else {
        fusionAttentionParam.selfAttentionParam.maskType = atb::infer::SelfAttentionParam::MaskType::MASK_TYPE_NORM;
    }
    fusionAttentionParam.enableQScale = !param.isFA && param.enableQScale;
    fusionAttentionParam.pageAttentionParam.qkScale =
        param.enableQScale ? 1.0 : fusionAttentionParam.pageAttentionParam.qkScale;
    fusionAttentionParam.selfAttentionParam.qkScale =
        param.enableQScale ? 1.0 : fusionAttentionParam.selfAttentionParam.qkScale;
}

std::map<unsigned int, std::vector<std::string>> QwenDecoderLayer::GetAttentionIntensor()
{
    std::map<unsigned int, std::vector<std::string>> attnInTensor =
        DecoderLayer<atb::infer::RmsNormParam>::GetAttentionIntensor();

    if (this->param.enableLogN) {
        attnInTensor[common::AttnInTensorCategory::ATTN_LOG_N_SCALE] = {"kv_cache_idx"};
    }
    return attnInTensor;
}

} // namespace qwen
} // namespace atb_speed