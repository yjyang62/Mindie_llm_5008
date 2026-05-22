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

#ifndef ATB_SPEED_MODELS_QWEN_DECODER_LAYER_H
#define ATB_SPEED_MODELS_QWEN_DECODER_LAYER_H

#include "models/base/layer/decoder_layer.h"
#include "models/base/param/layer_param.h"

namespace atb_speed {
namespace qwen {

class QwenLayerParam : public atb_speed::base::LayerParam {
public:
    void PrintParam() override;

    bool enableLogN = false;
    bool isEmbedding = false;
    bool enableQScale = false;
};

class QwenDecoderLayer : public atb_speed::base::DecoderLayer<atb::infer::RmsNormParam> {
public:
    explicit QwenDecoderLayer(const QwenLayerParam &param);
    ~QwenDecoderLayer() override{};

protected:
    void ConstructInTensorMap() override;
    void SetFusionAttentionParam(
        atb_speed::common::FusionAttentionParam<atb::infer::RmsNormParam> &fusionAttentionParam) override;
    std::map<unsigned int, std::vector<std::string>> GetAttentionIntensor() override;
    QwenLayerParam param;
};

} // namespace qwen
} // namespace atb_speed
#endif
