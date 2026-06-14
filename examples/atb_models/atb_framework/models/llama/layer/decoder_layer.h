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

#ifndef ATB_SPEED_MODELS_LLAMA_DECODER_LAYER_H
#define ATB_SPEED_MODELS_LLAMA_DECODER_LAYER_H

#include "atb/atb_infer.h"

#include "models/base/param/layer_param.h"
#include "models/base/layer/decoder_layer.h"

namespace atb_speed {
namespace llama {

class LlamaLayerParam : public atb_speed::base::LayerParam {
public:
    LlamaLayerParam() {};
    ~LlamaLayerParam() override {};
    void PrintParam() override;

    bool splitWithStride = false;
};

class LlamaDecoderLayer : public atb_speed::base::DecoderLayer<atb::infer::RmsNormParam> {
public:
    explicit LlamaDecoderLayer(const LlamaLayerParam &param);
    ~LlamaDecoderLayer() override {};

protected:
    void SetFusionAttentionLinearParam(
        atb_speed::common::FusionAttentionParam<atb::infer::RmsNormParam> &fusionAttentionParam) override;

    LlamaLayerParam param;
};

}  // namespace llama
}  // namespace atb_speed
#endif
