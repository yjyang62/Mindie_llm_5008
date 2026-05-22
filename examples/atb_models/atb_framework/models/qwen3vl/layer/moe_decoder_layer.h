/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
 * MindIE is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */
#ifndef ATB_SPEED_MODELS_QWEN3VL_MOE_DECODER_LAYER_H
#define ATB_SPEED_MODELS_QWEN3VL_MOE_DECODER_LAYER_H

#include "atb/atb_infer.h"
#include "models/moe/layer/decoder_layer.h"

namespace atb_speed {
namespace qwen3vl {
class MoeDecoderLayer : public atb_speed::moe::MoeDecoderLayer<atb::infer::RmsNormParam> {
public:
    explicit MoeDecoderLayer(const atb_speed::moe::MoeLayerParam &param);
    ~MoeDecoderLayer() override{};
    atb::Status BuildGraph(atb::Operation **operation) override;
protected:
    void ConstructInTensorMap() override;
    void SetFusionAttentionParam(
        atb_speed::common::FusionAttentionParam<atb::infer::RmsNormParam> &fusionAttentionParam) override;
    atb::Status AddDeepStack(int layerId);
};

}  // namespace qwen3vl
}  // namespace atb_speed
#endif