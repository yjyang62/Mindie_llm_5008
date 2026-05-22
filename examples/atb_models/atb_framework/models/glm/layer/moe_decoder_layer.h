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

#ifndef ATB_SPEED_MODELS_GLM_DECODER_LAYER_H
#define ATB_SPEED_MODELS_GLM_DECODER_LAYER_H

#include "atb/atb_infer.h"
#include "operations/fusion/attention/fusion_attention.h"
#include "operations/fusion/moe/sparse_moe.h"
#include "operations/fusion/moe/moe_shared_expert.h"
#include "models/base/param/layer_param.h"
#include "models/moe/layer/decoder_layer.h"

namespace atb_speed {
namespace glm {

template <typename NormType>
class MoeDecoderLayer : public atb_speed::moe::MoeDecoderLayer<NormType> {
public:
    explicit MoeDecoderLayer(const atb_speed::moe::MoeLayerParam &param);
    ~MoeDecoderLayer() override {};

protected:
    void SetFusionAttentionParam(
        atb_speed::common::FusionAttentionParam<NormType> &fusionAttentionParam) override;
};

}  // namespace glm
}  // namespace atb_speed
#endif  // ATB_SPEED_MODELS_GLM_DECODER_LAYER_H