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

#ifndef ATB_SPEED_MODELS_GLM41V_DECODER_LAYER_H
#define ATB_SPEED_MODELS_GLM41V_DECODER_LAYER_H

#include "atb/atb_infer.h"
#include "models/base/param/layer_param.h"
#include "models/base/layer/decoder_layer.h"

namespace atb_speed {
namespace glm41v {


class DecoderLayer : public atb_speed::base::DecoderLayer<atb::infer::RmsNormParam> {
public:
    explicit DecoderLayer(const atb_speed::base::LayerParam &param);
    ~DecoderLayer() override {};
protected:
    void ConstructInTensorMap() override;
    atb::Status AddOperationToGraph() override;
    void SetFusionAttentionParam(
        atb_speed::common::FusionAttentionParam<atb::infer::RmsNormParam> &fusionAttentionParam) override;
    atb::Status AddPostSelfAttentionRMSNorm();
    atb::Status AddPostMlpRMSNorm();
};


}  // namespace glm41v
}  // namespace atb_speed
#endif