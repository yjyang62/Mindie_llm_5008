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
#ifndef ATB_SPEED_MODELS_QWEN3VL_MOE_DECODER_MODEL_H
#define ATB_SPEED_MODELS_QWEN3VL_MOE_DECODER_MODEL_H

#include "atb_speed/utils/model_factory.h"
#include "models/qwen3vl/layer/moe_decoder_layer.h"
#include "models/moe/model/decoder_model.h"

namespace atb_speed {
namespace qwen3vl {
class MoeDecoderModel : public atb_speed::moe::MoeDecoderModel {
public:
    explicit MoeDecoderModel(const std::string &param);

private:
    void ConstructInTensorMap() override;
    void SetLayerNodeInput(atb_speed::Model::Node &layerNode, uint32_t layerId);
    atb::Status CreateLayerOperation(atb::Operation **op, uint32_t layerId);
};

REGISTER_MODEL(qwen3vl, MoeDecoderModel);

}  // namespace qwen3vl
}  // namespace atb_speed
#endif