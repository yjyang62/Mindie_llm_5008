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

#ifndef ATB_SPEED_MODELS_QWEN3VL_DECODER_MODEL_H
#define ATB_SPEED_MODELS_QWEN3VL_DECODER_MODEL_H

#include "models/base/model/decoder_model.h"
#include "models/qwen3vl/layer/decoder_layer.h"
#include "atb_speed/utils/model_factory.h"
#include "atb_speed/utils/check_util.h"

namespace atb_speed {
namespace qwen3vl {

class DecoderModel : public atb_speed::base::DecoderModel {
public:
    explicit DecoderModel(const std::string &param);
protected:
    void ConstructInTensorMap() override;
    atb::Status CreateLayerOperation(atb::Operation **op, uint32_t layerId) override;
    void SetLayerParam(atb_speed::base::LayerParam &layerParam, uint32_t layerId) override;
    void SetLayerNodeDefaultInput(atb_speed::Model::Node &layerNode, uint32_t layerId, uint32_t &inTensorId) override;
    void SetLayerNodeOptionalInput(atb_speed::Model::Node &layerNode, uint32_t layerId, uint32_t &inTensorId) override;
    atb::Status AddNodesBeforeLayer() override;
    atb::Status AddSingleLayer(uint32_t layerId) override;
};

REGISTER_MODEL(qwen3vl, DecoderModel);

} // namespace qwen3vl
} // namespace atb_speed
#endif