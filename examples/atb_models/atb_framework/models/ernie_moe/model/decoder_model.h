/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * MindIE is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

#ifndef ATB_SPEED_MODELS_ERNIE_MOE_DECODER_MODEL_H
#define ATB_SPEED_MODELS_ERNIE_MOE_DECODER_MODEL_H

#include <atb_speed/utils/model_factory.h>
#include "models/ernie_moe/layer/decoder_layer.h"
#include "models/moe/model/decoder_model.h"

namespace atb_speed {
namespace ernie_moe {

class DecoderModel : public atb_speed::moe::MoeDecoderModel {
public:
    explicit DecoderModel(const std::string &param);

protected:
    void SetLayerNodeDefaultInput(atb_speed::Model::Node &layerNode, uint32_t layerId, uint32_t &inTensorId) override;
    uint32_t CalcWeightTensorSize() override;
    atb::Status CreateLayerOperation(atb::Operation **op, uint32_t layerId) override;

    atb_speed::moe::MoeModelParam param;
    const uint32_t weightCountDenseLayer = 68;
    uint32_t weightCountMoeLayer = 50;
};

REGISTER_MODEL(ernie_moe, DecoderModel);

}  // namespace ernie_moe
}  // namespace atb_speed
#endif
