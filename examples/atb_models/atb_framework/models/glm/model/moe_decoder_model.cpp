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
#include "models/moe/layer/decoder_layer.h"
#include "models/glm/layer/moe_decoder_layer.h"
#include "models/glm/model/moe_decoder_model.h"

namespace atb_speed {
namespace glm {

constexpr int GLM_WEIGHT_COUNT_PER_LAYER = 68;

MoeDecoderModel::MoeDecoderModel(const std::string &param) : atb_speed::moe::MoeDecoderModel(param)
{
    this->weightCountPerLayer = GLM_WEIGHT_COUNT_PER_LAYER;
};

atb::Status MoeDecoderModel::CreateLayerOperation(atb::Operation **op, uint32_t layerId)
{
    atb_speed::moe::MoeLayerParam layerParam;
    this->SetLayerParam(layerParam, layerId);
    if (this->param.normType == atb_speed::base::RMS_NORM) {
        MoeDecoderLayer<atb::infer::RmsNormParam> decoderLayer(layerParam);
        CHECK_OPERATION_STATUS_RETURN(decoderLayer.BuildGraph(op));
    } else {
        MoeDecoderLayer<atb::infer::LayerNormParam> decoderLayer(layerParam);
        CHECK_OPERATION_STATUS_RETURN(decoderLayer.BuildGraph(op));
    }
    return atb::NO_ERROR;
}

} // namespace glm
} // namespace atb_speed