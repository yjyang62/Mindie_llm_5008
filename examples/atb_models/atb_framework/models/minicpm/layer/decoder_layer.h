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

#ifndef ATB_SPEED_MODELS_MINICPM_DECODER_LAYER_H
#define ATB_SPEED_MODELS_MINICPM_DECODER_LAYER_H

#include "nlohmann/json.hpp"

#include "atb/atb_infer.h"
#include "atb_speed/log.h"

#include "operations/fusion/linear/linear_parallel.h"
#include "models/base/param/layer_param.h"
#include "models/base/layer/decoder_layer.h"

namespace atb_speed {
namespace minicpm {

class MiniCPMLayerParam : public atb_speed::base::LayerParam {
public:
    void PrintParam() override;

    float numHiddenLayers = 52;
    float scaleDepth = 1.4 ;
};

class MiniCPMDecoderLayer : public atb_speed::base::DecoderLayer<atb::infer::RmsNormParam> {
public:
    explicit MiniCPMDecoderLayer(const MiniCPMLayerParam &param);
    ~MiniCPMDecoderLayer() override {};
    uint32_t GetLayerParamIndex(const std::string &param);
protected:
    atb::Status AddOperationToGraph() override;

    MiniCPMLayerParam param;
};


}  // namespace minicpm
}  // namespace atb_speed
#endif