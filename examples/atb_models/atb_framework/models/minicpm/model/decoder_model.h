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

#ifndef ATB_SPEED_MODELS_MINICPM_DECODER_MODEL_H
#define ATB_SPEED_MODELS_MINICPM_DECODER_MODEL_H

#include "atb_speed/base/model.h"
#include "models/minicpm/layer/decoder_layer.h"
#include "atb_speed/utils/model_factory.h"
#include "models/base/param/model_param.h"
#include "models/base/model/decoder_model.h"

namespace atb_speed {
namespace minicpm {

class MiniCPMModelParam : public  atb_speed::base::ModelParam {
public:
    uint32_t hiddenSize = 0;
    float scaleEmb = 12.0;
    float scaleDepth = 1.4;
    int dimModelBase = 256 ;

    void PrintParam() override;
protected:
    void ParseParam(const nlohmann::json &paramJson) override;
};

class MiniCPMDecoderModel : public atb_speed::base::DecoderModel {
public:
    explicit MiniCPMDecoderModel(const std::string &param);
    ~MiniCPMDecoderModel() override;

private:
    atb::Status CreateLayerOperation(atb::Operation **op, uint32_t layerId) override;
    int64_t AddMuls();
    atb::Status AddOperationToGraph() override;
    int64_t AddLmMuls();
    void SetLayerParam(MiniCPMLayerParam &layerParam, uint32_t layerId);

    MiniCPMModelParam param;
};

REGISTER_MODEL(minicpm, MiniCPMDecoderModel);

}  // namespace minicpm
}  // namespace atb_speed
#endif
