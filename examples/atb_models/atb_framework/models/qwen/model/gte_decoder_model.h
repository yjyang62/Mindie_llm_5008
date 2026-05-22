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

#ifndef ATB_SPEED_MODELS_QWEN_DECODER_MODEL_H
#define ATB_SPEED_MODELS_QWEN_DECODER_MODEL_H

#include <vector>
#include "atb_speed/base/model.h"
#include "models/qwen/layer/decoder_layer.h"
#include "atb_speed/utils/model_factory.h"
#include "models/base/param/model_param.h"
#include "models/base/model/decoder_model.h"

namespace atb_speed {
namespace qwen {
class GteDecoderModelParam : public atb_speed::base::ModelParam {
public:
    bool isEmbedding = false;
    bool withEmbedding = true;
    bool enableLogN = false;

    uint32_t quantGroupSize = 64;
    void PrintParam() override;

protected:
    void ParseParam(const nlohmann::json &paramJson) override;
};
    

class GteDecoderModel : public atb_speed::base::DecoderModel {
public:
    explicit GteDecoderModel(const std::string &param);
    ~GteDecoderModel() override;
protected:
    void ConstructInTensorMap() override;
    void ConstructInternalTensorMap() override;
    atb::Status InferShape(const std::vector<atb::TensorDesc> &inTensorDescs,
                           std::vector<atb::TensorDesc> &outTensorDescs) override;
    atb::Status AddFinalNorm() override;
    int64_t BuildGraph() override;
    void SetLayerParam(QwenLayerParam &layerParam, uint32_t layerId);
    void SetLayerNodeInput(atb_speed::Model::Node &layerNode, uint32_t layerId) override;
    atb::Status CreateLayerOperation(atb::Operation **op, uint32_t layerId) override;
    atb::Status AddOperationToGraph() override;
    GteDecoderModelParam param;
};

REGISTER_MODEL(qwen, GteDecoderModel);

} // namespace base
} // namespace atb_speed
#endif