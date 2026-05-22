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

#ifndef ATB_SPEED_MODELS_LLAMA_DECODER_MODEL_H
#define ATB_SPEED_MODELS_LLAMA_DECODER_MODEL_H

#include "atb_speed/base/model.h"
#include "models/base/param/model_param.h"
#include "atb_speed/utils/model_factory.h"
#include "models/llama/layer/decoder_layer.h"
#include "models/base/model/decoder_model.h"

namespace atb_speed {
namespace llama {

class LlamaModelParam : public atb_speed::base::ModelParam {
public:
    LlamaModelParam() {};
    ~LlamaModelParam() override {};
    void PrintParam() override;

    // 是否需要在QKV切分之前进行reshape
    bool splitWithStride = false;
    // 输入是否为长序列
    bool isLongSeq = false;

protected:
    void ParseParam(const nlohmann::json &paramJson) override;
};

class LlamaDecoderModel : public atb_speed::base::DecoderModel {
public:
    explicit LlamaDecoderModel(const std::string &param);

protected:
    void ConstructInTensorMap() override;
    void ConstructInternalTensorMap() override;
    atb::Status ParseParam(const std::string &paramString) override;
    atb::Status BindParamHostTensor(uint32_t nodeId) override;
    void SetLayerParam(LlamaLayerParam &layerParam, uint32_t layerId);
    atb::Status CreateLayerOperation(atb::Operation **op, uint32_t layerId) override;
    atb::Status AddPositionalEmbedding() override;
    atb::Status AddNodesBeforeLayer() override;
    atb::Status AddNodesAfterLayer() override;

    LlamaModelParam param;
    std::vector<int> blockNumsList_;

private:
    atb::Status AddDynamicNTK();
};

REGISTER_MODEL(llama, LlamaDecoderModel);

}  // namespace llama
}  // namespace atb_speed
#endif
