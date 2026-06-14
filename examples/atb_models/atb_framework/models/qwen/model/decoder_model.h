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

#ifndef ATB_SPEED_MODELS_QWEN_DECODER_MODEL_H
#define ATB_SPEED_MODELS_QWEN_DECODER_MODEL_H

#include "models/base/model/decoder_model.h"
#include "models/qwen/layer/decoder_layer.h"
#include "atb_speed/utils/model_factory.h"

namespace atb_speed {
namespace qwen {

class QwenModelParam : public atb_speed::base::ModelParam {
public:
    void PrintParam() override;

    // withEmbedding为true时，模型包含word embedding层; 反之输入为hidden states; 该选项用于多模态模型适配
    bool withEmbedding = true;

    // 是否使用kv cache int8 量化
    bool enableLogN = false;
    bool isLongSeq = false;
    bool isYarn = false;
    float mscale = 1.0;
    bool enableQScale = false;

protected:
    void ParseParam(const nlohmann::json &paramJson) override;
};

class QwenDecoderModel : public atb_speed::base::DecoderModel {
public:
    explicit QwenDecoderModel(const std::string &param);

protected:
    void ConstructInTensorMap() override;

    void ConstructInternalTensorMap() override;

    void ConstructOutTensorMap() override;

    atb::Status AddPositionalEmbedding() override;

    void SetLayerParam(QwenLayerParam &layerParam, uint32_t layerId);

    atb::Status BindParamHostTensor(uint32_t nodeId) override;

    void SetLayerNodeInput(atb_speed::Model::Node &layerNode, uint32_t layerId) override;

    void SetLmHeadParam(atb_speed::common::LmHeadParam &lmHeadParam) override;

    atb::Status CreateLayerOperation(atb::Operation **op, uint32_t layerId) override;

    atb::Status AddNodesBeforeLayer() override;

    atb::Status AddNodesAfterLayer() override;

    bool ReuseEmbedTable();

    bool OutputEmbedTable();

    atb::Tensor *GetSineEmbedTable();

    atb::Tensor *GetCosineEmbedTable();

    QwenModelParam param;

private:
    atb::Status AddMuls();
    atb::Status AddDynamicNTK();
};

REGISTER_MODEL(qwen, QwenDecoderModel);

} // namespace qwen
} // namespace atb_speed
#endif
