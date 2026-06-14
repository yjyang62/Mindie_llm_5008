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

#ifndef ATB_SPEED_MODELS_QWEN_MOE_DECODER_MODEL_H
#define ATB_SPEED_MODELS_QWEN_MOE_DECODER_MODEL_H

#include <vector>
#include "atb_speed/base/model.h"
#include "atb_speed/utils/model_factory.h"
#include "models/qwen/layer/moe_decoder_layer.h"
#include "models/moe/model/decoder_model.h"

namespace atb_speed {
namespace qwen {
class QwenMoeParam : public atb_speed::moe::MoeModelParam {
public:
    int numAttentionHeadsPerRank = 0;
    int hiddenSizePerAttentionHead = 0;
    bool hasSharedExpert = true;
    bool enableAllToAllMC2 = false;
    bool enableEPWB = false;
    uint32_t numOfRedundantExpert = 0;
    bool enableExpertCumSumOutput = false;
    bool enableLoadBalance = false;
    std::vector<int> tokenOffset = {};
    std::vector<int> seqLen = {};
    std::vector<bool> isDenseLayer = {};
    std::vector<std::vector<int>> attnLinearQuantType = {};
    std::vector<std::vector<int>> attnLinearTransposeType = {};

    static HcclComm dispatchAndCombineHcclComm;
    static std::string dispatchAndCombinecommDomain;
    void SetHcclComm() const;
    void AddParamJson(const std::string &param);
    void AddLogInfo();
    virtual void FromString(const std::string &param);
    void ParseBasicParams(const nlohmann::json &paramJson);
};

class MoeDecoderModel : public atb_speed::moe::MoeDecoderModel {
public:
    explicit MoeDecoderModel(const std::string &param);
    atb::Status InferShape(const std::vector<atb::TensorDesc> &inTensorDescs,
                           std::vector<atb::TensorDesc> &outTensorDescs) override;

private:
    // int64_t BuildGraph() override;
    uint32_t CalcWeightTensorSize() override;
    void ConstructInTensorMap() override;
    void ConstructInternalTensorMap() override;
    void ConstructOutTensorMap() override;
    atb::Status AddWordEmbedding() override;
    atb::Status AddPositionalEmbedding() override;
    atb::Status AddNodesBeforeLayer() override;
    atb::Status AddNodesAfterLayer() override;
    atb::Status AddSingleLayer(uint32_t layerId) override;
    atb::Status AddFinalNorm() override;
    atb::Status AddLmhead() override;
    atb::Status SetLayerParam(atb_speed::qwen::MoeDecoderLayerParam &layerParam, const int layerId);
    atb::Status SetLayerMoeParam(atb_speed::qwen::MoeDecoderLayerParam &layerParam, const int layerId);
    atb::Status AddParallelHostWeight(atb_speed::Model::Node &layerNode, size_t &inTensorId);
    atb::Status AddLayerTensor(atb_speed::Model::Node &layerNode, size_t &inTensorId, const int &layerId);
    atb::Status ParseParam(const std::string &param) override;
    atb::Status BindParamHostTensor(uint32_t nodeId) override;
    QwenMoeParam param_;
    std::vector<int> tokenOffset_;
    std::vector<int> seqLen_;
    std::vector<int> qLen_;
};

REGISTER_MODEL(qwen, MoeDecoderModel);
}  // namespace qwen
}  // namespace atb_speed
#endif
