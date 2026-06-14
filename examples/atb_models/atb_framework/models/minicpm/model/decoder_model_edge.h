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

#ifndef ATB_SPEED_MODELS_MINICPM_DECODER_MODEL_310B_H
#define ATB_SPEED_MODELS_MINICPM_DECODER_MODEL_310B_H

#include <vector>
#include "atb_speed/base/model.h"
#include "atb_speed/utils/model_factory.h"
#include "models/minicpm/layer/decoder_layer_edge.h"


namespace atb_speed {
namespace minicpm {

class DecoderModelEdge : public Model {
public:
    struct Param {
        float rmsNormEps = 0;
        int numAttentionHeadsPerRank = 0;
        int hiddenSize = 0;
        int numKeyValueHeadsPerRank = 0;
        int scaleEmb = 0;
        float scaleDepth = 1.4;
        int dimModelBase = 0;
        int numHiddenLayers = 40;
        int numAttentionHeads = 8;
        int numKeyValueHeads = 5;
        int vocabSize = 0;
        int seqLength = 1;
        bool isGQA = false;
        bool isPrefill = false;
        bool isQuant = false;
        std::vector<std::vector<int>> packQuantType = {};
        std::vector<std::vector<int>> linearQuantType = {};
        std::vector<std::vector<int>> linearTransposeType = {};
        void FromString(const std::string &param);
        void ParseBasicParams(const nlohmann::json &paramJson);
        void CheckParam() const;
    };

    explicit DecoderModelEdge(const std::string &param);
    ~DecoderModelEdge() override;
    uint32_t GetInputNum() const override;
    uint32_t GetOutputNum() const override;
    atb::Status InferShape(const std::vector<atb::TensorDesc> &inTensorDescs,
                           std::vector<atb::TensorDesc> &outTensorDescs) override;

private:
    int64_t BuildGraph() override;
    int64_t AddWordEmbedding();
    int64_t AddMuls();
    int64_t AddPositionalEmbedding();
    int64_t AddLayer();
    int64_t AddFinalNorm();
    int64_t AddLmMuls();
    int64_t AddLmhead();
    Param param_;
    int32_t layerId_ = 0;
    uint64_t weightCountPerLayer = 6;
    uint32_t inTensorCount_ = 0;
    uint32_t internalTensorCount_ = 0;
    void ConstructInTensorMap();
    void ConstructInternalTensorMap();
    std::map<std::string, uint32_t> inTensorMap_;
    std::map<std::string, uint32_t> internalTensorMap_;
    int64_t QuantLmHeadInput();
    int64_t SetLayerParam(int layerId, atb_speed::minicpm::DecoderLayerParam &layerParam);
    atb::Status ParseParam(const std::string &param) override;
};

REGISTER_MODEL(minicpm, DecoderModelEdge);

}  // namespace minicpm
}  // namespace atb_speed
#endif
