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

#include <vector>
#include "atb_speed/base/model.h"
#include "models/llama/layer/decoder_layer_edge.h"
#include "atb_speed/utils/model_factory.h"

namespace atb_speed {
namespace llama {

class DecoderModelEdge : public Model {
public:
    struct Param {
        // skipWordEmbedding为true会跳过Word Embedding阶段，直接使用入参中的IN_TENSOR_INPUT_EMBEDDING
        bool skipWordEmbedding = false;
        // isFA为true则使用Flash Attention; 反之，则使用Paged Attention
        bool isFA = true;
        // isPrefill为true时为全量阶段，encoder的isPrefill参数应为true; isPrefill为false时为增量阶段，decoder的isPrefill参数应为false
        bool isPrefill = false;
        // isBF16为true时采用BF16精度; 反之，则采用FP16精度
        bool isBF16 = false;
        // isEmbeddingParallel为true时，embedding的权重在hiddenSize维度进行切分; 反之，则不对权重进行切分; 测试表明embedding切分并不会带来性能提升
        bool isEmbeddingParallel = false;
        // isLmHeadParallel为true时，LmHead的权重在vacobSize维度进行切分; 反之，则不对权重进行切分
        bool isLmHeadParallel = true;
        int lmHeadTransposeType = -1;
        // MLP是否使用SwiGLU，若为true时，则使用；反之，使用swish
        bool supportSwiGLU = false;
        // 是否支持通信计算掩盖
        bool supportLcoc = false;
        // outputHiddenStates为true时仅输出hidden states，否则输出logits
        bool outputHiddenStates = false;
        bool enableAddNorm = false;
        bool isQuant = false;
        float rmsNormEps = 0;
        int vocabSize = 0;
        uint32_t quantGroupSize = 64;
        uint32_t numAttentionHeadsPerRank = 0;
        uint32_t hiddenSizePerAttentionHead = 0;
        uint32_t numHiddenLayers = 0;
        uint32_t numKeyValueHeadsPerRank = 0;
        uint32_t hiddenSize = 0;
        uint32_t seqLength = 1;
        int rank = 0;
        int worldSize = 1;
        int attnBackend = atb_speed::common::OpBackend::ATB;
        std::string backend = "hccl";
        std::string rankTableFile = "";
        std::string positionEmbeddingType = "ROPE";
        std::vector<std::vector<int>> packQuantType = {};
        std::vector<std::vector<int>> linearQuantType = {};
        std::vector<std::vector<int>> linearTransposeType = {};
        std::vector<bool> linearHasBias = {false, false, false, false};
        void AddParamJson(const std::string &param);
        void FromString(const std::string &param);
        void PrintParam();
        void ParseBasicParams(const nlohmann::json &paramJson);
    };

    explicit DecoderModelEdge(const std::string &param);
    ~DecoderModelEdge() override;
    uint32_t GetInputNum() const override;
    uint32_t GetOutputNum() const override;
    atb::Status InferShape(const std::vector<atb::TensorDesc> &inTensorDescs,
                           std::vector<atb::TensorDesc> &outTensorDescs) override;
    atb::Status PrefillInferShape(const std::vector<atb::TensorDesc> &inTensorDescs,
                           std::vector<atb::TensorDesc> &outTensorDescs);
    atb::Status DecodeInferShape(const std::vector<atb::TensorDesc> &inTensorDescs,
                           std::vector<atb::TensorDesc> &outTensorDescs);

private:
    int64_t BuildGraph() override;
    int64_t AddWordEmbedding();
    int64_t AddMuls();
    int64_t AddPositionalEmbedding();
    int64_t SetLayerNodeDefaultInput(atb_speed::Model::Node &layerNode, uint32_t layerId, uint32_t &inTensorId);
    int64_t SetLayerNodeRaInput(atb_speed::Model::Node &layerNode, uint32_t &inTensorId);
    int64_t SetLayerNodeOptionalInput(atb_speed::Model::Node &layerNode, uint32_t layerId, uint32_t &inTensorId);
    int64_t AddLayer();
    int64_t AddFinalNorm();
    int64_t AddLmhead();
    int64_t AddSlice(atb::SVector<int64_t> offsets, atb::SVector<int64_t> size,\
        const std::string &inTensorName, const std::string &internalTersorMap);
    void SetLayerParam(DecoderLayerParam &layerParam, uint32_t layerId);
    atb::Status ParseParam(const std::string &param) override;
    atb::Status BindParamHostTensor(uint32_t nodeId) override;
    void BuildNodeOutTensors(int nodeId, atb_speed::Model::Node &node, atb::SVector<atb::TensorDesc>& inTensorDescs);
    void BuildNodeVariantPack(int nodeId) override;
    void ConstructInTensorMap();
    void ConstructInternalTensorMap();
    Param param_;
    std::map<std::string, uint32_t> inTensorMap_;
    std::map<std::string, uint32_t> internalTensorMap_;
    std::vector<int> tokenOffset_;
    std::vector<int> seqLen_;
    std::vector<int> qLen_;
    std::vector<int> blockNumsList_;
    uint32_t layerId_ = 0;
    uint32_t inTensorCount_ = 0;
    uint32_t internalTensorCount_ = 0;
    uint32_t weightCountPerLayer_ = 50;
};

REGISTER_MODEL(llama, DecoderModelEdge);

}  // namespace llama
}  // namespace atb_speed
#endif
