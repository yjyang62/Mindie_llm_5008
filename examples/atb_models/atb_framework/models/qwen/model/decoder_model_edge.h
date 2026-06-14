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

#include <vector>
#include "atb_speed/base/model.h"
#include "models/qwen/layer/decoder_layer_edge.h"
#include "atb_speed/utils/model_factory.h"

namespace atb_speed {
namespace qwen {
class DecoderModelEdge : public Model {
public:
    struct Param {
        // isFA为true则使用Flash Attention; 反之，则使用Paged Attention
        bool isFA = false;
        // isPrefill为true时为全量阶段，encoder的isPrefill参数应为true; isPrefill为false时为增量阶段，decoder的isPrefill参数应为false
        bool isPrefill = false;
        // isBF16为true时采用BF16精度; 反之，则采用FP16精度
        bool isBF16 = false;
        // withEmbedding为true时，模型包含word embedding层; 反之输入为hidden states; 该选项用于多模态模型适配
        bool withEmbedding = true;
        // isEmbeddingParallel为true时，embedding的权重在hiddenSize维度进行切分; 反之，则不对权重进行切分; 测试表明embedding切分并不会带来性能提升
        bool isEmbeddingParallel = false;
        // isLmHeadParallel为true时，LmHead的权重在vacobSize维度进行切分; 反之，则不对权重进行切分
        bool isLmHeadParallel = true;
        // 0 - No quant; 1- Quant in RmsNorm，dequant in Linear; 2 - Both quant and dequant in Linear
        int lmHeadTransposeType = -1;
        // MLP是否使用SwiGLU，若为true时，则使用；反之，使用swish
        bool supportSwiGLU = false;
        // 是否支持通信计算掩盖
        bool supportLcoc = false;
        // 是否并行解码
        bool supportSpeculate = false;
        // 是否使用Prefix cache
        bool supportPrefixCache = false;
        // 是否开启split fuse功能
        bool enableSplitFuse = false;
        // 是否支持lora
        bool supportLora = false;
        bool loraEnableGMM = false;
        // 是否使用kv cache int8 量化
        bool kvQuant = false;
        // 是否使用FA3量化
        bool enableFA3 = false;
        // 是否在PA算子执行前，提前对q缩放
        bool enableQScale = false;
        bool enableLogN = false;
        bool isLongSeq = false;
        bool isYarn = false;
        bool enableAddNorm = false;
        bool isQuant = false;
        float mscale = 1.0;
        float rmsNormEps = 0;
        uint32_t quantGroupSize = 64;
        uint32_t numAttentionHeadsPerRank = 0;
        uint32_t hiddenSizePerAttentionHead = 0;
        uint32_t numHiddenLayers = 0;
        uint32_t numKeyValueHeadsPerRank = 0;
        bool isEdgeHardware = true;
        uint32_t vocabSize = 151936;
        uint32_t hiddenSize = 0;
        bool useQKNorm = false;

        int rank = 0;
        int worldSize = 1;
        std::string backend = "hccl";
        std::vector<std::vector<int>> packQuantType = {};
        std::vector<std::vector<int>> linearQuantType = {};
        std::vector<std::vector<int>> linearTransposeType = {};
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

private:
    int64_t BuildGraph() override;
    int64_t AddWordEmbedding();
    int64_t AddPositionalEmbedding();
    int64_t SetLayerNodeDefaultInput(atb_speed::Model::Node &layerNode, uint32_t layerId, uint32_t &inTensorId);
    int64_t SetLayerNodeoutput(atb_speed::Model::Node &layerNode, uint32_t layerId);
    int64_t AddLayer();
    int64_t AddFinalNorm();
    int64_t AddLmhead();
    void SetLayerParam(DecoderLayerParam &layerParam, uint32_t layerId);
    atb::Status ParseParam(const std::string &param) override;
    void ConstructInTensorMap();
    void ConstructInternalTensorMap();
    void ConstructOutTensorMap();
    Param param_;
    std::map<std::string, uint32_t> inTensorMap_;
    std::map<std::string, uint32_t> internalTensorMap_;
    std::map<std::string, uint32_t> outTensorMap_;
    std::vector<int> tokenOffset_;
    std::vector<int> seqLen_;
    std::vector<int> qLen_;
    uint32_t layerId_ = 0;
    uint32_t inTensorCount_ = 0;
    uint32_t internalTensorCount_ = 0;
    uint32_t outTensorCount_ = 0;
    uint32_t weightCountPerLayer_ = 50;
};

REGISTER_MODEL(qwen, DecoderModelEdge);

}  // namespace qwen
}  // namespace atb_speed
#endif
