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
#include <gtest/gtest.h>
#include <mockcpp/mockcpp.hpp>
#include <acl/acl.h>
#include <aclnn/acl_meta.h>
#include <atb/atb_infer.h>
#define private public
#include "operations/fusion/embedding/word_embedding.h"
#include "operations/fusion/embedding/positional_embedding.h"
#include "operations/fusion/lmhead/lmhead.h"
#include "models/llama/model/decoder_model_edge.h"

#define MOCKER_CPP(api, TT) (MOCKCPP_NS::mockAPI((#api), (reinterpret_cast<TT>(api))))

namespace atb_speed {

namespace llama {
    extern uint64_t g_headDim;
}

TEST(DecoderModelEdgeParamTest, FromString) {
    GlobalMockObject::verify();

    std::string param = R"({
        "skipWordEmbedding": true,
        "isFA": false,
        "isPrefill": true,
        "isBF16": false,
        "lmHeadTransposeType": 1,
        "enableSwiGLU": true,
        "attnBackend": 2,
        "normEps": 0.1,
        "numAttentionHeadsPerRank": 8,
        "hiddenSizePerAttentionHead": 64,
        "numHiddenLayers": 1,
        "numKeyValueHeadsPerRank": 8,
        "hiddenSize": 768,
        "rank": 0,
        "worldSize": 8,
        "vocabSize": 30522,
        "head_dim": 64,
        "isQuant": true,
        "outputHiddenStates": true,
        "enableAddNorm": true,
        "rankTableFile": "rank_table.txt",
        "positionEmbeddingType": "ROPE",
        "packQuantType": [[0, 1]],
        "linearQuantType": [[0, 0, 1, 1, 1, 1, 1]],
        "linearTransposeType": [[1, 1, 0, 0, 0, 0, 0]],
        "linearHasBias": [true],
        "quantGroupSize": 16,
        "backend": "NPU"
    })";

    llama::DecoderModelEdge::Param decoderParam;
    
    // 测试正常情况
    EXPECT_NO_THROW(decoderParam.FromString(param));
    
    // 验证ParseBasicParams解析的参数
    EXPECT_TRUE(decoderParam.skipWordEmbedding);
    EXPECT_FALSE(decoderParam.isFA);
    EXPECT_TRUE(decoderParam.isPrefill);
    EXPECT_FALSE(decoderParam.isBF16);
    EXPECT_EQ(decoderParam.lmHeadTransposeType, 1);
    EXPECT_TRUE(decoderParam.supportSwiGLU);
    EXPECT_EQ(decoderParam.attnBackend, 2);
    EXPECT_FLOAT_EQ(decoderParam.rmsNormEps, 0.1f);
    EXPECT_EQ(decoderParam.numAttentionHeadsPerRank, 8U);
    EXPECT_EQ(decoderParam.hiddenSizePerAttentionHead, 64U);
    EXPECT_EQ(decoderParam.numHiddenLayers, 1U);
    EXPECT_EQ(decoderParam.numKeyValueHeadsPerRank, 8U);
    EXPECT_EQ(decoderParam.hiddenSize, 768U);
    EXPECT_EQ(decoderParam.rank, 0);
    EXPECT_EQ(decoderParam.worldSize, 8);
    EXPECT_EQ(decoderParam.vocabSize, 30522);
    EXPECT_EQ(atb_speed::llama::g_headDim, 64U);
    EXPECT_TRUE(decoderParam.isQuant);
    EXPECT_TRUE(decoderParam.outputHiddenStates);
    
    // 验证AddParamJson解析的参数
    EXPECT_TRUE(decoderParam.enableAddNorm);
    EXPECT_EQ(decoderParam.rankTableFile, "rank_table.txt");
    EXPECT_EQ(decoderParam.positionEmbeddingType, "ROPE");
    ASSERT_EQ(decoderParam.packQuantType.size(), 1ULL);
    EXPECT_EQ(decoderParam.packQuantType[0], std::vector<int>({0, 1}));
    ASSERT_EQ(decoderParam.linearQuantType.size(), 1ULL);
    EXPECT_EQ(decoderParam.linearQuantType[0], std::vector<int>({0, 0, 1, 1, 1, 1, 1}));
    ASSERT_EQ(decoderParam.linearTransposeType.size(), 1ULL);
    EXPECT_EQ(decoderParam.linearTransposeType[0], std::vector<int>({1, 1, 0, 0, 0, 0, 0}));
    ASSERT_EQ(decoderParam.linearHasBias.size(), 1ULL);
    EXPECT_TRUE(decoderParam.linearHasBias[0]);
    EXPECT_EQ(decoderParam.quantGroupSize, 16U);
    EXPECT_EQ(decoderParam.backend, "NPU");
    
    // 测试rank >= worldSize的情况
    std::string invalidParam = R"({
        "skipWordEmbedding": true,
        "isFA": false,
        "isPrefill": true,
        "isBF16": false,
        "lmHeadTransposeType": 1,
        "enableSwiGLU": true,
        "attnBackend": 2,
        "normEps": 0.1,
        "numAttentionHeadsPerRank": 8,
        "hiddenSizePerAttentionHead": 64,
        "numHiddenLayers": 1,
        "numKeyValueHeadsPerRank": 8,
        "hiddenSize": 768,
        "rank": 8,
        "worldSize": 8,
        "vocabSize": 30522,
        "head_dim": 64,
        "isQuant": true,
        "outputHiddenStates": true,
        "enableAddNorm": true,
        "rankTableFile": "rank_table.txt",
        "positionEmbeddingType": "ROPE",
        "packQuantType": [[0, 1]],
        "linearQuantType": [[0, 0, 1, 1, 1, 1, 1]],
        "linearTransposeType": [[1, 1, 0, 0, 0, 0, 0]],
        "linearHasBias": [true],
        "quantGroupSize": 16,
        "backend": "NPU"
    })";
    EXPECT_THROW(decoderParam.FromString(invalidParam), std::runtime_error);
    
    // 测试JSON格式错误的情况
    invalidParam = R"({
        "skipWordEmbedding": true,
        "isFA": false,
        "isPrefill": true,
        "isBF16": false,
        "lmHeadTransposeType": 1,
        "enableSwiGLU": true,
        "attnBackend": 2,
        "normEps": 0.1,
        "numAttentionHeadsPerRank": 8,
        "hiddenSizePerAttentionHead": 64,
        "numHiddenLayers": 1,
        "numKeyValueHeadsPerRank": 8,
        "hiddenSize": 768,
        "rank": 8,
        "worldSize": 8,
        "vocabSize": 30522,
        "head_dim": 64,
        "isQuant": true,
        "outputHiddenStates": true,
        "enableAddNorm": true,
        "rankTableFile": "rank_table.txt",
        "positionEmbeddingType": "ROPE",
        "packQuantType": [[0, 1]],
        "linearQuantType": [[0, 0, 1, 1, 1, 1, 1]],
        "linearTransposeType": [[1, 1, 0, 0, 0, 0, 0]],
        "linearHasBias": [true],
        "quantGroupSize": 16,
        "backend": "NPU"
    )"; // 缺少闭合括号
    EXPECT_THROW(decoderParam.FromString(invalidParam), std::runtime_error);
}

class TestableDecoderModelEdge : public llama::DecoderModelEdge {
public:
    using DecoderModelEdge::DecoderModelEdge;

    const std::string& get_model_name() const { return modelName_; }
    const std::map<std::string, uint32_t>& get_in_tensor_map() const { return inTensorMap_; }
    const std::map<std::string, uint32_t>& get_internal_tensor_map() const { return internalTensorMap_; }
    uint32_t get_in_tensor_count() const { return inTensorCount_; }
    uint32_t get_internal_tensor_count() const { return internalTensorCount_; }
    uint32_t get_weightCountPerLayer() const { return weightCountPerLayer_; }
    const DecoderModelEdge::Param& get_param() const { return param_; }
    const Model::Graph& get_graph() const { return graph_; }
    const std::vector<int>& get_tokenOffset() const { return tokenOffset_; }
    const std::vector<int>& get_seqLen() const { return seqLen_; }
    const std::vector<int>& get_qLen() const { return qLen_; }
    const std::vector<int>& get_blockNumsList() const { return blockNumsList_; }
};

TEST(DecoderModelEdgeTest, Constructor) {
    GlobalMockObject::verify();

    std::string prefill_param = R"({
        "skipWordEmbedding": true,
        "isFA": false,
        "isPrefill": true,
        "isBF16": false,
        "lmHeadTransposeType": 1,
        "enableSwiGLU": true,
        "attnBackend": 2,
        "normEps": 0.1,
        "numAttentionHeadsPerRank": 8,
        "hiddenSizePerAttentionHead": 64,
        "numHiddenLayers": 1,
        "numKeyValueHeadsPerRank": 8,
        "hiddenSize": 768,
        "rank": 0,
        "worldSize": 8,
        "vocabSize": 30522,
        "head_dim": 64,
        "isQuant": true,
        "outputHiddenStates": true,
        "enableAddNorm": true,
        "rankTableFile": "rank_table.txt",
        "positionEmbeddingType": "ROPE",
        "packQuantType": [[0, 1]],
        "linearQuantType": [[0, 0, 1, 1, 1, 1, 1]],
        "linearTransposeType": [[1, 1, 0, 0, 0, 0, 0]],
        "linearHasBias": [true],
        "quantGroupSize": 16,
        "backend": "NPU"
    })";
    TestableDecoderModelEdge prefill_model(prefill_param);
    EXPECT_EQ(prefill_model.get_model_name(), "DecoderModelEdge_Prefill");
    EXPECT_TRUE(prefill_model.get_param().isPrefill);

    std::string decoder_param = R"({
        "skipWordEmbedding": true,
        "isFA": false,
        "isPrefill": false,
        "isBF16": false,
        "lmHeadTransposeType": 1,
        "enableSwiGLU": true,
        "attnBackend": 2,
        "normEps": 0.1,
        "numAttentionHeadsPerRank": 8,
        "hiddenSizePerAttentionHead": 64,
        "numHiddenLayers": 1,
        "numKeyValueHeadsPerRank": 8,
        "hiddenSize": 768,
        "rank": 0,
        "worldSize": 8,
        "vocabSize": 30522,
        "head_dim": 64,
        "isQuant": true,
        "outputHiddenStates": true,
        "enableAddNorm": true,
        "rankTableFile": "rank_table.txt",
        "positionEmbeddingType": "ROPE",
        "packQuantType": [[0, 1]],
        "linearQuantType": [[0, 0, 1, 1, 1, 1, 1]],
        "linearTransposeType": [[1, 1, 0, 0, 0, 0, 0]],
        "linearHasBias": [true],
        "quantGroupSize": 16,
        "backend": "NPU"
    })";
    TestableDecoderModelEdge decoder_model(decoder_param);
    EXPECT_EQ(decoder_model.get_model_name(), "DecoderModelEdge_Decoder");
    EXPECT_FALSE(decoder_model.get_param().isPrefill);
}

// 测试2：GetLlamaModelInTensorCandidates函数测试
TEST(DecoderModelEdgeTest, ConstructInTensorMapWithoutSkipEmbedding) {
    GlobalMockObject::verify();

    std::string param = R"({
        "skipWordEmbedding": false,
        "isFA": false,
        "isPrefill": true,
        "isBF16": false,
        "lmHeadTransposeType": 1,
        "enableSwiGLU": true,
        "attnBackend": 2,
        "normEps": 0.1,
        "numAttentionHeadsPerRank": 8,
        "hiddenSizePerAttentionHead": 64,
        "numHiddenLayers": 1,
        "numKeyValueHeadsPerRank": 8,
        "hiddenSize": 768,
        "rank": 0,
        "worldSize": 8,
        "packQuantType": [[0, 1]],
        "linearQuantType": [[0, 0, 1, 1, 1, 1, 1]],
        "linearTransposeType": [[1, 1, 0, 0, 0, 0, 0]],
        "backend": "NPU"
    })";
    TestableDecoderModelEdge model(param);
    
    model.ConstructInTensorMap();
    
    EXPECT_EQ(model.get_in_tensor_count(), 8U);
    
    const auto& tensor_map = model.get_in_tensor_map();
    EXPECT_EQ(tensor_map.at("input_ids"), 0U);
    EXPECT_EQ(tensor_map.at("positional_ids"), 1U);
    EXPECT_EQ(tensor_map.at("cosine_table"), 2U);
    EXPECT_EQ(tensor_map.at("sine_table"), 3U);
    EXPECT_EQ(tensor_map.at("attention_mask"), 4U);
    EXPECT_EQ(tensor_map.at("place_holder"), 5U);
    EXPECT_EQ(tensor_map.at("seq_len"), 6U);
    EXPECT_EQ(tensor_map.at("in_tensor_past_key"), 7U);
}

TEST(DecoderModelEdgeTest, ConstructInTensorMapWithSkipEmbedding) {
    GlobalMockObject::verify();

    std::string param = R"({
        "skipWordEmbedding": true,
        "isFA": false,
        "isPrefill": true,
        "isBF16": false,
        "lmHeadTransposeType": 1,
        "enableSwiGLU": true,
        "attnBackend": 2,
        "normEps": 0.1,
        "numAttentionHeadsPerRank": 8,
        "hiddenSizePerAttentionHead": 64,
        "numHiddenLayers": 1,
        "numKeyValueHeadsPerRank": 8,
        "hiddenSize": 768,
        "rank": 0,
        "worldSize": 8,
        "packQuantType": [[0, 1]],
        "linearQuantType": [[0, 0, 1, 1, 1, 1, 1]],
        "linearTransposeType": [[1, 1, 0, 0, 0, 0, 0]],
        "backend": "NPU"
    })";
    TestableDecoderModelEdge model(param);
    
    model.ConstructInTensorMap();
    
    EXPECT_EQ(model.get_in_tensor_count(), 8U);
    
    const auto& tensor_map = model.get_in_tensor_map();
    EXPECT_EQ(tensor_map.at("input_embedding"), 0U);
    EXPECT_EQ(tensor_map.at("positional_ids"), 1U);
    EXPECT_EQ(tensor_map.at("cosine_table"), 2U);
    EXPECT_EQ(tensor_map.at("sine_table"), 3U);
    EXPECT_EQ(tensor_map.at("attention_mask"), 4U);
    EXPECT_EQ(tensor_map.at("place_holder"), 5U);
    EXPECT_EQ(tensor_map.at("seq_len"), 6U);
    EXPECT_EQ(tensor_map.at("in_tensor_past_key"), 7U);
}

TEST(DecoderModelEdgeTest, ConstructInternalTensorMapWithROPE) {
    GlobalMockObject::verify();

    std::string param = R"({
        "skipWordEmbedding": true,
        "isFA": false,
        "isPrefill": true,
        "isBF16": false,
        "lmHeadTransposeType": 1,
        "enableSwiGLU": true,
        "attnBackend": 2,
        "normEps": 0.1,
        "numAttentionHeadsPerRank": 8,
        "hiddenSizePerAttentionHead": 64,
        "numHiddenLayers": 1,
        "numKeyValueHeadsPerRank": 8,
        "hiddenSize": 768,
        "rank": 0,
        "worldSize": 8,
        "positionEmbeddingType": "ROPE",
        "packQuantType": [[0, 1]],
        "linearQuantType": [[0, 0, 1, 1, 1, 1, 1]],
        "linearTransposeType": [[1, 1, 0, 0, 0, 0, 0]],
        "backend": "NPU"
    })";
    TestableDecoderModelEdge model(param);
    
    model.ConstructInternalTensorMap();
    
    const auto& internalTensorMap = model.get_internal_tensor_map();

    EXPECT_EQ(internalTensorMap.at("internal_tensor_hidden_states"), 0U);
    EXPECT_EQ(internalTensorMap.at("cosine_embedding"), 1U);
    EXPECT_EQ(internalTensorMap.at("sine_embedding"), 2U);
    EXPECT_EQ(internalTensorMap.size(), 3ULL);
}

TEST(DecoderModelEdgeTest, ConstructInternalTensorMapWithoutROPE) {
    GlobalMockObject::verify();

    std::string param = R"({
        "skipWordEmbedding": true,
        "isFA": false,
        "isPrefill": true,
        "isBF16": false,
        "lmHeadTransposeType": 1,
        "enableSwiGLU": true,
        "attnBackend": 2,
        "normEps": 0.1,
        "numAttentionHeadsPerRank": 8,
        "hiddenSizePerAttentionHead": 64,
        "numHiddenLayers": 1,
        "numKeyValueHeadsPerRank": 8,
        "hiddenSize": 768,
        "rank": 0,
        "worldSize": 8,
        "positionEmbeddingType": "NTK",
        "packQuantType": [[0, 1]],
        "linearQuantType": [[0, 0, 1, 1, 1, 1, 1]],
        "linearTransposeType": [[1, 1, 0, 0, 0, 0, 0]],
        "backend": "NPU"
    })";
    TestableDecoderModelEdge model(param);

    model.ConstructInternalTensorMap();

    const auto& internalTensorMap = model.get_internal_tensor_map();
    EXPECT_EQ(internalTensorMap.at("internal_tensor_hidden_states"), 0U);
    EXPECT_EQ(internalTensorMap.size(), 1ULL);
}

TEST(DecoderModelEdgeTest, ParseParamNormal) {
    GlobalMockObject::verify();

    std::string param = R"({
        "skipWordEmbedding": true,
        "isFA": false,
        "isPrefill": true,
        "isBF16": false,
        "lmHeadTransposeType": 1,
        "enableSwiGLU": true,
        "attnBackend": 2,
        "normEps": 0.1,
        "numAttentionHeadsPerRank": 8,
        "hiddenSizePerAttentionHead": 64,
        "numHiddenLayers": 1,
        "numKeyValueHeadsPerRank": 8,
        "hiddenSize": 768,
        "rank": 0,
        "worldSize": 8,
        "vocabSize": 30522,
        "head_dim": 64,
        "isQuant": true,
        "outputHiddenStates": true,
        "enableAddNorm": true,
        "rankTableFile": "rank_table.txt",
        "positionEmbeddingType": "NTK",
        "packQuantType": [[0, 1]],
        "linearQuantType": [[0, 0, 1, 1, 1, 1, 1]],
        "linearTransposeType": [[1, 1, 0, 0, 0, 0, 0]],
        "linearHasBias": [true],
        "quantGroupSize": 16,
        "backend": "NPU"
    })";
    TestableDecoderModelEdge model(param);

    std::string _param = R"({
        "tokenOffset": [1, 2, 3],
        "seqLen": [10, 20, 30],
        "qLen": [5, 10, 15],
        "blockNumsList": [100, 200, 300]
    })";

    EXPECT_NO_THROW(model.ParseParam(_param));
    EXPECT_EQ(model.get_tokenOffset(), std::vector<int>({1, 2, 3}));
    EXPECT_EQ(model.get_seqLen(), std::vector<int>({10, 20, 30}));
    EXPECT_EQ(model.get_qLen(), std::vector<int>({5, 10, 15}));
    EXPECT_EQ(model.get_blockNumsList(), std::vector<int>({100, 200, 300}));
}

} // namespace atb_speed
