/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
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
#include <atb/atb_infer.h>
#include <atb/types.h>
#include "models/qwen3vl/model/decoder_model.h"
#include "operations/fusion/infer_shape_functions.h"
#include <sstream>
#include <map>

namespace atb_speed {

constexpr int QWEN3VL_WEIGHT_COUNT_PER_LAYER = 52;
constexpr int QWEN3VL_INTENSORS_SIZE_PREFILL = 64;
constexpr int QWEN3VL_INTENSORS_SIZE_DECODE = 61;
constexpr int QWEN3VL_INTENSORS_SIZE_FLASHCOMM = 72;
constexpr int NUM3 = 3;
constexpr int NUM4 = 4;

std::string GenerateModelParam(bool isPrefillValue)
{
    std::ostringstream oss;
    oss << R"({
        "enableAddNorm": false,
        "normEps": 1e-06,
        "normType": 0,
        "numAttentionHeadsPerRank": 8,
        "hiddenSizePerAttentionHead": 128,
        "numHiddenLayers": 4,
        "numKeyValueHeadsPerRank": 1,
        "isFA": false,
        "isBF16": true,
        "packQuantType": [[1, 1], [1, 1], [1, 1], [1, 1]],
        "quantGroupSize": 0,
        "linearQuantType": [[0, -1, -1, 0, -1, -1, -1],
                            [0, -1, -1, 0, -1, -1, -1],
                            [0, -1, -1, 0, -1, -1, -1],
                            [0, -1, -1, 0, -1, -1, -1]],
        "linearTransposeType": [[1, -1, -1, 1, 1, -1, 1],
                                [1, -1, -1, 1, 1, -1, 1],
                                [1, -1, -1, 1, 1, -1, 1],
                                [1, -1, -1, 1, 1, -1, 1]],
        "lmHeadTransposeType": 1,
        "isUnpadInputs": true,
        "skipWordEmbedding": true,
        "isLmHeadParallel": false,
        "enableSwiGLU": false,
        "rank": 0,
        "worldSize": 1,
        "backend": "hccl",
        "positionEmbeddingType": 0,
        "linearHasBias": [[false, false, false, false],
                          [false, false, false, false],
                          [false, false, false, false],
                          [false, false, false, false]],
        "useQKNorm": true,
        "isPrefill":)";
    oss << (isPrefillValue ? "true" : "false");
    oss << R"(,
        "enableLcoc": false
    })";
    return oss.str();
}

void AddFlashCommParam(std::string &param)
{
    std::string target = "\"enableLcoc\": false";
    size_t pos = param.find(target);
    if (pos != std::string::npos) {
        param.insert(pos + target.length(), ",\"enableFlashComm\": true");
    }
}

std::string CreateModelParam(bool isPrefillValue)
{
    return GenerateModelParam(isPrefillValue);
}

bool CheckQwen3vlDeepStack(const atb_speed::qwen3vl::DecoderModel &decoderModel)
{
    constexpr size_t QWEN3VL_DEEPSTACK_COUNT = 3;
    const std::string visualEmbedKey = "deepstack_visual_embeds";
    auto it = decoderModel.inTensorCandidates.find(visualEmbedKey);
    return it != decoderModel.inTensorCandidates.end() && it->second.size() == QWEN3VL_DEEPSTACK_COUNT;
}


TEST(Qwen3vlDecoderModelTest, Qwen3vlDecoderModelConstructInTensorMapPrefillTrue)
{
    GlobalMockObject::verify();

    atb_speed::qwen3vl::DecoderModel decoderModel(GenerateModelParam(true));
    EXPECT_TRUE(CheckQwen3vlDeepStack(decoderModel));

    decoderModel.ConstructInTensorMap();
    EXPECT_EQ(decoderModel.inTensorMap.size(), static_cast<size_t>(15));
}

TEST(Qwen3vlDecoderModelTest, Qwen3vlDecoderModelSetLayerNodeInputPrefillTrue)
{
    GlobalMockObject::verify();

    atb_speed::qwen3vl::DecoderModel decoderModel(GenerateModelParam(true));
    decoderModel.ConstructInTensorMap();
    decoderModel.ConstructInternalTensorMap();
    decoderModel.ConstructOutTensorMap();

    // prefill的inTensor大小是70否则是67(3个deepstack_visual_embeds)
    atb_speed::Model::Node layerNode;
    decoderModel.graph_.inTensors.resize(decoderModel.inTensorMap.size());
    decoderModel.graph_.outTensors.resize(decoderModel.outTensorMap.size());
    decoderModel.graph_.weightTensors.resize(QWEN3VL_WEIGHT_COUNT_PER_LAYER * NUM4);
    decoderModel.graph_.internalTensors.resize(decoderModel.internalTensorMap.size());
    decoderModel.graph_.kCacheTensors.resize(decoderModel.param.numHiddenLayers);
    decoderModel.graph_.vCacheTensors.resize(decoderModel.param.numHiddenLayers);
    layerNode.inTensors.resize(QWEN3VL_INTENSORS_SIZE_PREFILL);
    layerNode.inTensorReshapeFuncs.resize(QWEN3VL_INTENSORS_SIZE_PREFILL);
    decoderModel.SetLayerNodeInput(layerNode, 0);
    EXPECT_NE(layerNode.inTensors.back(), nullptr);
    
}

TEST(Qwen3vlDecoderModelTest, Qwen3vlDecoderModelCreateLayerOperationPrefillTrue)
{
    GlobalMockObject::verify();

    atb_speed::qwen3vl::DecoderModel decoderModel(GenerateModelParam(true));
    decoderModel.ConstructInTensorMap();

    atb::Operation *op = nullptr;
    MOCKER(atb::CreateOperation<atb::GraphParam>).expects(atLeast(1))
    .with(any(), any()).will(returnValue(0));
    atb::Status ret = decoderModel.CreateLayerOperation(&op, 0);
    EXPECT_EQ(ret, atb::NO_ERROR);
}

TEST(Qwen3vlDecoderModelTest, Qwen3vlDecoderModelConstructInTensorMapPrefillFalse)
{
    GlobalMockObject::verify();

    atb_speed::qwen3vl::DecoderModel decoderModel(GenerateModelParam(false));
    EXPECT_TRUE(CheckQwen3vlDeepStack(decoderModel));

    decoderModel.ConstructInTensorMap();
    EXPECT_EQ(decoderModel.inTensorMap.size(), static_cast<size_t>(12));
}

TEST(Qwen3vlDecoderModelTest, Qwen3vlDecoderModelSetLayerNodeInputPrefillFalse)
{
    GlobalMockObject::verify();

    atb_speed::qwen3vl::DecoderModel decoderModel(GenerateModelParam(false));
    decoderModel.ConstructInTensorMap();
    decoderModel.ConstructInternalTensorMap();
    decoderModel.ConstructOutTensorMap();

    // prefill的inTensor大小是70否则是67(3个deepstack_visual_embeds)
    atb_speed::Model::Node layerNode;
    decoderModel.graph_.inTensors.resize(decoderModel.inTensorMap.size());
    decoderModel.graph_.outTensors.resize(decoderModel.outTensorMap.size());
    decoderModel.graph_.weightTensors.resize(QWEN3VL_WEIGHT_COUNT_PER_LAYER * NUM4);
    decoderModel.graph_.internalTensors.resize(decoderModel.internalTensorMap.size());
    decoderModel.graph_.kCacheTensors.resize(decoderModel.param.numHiddenLayers);
    decoderModel.graph_.vCacheTensors.resize(decoderModel.param.numHiddenLayers);
    layerNode.inTensors.resize(QWEN3VL_INTENSORS_SIZE_DECODE);
    layerNode.inTensorReshapeFuncs.resize(QWEN3VL_INTENSORS_SIZE_DECODE);
    decoderModel.SetLayerNodeInput(layerNode, 0);

    EXPECT_NE(layerNode.inTensors.back(), nullptr);
}

TEST(Qwen3vlDecoderModelTest, Qwen3vlDecoderModelCreateLayerOperationPrefillFalse)
{
    GlobalMockObject::verify();

    atb_speed::qwen3vl::DecoderModel decoderModel(GenerateModelParam(false));
    decoderModel.ConstructInTensorMap();

    atb::Operation *op = nullptr;
    MOCKER(atb::CreateOperation<atb::GraphParam>).expects(atLeast(1))
    .with(any(), any()).will(returnValue(0));
    atb::Status ret = decoderModel.CreateLayerOperation(&op, 0);
    EXPECT_EQ(ret, atb::NO_ERROR);
}

TEST(Qwen3vlDecoderModelTest, Qwen3vlDecoderModelSetLayerNodeInputFlashComm)
{
    GlobalMockObject::verify();
    std::string param = GenerateModelParam(true);
    AddFlashCommParam(param);
    atb_speed::qwen3vl::DecoderModel decoderModel(param);
    decoderModel.ConstructInTensorMap();
    decoderModel.ConstructInternalTensorMap();
    decoderModel.ConstructOutTensorMap();

    atb_speed::Model::Node layerNode;
    decoderModel.graph_.inTensors.resize(decoderModel.inTensorMap.size());
    decoderModel.graph_.outTensors.resize(decoderModel.outTensorMap.size());
    decoderModel.graph_.weightTensors.resize(QWEN3VL_WEIGHT_COUNT_PER_LAYER * NUM4);
    decoderModel.graph_.internalTensors.resize(decoderModel.internalTensorMap.size());
    decoderModel.graph_.kCacheTensors.resize(decoderModel.param.numHiddenLayers);
    decoderModel.graph_.vCacheTensors.resize(decoderModel.param.numHiddenLayers);
    layerNode.inTensors.resize(QWEN3VL_INTENSORS_SIZE_FLASHCOMM);
    layerNode.inTensorReshapeFuncs.resize(QWEN3VL_INTENSORS_SIZE_FLASHCOMM);
    decoderModel.SetLayerNodeInput(layerNode, NUM3);
    EXPECT_NE(layerNode.inTensors.back(), nullptr);
}

}