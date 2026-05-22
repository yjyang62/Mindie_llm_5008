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
#include "models/qwen3vl/model/moe_decoder_model.h"

namespace atb_speed {

constexpr int QWEN3VL_MOE_WEIGHT_COUNT_PER_LAYER = 52;
constexpr int QWEN3VL_MOE_INTENSORS_SIZE_PREFILL = 68;
constexpr int QWEN3VL_MOE_INTENSORS_SIZE_DECODE = 65;

bool CheckQwen3vlMoeDeepStack(const atb_speed::qwen3vl::MoeDecoderModel &decoderModel)
{
    constexpr int QWEN3VL_MOE_DEEPSTACK_COUNT = 3;
    const std::string visualEmbedKey = "deepstack_visual_embeds";
    auto it = decoderModel.inTensorCandidates.find(visualEmbedKey);
    if (it == decoderModel.inTensorCandidates.end() || it->second.size() != QWEN3VL_MOE_DEEPSTACK_COUNT) {
        return false;
    }
    return true;
}

bool CheckQwen3vlSkipWordEmbedding(const atb_speed::qwen3vl::MoeDecoderModel &decoderModel)
{
    if (!decoderModel.param.skipWordEmbedding) {
        return false;
    }
    return true;
}

bool CheckQwen3vlInTensorMap(const atb_speed::qwen3vl::MoeDecoderModel &decoderModel, bool isPrefill)
{
    constexpr int QWEN3VL_MOE_INTENSORMAP_COUNT_DECODE = 16;
    constexpr int QWEN3VL_MOE_INTENSORMAP_COUNT_PREFILL = 19;
    if (!isPrefill && decoderModel.inTensorMap.size() == QWEN3VL_MOE_INTENSORMAP_COUNT_DECODE) {
        return true;
    } else if (isPrefill && decoderModel.inTensorMap.size() == QWEN3VL_MOE_INTENSORMAP_COUNT_PREFILL) {
        return true;
    } else {
        return false;
    }

}

std::string CreateQwen3vlMoeModelParam(bool isPrefillValue)
{
    std::ostringstream param;
    param << R"({
        "enableAddNorm": false,
        "normEps": 1e-06,
        "normType": 0,
        "numAttentionHeadsPerRank": 8,
        "hiddenSizePerAttentionHead": 128,
        "numHiddenLayers": 1,
        "numKeyValueHeadsPerRank": 1,
        "isFA": false,
        "isBF16": true,
        "packQuantType": [[1, 1]],
        "quantGroupSize": 0,
        "linearQuantType": [[0, -1, -1, 0, -1, -1, -1]],
        "mlpLinearQuantType": [[-1, -1, -1, -1]],
        "moeLinearQuantType": [[0, 0, -1, 0]],
        "linearTransposeType": [[1, -1, -1, 1, 1, -1, 1]],
        "mlpLinearTransposeType": [[-1, -1, -1, -1]],
        "moeLinearTransposeType": [[1, 1, -1, 1]],
        "lmHeadTransposeType": 1,
        "isUnpadInputs": true,
        "skipWordEmbedding": true,
        "isLmHeadParallel": false,
        "enableSwiGLU": false,
        "rank": 0,
        "worldSize": 1,
        "numOfExperts": 128,
        "numOfSelectedExperts": 8,
        "routingMethod": "softMaxTopK",
        "enableFusedRouting": true,
        "isDenseLayer": [false],
        "hasSharedExpert": false,
        "backend": "hccl",
        "mapping": {
            "worldSize": 1,
            "rank": 0,
            "rankTableFile": "",
            "localWorldSize": 1,
            "lcclCommDomainLowerBound": 0,
            "lcclCommDomainUpperBound": 65536,
            "wordEmbedTp": {"groupId": 0, "rankIds": [0], "rank": 0, "bufferSize": 128},
            "wordEmbedDp": {"groupId": 0, "rankIds": [0], "rank": 0, "bufferSize": 128},
            "attnTp": {"groupId": 0, "rankIds": [0], "rank": 0, "bufferSize": 128},
            "attnDp": {"groupId": 0, "rankIds": [0], "rank": 0, "bufferSize": 128},
            "attnInnerSp": {"groupId": 0, "rankIds": [0], "rank": 0, "bufferSize": 128},
            "attnCp": {"groupId": 0, "rankIds": [0], "rank": 0, "bufferSize": 128},
            "attnPrefixcacheCp": {"groupId": 0, "rankIds": [0], "rank": 0, "bufferSize": 128},
            "attnOProjTp": {"groupId": 0, "rankIds": [0], "rank": 0, "bufferSize": 128},
            "attnOProjDp": {"groupId": 0, "rankIds": [0], "rank": 0, "bufferSize": 128},
            "mlpTp": {"groupId": 0, "rankIds": [0], "rank": 0, "bufferSize": 128},
            "mlpDp": {"groupId": 0, "rankIds": [0], "rank": 0, "bufferSize": 128},
            "moeTp": {"groupId": 0, "rankIds": [0], "rank": 0, "bufferSize": 64},
            "moeEp": {"groupId": 0, "rankIds": [0], "rank": 0, "bufferSize": 512},
            "moeEpIntraNode": {"groupId": 0, "rankIds": [0], "rank": 0, "bufferSize": 128},
            "moeEpInterNode": {"groupId": 0, "rankIds": [0], "rank": 0, "bufferSize": 128},
            "lmHeadTp": {"groupId": 0, "rankIds": [0], "rank": 0, "bufferSize": 128},
            "lmHeadDp": {"groupId": 0, "rankIds": [0], "rank": 0, "bufferSize": 128},
            "denseTp": {"groupId": 0, "rankIds": [0], "rank": 0, "bufferSize": 128},
            "dynamicEplb": {"groupId": 0, "rankIds": [0], "rank": 0, "bufferSize": 128}
        },
        "positionEmbeddingType": 0,
        "linearHasBias": [[false, false, false, false]],
        "useQKNorm": true,
        "supportLcoc": false,
        "isPrefill":)";
    param << (isPrefillValue ? "true" : "false");
    param << R"(
    })";
    return param.str();
}

TEST(Qwen3vlMoeDecoderModelTest, Qwen3vlMoeDecoderModelConstructInTensorMapDecode)
{
    GlobalMockObject::verify();

    bool isPrefill = false;
    const std::string param = CreateQwen3vlMoeModelParam(isPrefill);
    atb_speed::qwen3vl::MoeDecoderModel decoderModel(param);
    decoderModel.ConstructInTensorMap();
    EXPECT_TRUE(CheckQwen3vlInTensorMap(decoderModel, isPrefill));
}

TEST(Qwen3vlMoeDecoderModelTest, Qwen3vlMoeDecoderModelSetLayerNodeInputDecode)
{
    GlobalMockObject::verify();

    bool isPrefill = false;
    const std::string param = CreateQwen3vlMoeModelParam(isPrefill);
    atb_speed::qwen3vl::MoeDecoderModel decoderModel(param);
    decoderModel.ConstructInTensorMap();
    decoderModel.ConstructInternalTensorMap();
    decoderModel.ConstructOutTensorMap();

    // The size of inTensor for prefill is 70; otherwise, it is 67 (corresponding to 3 deepstack_visual_embeds).
    atb_speed::Model::Node layerNode;
    decoderModel.graph_.inTensors.resize(decoderModel.inTensorMap.size());
    decoderModel.graph_.outTensors.resize(decoderModel.outTensorMap.size());
    decoderModel.graph_.weightTensors.resize(QWEN3VL_MOE_WEIGHT_COUNT_PER_LAYER);
    decoderModel.graph_.internalTensors.resize(decoderModel.internalTensorMap.size());
    decoderModel.graph_.kCacheTensors.resize(decoderModel.param.numHiddenLayers);
    decoderModel.graph_.vCacheTensors.resize(decoderModel.param.numHiddenLayers);
    layerNode.inTensors.resize(QWEN3VL_MOE_INTENSORS_SIZE_DECODE);
    layerNode.inTensorReshapeFuncs.resize(QWEN3VL_MOE_INTENSORS_SIZE_DECODE);
    decoderModel.SetLayerNodeInput(layerNode, 0);
    EXPECT_NE(layerNode.inTensors.back(), nullptr);
}

TEST(Qwen3vlMoeDecoderModelTest, Qwen3vlMoeDecoderModelCreateLayerOperationDecode)
{
    GlobalMockObject::verify();

    bool isPrefill = false;
    const std::string param = CreateQwen3vlMoeModelParam(isPrefill);
    atb_speed::qwen3vl::MoeDecoderModel decoderModel(param);
    atb::Operation *op = nullptr;
    MOCKER(atb::CreateOperation<atb::GraphParam>).expects(atLeast(1))
    .with(any(), any()).will(returnValue(0));
    atb::Status ret = decoderModel.CreateLayerOperation(&op, 0);
    EXPECT_EQ(ret, atb::NO_ERROR);
}

TEST(Qwen3vlMoeDecoderModelTest, Qwen3vlMoeDecoderModelConstructInTensorMapPrefill)
{
    GlobalMockObject::verify();

    bool isPrefill = true;
    const std::string param = CreateQwen3vlMoeModelParam(isPrefill);
    atb_speed::qwen3vl::MoeDecoderModel decoderModel(param);
    decoderModel.ConstructInTensorMap();
    EXPECT_TRUE(CheckQwen3vlInTensorMap(decoderModel, isPrefill));
}

TEST(Qwen3vlMoeDecoderModelTest, Qwen3vlMoeDecoderModelSetLayerNodeInputPrefill)
{
    GlobalMockObject::verify();

    bool isPrefill = true;
    const std::string param = CreateQwen3vlMoeModelParam(isPrefill);
    atb_speed::qwen3vl::MoeDecoderModel decoderModel(param);
    decoderModel.ConstructInTensorMap();
    decoderModel.ConstructInternalTensorMap();
    decoderModel.ConstructOutTensorMap();

    atb_speed::Model::Node layerNode;
    decoderModel.graph_.inTensors.resize(decoderModel.inTensorMap.size());
    decoderModel.graph_.outTensors.resize(decoderModel.outTensorMap.size());
    decoderModel.graph_.weightTensors.resize(QWEN3VL_MOE_WEIGHT_COUNT_PER_LAYER);
    decoderModel.graph_.internalTensors.resize(decoderModel.internalTensorMap.size());
    decoderModel.graph_.kCacheTensors.resize(decoderModel.param.numHiddenLayers);
    decoderModel.graph_.vCacheTensors.resize(decoderModel.param.numHiddenLayers);
    layerNode.inTensors.resize(QWEN3VL_MOE_INTENSORS_SIZE_PREFILL);
    layerNode.inTensorReshapeFuncs.resize(QWEN3VL_MOE_INTENSORS_SIZE_PREFILL);
    decoderModel.SetLayerNodeInput(layerNode, 0);
    EXPECT_NE(layerNode.inTensors.back(), nullptr);
}

TEST(Qwen3vlMoeDecoderModelTest, Qwen3vlMoeDecoderModelCreateLayerOperationPrefill)
{
    GlobalMockObject::verify();

    bool isPrefill = true;
    const std::string param = CreateQwen3vlMoeModelParam(isPrefill);
    atb_speed::qwen3vl::MoeDecoderModel decoderModel(param);
    atb::Operation *op = nullptr;
    MOCKER(atb::CreateOperation<atb::GraphParam>).expects(atLeast(1))
    .with(any(), any()).will(returnValue(0));
    atb::Status ret = decoderModel.CreateLayerOperation(&op, 0);
    EXPECT_EQ(ret, atb::NO_ERROR);
}

} // namespace atb_speed