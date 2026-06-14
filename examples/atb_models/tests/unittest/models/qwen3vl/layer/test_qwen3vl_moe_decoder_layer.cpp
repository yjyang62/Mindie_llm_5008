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
#include "models/qwen3vl/layer/moe_decoder_layer.h"

namespace atb_speed {

static const uint64_t NUM128 = 128;
static const uint64_t NUM70 = 70;

bool CheckQwen3vlMoeFusionAttentionParam(
    const atb_speed::common::FusionAttentionParam<atb::infer::RmsNormParam> &fusionAttentionParam)
{
    if (fusionAttentionParam.rotaryType != atb_speed::common::RotaryType::ALL_ROTARY) {
        return false;
    }
    return true;
}

atb_speed::common::ParallelInfo CreatParallelInfo(uint32_t bufferSize)
{
    atb_speed::common::ParallelInfo parallelInfo;
    parallelInfo.groupId = 0;
    parallelInfo.rankIds = {0};
    parallelInfo.rank = 0;
    parallelInfo.bufferSize = bufferSize;
    return parallelInfo;
}

atb_speed::moe::MoeLayerParam CreatQwen3vlMoeLayerParam()
{
    atb_speed::moe::MoeLayerParam layerParam;
    std::map<atb_speed::base::ParallelType, atb_speed::common::ParallelInfo> testparallelStrategies;
    atb_speed::common::ParallelInfo parallelInfo = CreatParallelInfo(128);
    testparallelStrategies[atb_speed::base::WORD_EMBED_TP] = parallelInfo;
    testparallelStrategies[atb_speed::base::WORD_EMBED_DP] = parallelInfo;
    testparallelStrategies[atb_speed::base::ATTN_TP] = parallelInfo;
    testparallelStrategies[atb_speed::base::ATTN_DP] = parallelInfo;
    testparallelStrategies[atb_speed::base::ATTN_INNER_SP] = parallelInfo;
    testparallelStrategies[atb_speed::base::ATTN_CP] = parallelInfo;
    testparallelStrategies[atb_speed::base::ATTN_PREFIX_CACHE_CP] = parallelInfo;
    testparallelStrategies[atb_speed::base::ATTN_O_PROJ_TP] = parallelInfo;
    testparallelStrategies[atb_speed::base::ATTN_O_PROJ_DP] = parallelInfo;
    testparallelStrategies[atb_speed::base::MLP_TP] = parallelInfo;
    testparallelStrategies[atb_speed::base::MLP_DP] = parallelInfo;
    testparallelStrategies[atb_speed::base::MOE_TP] = CreatParallelInfo(64);
    testparallelStrategies[atb_speed::base::MOE_EP] = CreatParallelInfo(512);
    testparallelStrategies[atb_speed::base::MOE_EP_INTER_NODE] = parallelInfo;
    testparallelStrategies[atb_speed::base::MOE_EP_INTRA_NODE] = parallelInfo;
    testparallelStrategies[atb_speed::base::LM_HEAD_TP] = parallelInfo;
    testparallelStrategies[atb_speed::base::LM_HEAD_DP] = parallelInfo;
    testparallelStrategies[atb_speed::base::DENSE_TP] = parallelInfo;
    testparallelStrategies[atb_speed::base::DYNAMIC_EPLB] = parallelInfo;
    testparallelStrategies[atb_speed::base::LM_HEAD_DP] = parallelInfo;
    testparallelStrategies[atb_speed::base::LM_HEAD_DP] = parallelInfo;
    layerParam.mapping.parallelStrategies_ = testparallelStrategies;
    layerParam.mapping.localWorldSize_ = 1;
    layerParam.mapping.worldSize_ = 1;
    layerParam.mapping.rank_ = 0;
    layerParam.hiddenSizePerAttentionHead = NUM128;
    layerParam.numAttentionHeadsPerRank = 8;
    layerParam.numKeyValueHeadsPerRank = 1;
    layerParam.linearQuantType = {0, -1, -1, 0, -1, -1, -1};
    layerParam.mlpLinearQuantType = {-1, -1, -1, -1};
    layerParam.moeLinearQuantType = {0, 0, -1, 0};
    layerParam.linearTransposeType = {1, -1, -1, 1, 1, -1, 1};
    layerParam.mlpLinearTransposeType = {-1, -1, -1, -1};
    layerParam.moeLinearTransposeType = {1, 1, -1, 1};
    layerParam.useQKNorm = true;
    layerParam.isPrefill = true;

    return layerParam;
}

TEST(Qwen3vlMoeDecoderLayerTest, TestConstructInTensorMap)
{
    GlobalMockObject::verify();

    atb_speed::moe::MoeLayerParam layerParam = CreatQwen3vlMoeLayerParam();
    atb_speed::qwen3vl::MoeDecoderLayer decoderLayer(layerParam);
    decoderLayer.ConstructInTensorMap();
    EXPECT_TRUE(decoderLayer.inTensorList.size() == NUM70);
}

TEST(Qwen3vlMoeDecoderLayerTest, TestSetFusionAttentionParam)
{
    GlobalMockObject::verify();

    atb_speed::moe::MoeLayerParam layerParam = CreatQwen3vlMoeLayerParam();
    atb_speed::qwen3vl::MoeDecoderLayer decoderLayer(layerParam);
    atb_speed::common::FusionAttentionParam<atb::infer::RmsNormParam> fusionAttentionParam;
    decoderLayer.SetFusionAttentionParam(fusionAttentionParam);
    EXPECT_TRUE(CheckQwen3vlMoeFusionAttentionParam(fusionAttentionParam));
}

TEST(Qwen3vlMoeDecoderLayerTest, TestAddDeepStack)
{
    GlobalMockObject::verify();

    atb_speed::moe::MoeLayerParam layerParam = CreatQwen3vlMoeLayerParam();
    atb_speed::qwen3vl::MoeDecoderLayer decoderLayer(layerParam);
    MOCKER(atb::CreateOperation<atb::infer::ElewiseParam>).expects(atLeast(1))
    .with(any(), any()).will(returnValue(0));
    atb::Status ret = decoderLayer.AddDeepStack(0);
    EXPECT_EQ(ret, atb::NO_ERROR);
}

TEST(Qwen3vlMoeDecoderLayerTest, TestBuildGraph)
{
    GlobalMockObject::verify();

    atb_speed::moe::MoeLayerParam layerParam = CreatQwen3vlMoeLayerParam();
    atb_speed::qwen3vl::MoeDecoderLayer decoderLayer(layerParam);
    atb::Operation *op = nullptr;
    MOCKER(atb::CreateOperation<atb::GraphParam>).expects(atLeast(1))
    .with(any(), any()).will(returnValue(0));
    atb::Status ret = decoderLayer.BuildGraph(&op);
    EXPECT_EQ(ret, atb::NO_ERROR);
}

} // namespace atb_speed