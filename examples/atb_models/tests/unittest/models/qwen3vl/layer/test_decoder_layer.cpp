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
#include "models/qwen3vl/layer/decoder_layer.h"

namespace atb_speed {

static const uint64_t NUM128 = 128;
static const uint64_t NUM66 = 66;

bool CheckQwen3vlFusionAttentionParam(
    const atb_speed::common::FusionAttentionParam<atb::infer::RmsNormParam> &fusionAttentionParam)
{
    if (fusionAttentionParam.rotaryType != atb_speed::common::RotaryType::ALL_ROTARY) {
        return false;
    }
    return true;
}

atb_speed::base::LayerParam CreatLayerParam()
{
    atb_speed::base::LayerParam layerParam;
    layerParam.hiddenSizePerAttentionHead = NUM128;
    layerParam.numAttentionHeadsPerRank = 8;
    layerParam.numKeyValueHeadsPerRank = 1;
    layerParam.linearQuantType = {0, -1, -1, 0, -1, -1, -1};
    layerParam.linearTransposeType = {1, -1, -1, 1, 1, -1, 1};
    layerParam.useQKNorm = true;
    layerParam.isPrefill = true;
    
    return layerParam;
}

TEST(Qwen3vlDecoderLayerTest, TestConstructInTensorMap)
{
    GlobalMockObject::verify();

    atb_speed::base::LayerParam layerParam = CreatLayerParam();
    atb_speed::qwen3vl::DecoderLayer decoderLayer(layerParam);
    decoderLayer.ConstructInTensorMap();
    EXPECT_EQ(decoderLayer.inTensorList.size(), NUM66);
}

TEST(Qwen3vlDecoderLayerTest, TestAddDeepStack)
{
    GlobalMockObject::verify();

    atb_speed::base::LayerParam layerParam = CreatLayerParam();
    atb_speed::qwen3vl::DecoderLayer decoderLayer(layerParam);
    MOCKER(atb::CreateOperation<atb::infer::ElewiseParam>).expects(atLeast(1))
    .with(any(), any()).will(returnValue(0));
    atb::Status ret = decoderLayer.AddDeepStack(0);
    EXPECT_EQ(ret, atb::NO_ERROR);
}

TEST(Qwen3vlDecoderLayerTest, TestBuildGraph)
{
    GlobalMockObject::verify();

    atb_speed::base::LayerParam layerParam = CreatLayerParam();
    atb_speed::qwen3vl::DecoderLayer decoderLayer(layerParam);
    atb::Operation *op = nullptr;
    MOCKER(atb::CreateOperation<atb::GraphParam>).expects(atLeast(1))
    .with(any(), any()).will(returnValue(0));
    atb::Status ret = decoderLayer.BuildGraph(&op);
    EXPECT_EQ(ret, atb::NO_ERROR);
}

} // namespace atb_speed
