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
#include <atb/atb_infer.h>
#include <atb/types.h>
#include "models/glm41v/layer/decoder_layer.h"

namespace atb_speed {

static const uint64_t NUM128 = 128;
static const uint64_t NUM63 = 63;
static const uint64_t NUM64 = 64;

bool CheckGlm41vFusionAttentionParam(
    const atb_speed::common::FusionAttentionParam<atb::infer::RmsNormParam> &fusionAttentionParam)
{
    if (fusionAttentionParam.rotaryType != atb_speed::common::RotaryType::HALF_ROTARY) {
        return false;
    }
    if (fusionAttentionParam.ropeParam.rotaryCoeff != NUM64) {
        return false;
    }
    return true;
}

TEST(Glm41vDecoderLayerTest, DecoderLayer)
{
    GlobalMockObject::verify();

    atb_speed::base::LayerParam layerParam;
    layerParam.hiddenSizePerAttentionHead = NUM128;

    atb_speed::glm41v::DecoderLayer decoderLayer(layerParam);
    decoderLayer.ConstructInTensorMap();
    EXPECT_TRUE(decoderLayer.inTensorList.size() == NUM63);
    atb_speed::common::FusionAttentionParam<atb::infer::RmsNormParam> fusionAttentionParam;
    decoderLayer.SetFusionAttentionParam(fusionAttentionParam);
    EXPECT_TRUE(CheckGlm41vFusionAttentionParam(fusionAttentionParam));
    MOCKER(atb::CreateOperation<atb::infer::RmsNormParam>).expects(atLeast(1))
    .with(any(), any()).will(returnValue(0));
    atb::Status ret = decoderLayer.AddPostSelfAttentionRMSNorm();
    EXPECT_EQ(ret, atb::NO_ERROR);
    ret = decoderLayer.AddPostMlpRMSNorm();
    EXPECT_EQ(ret, atb::NO_ERROR);
}


} // namespace atb_speed