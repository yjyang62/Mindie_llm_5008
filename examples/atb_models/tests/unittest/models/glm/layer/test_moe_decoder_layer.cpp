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
#include "models/moe/layer/decoder_layer.h"
#include "models/glm/layer/moe_decoder_layer.h"

namespace atb_speed {

bool CheckFusionAttentionParam(
    const atb_speed::common::FusionAttentionParam<atb::infer::RmsNormParam> &fusionAttentionParam)
{
    if (fusionAttentionParam.rotaryType != atb_speed::common::RotaryType::HALF_ROTARY) {
        return false;
    }
    // Check if rotaryCoeff == 2 (Half rotary)
    if (fusionAttentionParam.ropeParam.rotaryCoeff != 2) {
        return false;
    }
    return true;
}

TEST(glmMoeDecoderLayer, MoeDecoderLayer)
{
    GlobalMockObject::verify();

    atb_speed::moe::MoeLayerParam layerParam;
    layerParam.hiddenSizePerAttentionHead = 1;
    atb_speed::glm::MoeDecoderLayer<atb::infer::RmsNormParam> decoderLayer(layerParam);

    atb_speed::common::FusionAttentionParam<atb::infer::RmsNormParam> fusionAttentionParam;
    decoderLayer.SetFusionAttentionParam(fusionAttentionParam);

    EXPECT_TRUE(CheckFusionAttentionParam(fusionAttentionParam));
}
} // namespace atb_speed