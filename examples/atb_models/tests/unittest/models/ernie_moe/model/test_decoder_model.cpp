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
#include "models/ernie_moe/model/decoder_model.h"

namespace atb_speed {

bool CheckErnieWeightCountPerLayer(const atb_speed::ernie_moe::DecoderModel &decoderModel)
{
    const uint32_t WEIGHT_COUNT_DENSE_LAYER = 68;
    const uint32_t WEIGHT_COUNT_MOE_LAYER = 50;
    if (decoderModel.weightCountDenseLayer != WEIGHT_COUNT_DENSE_LAYER) {
        return false;
    }
    if (decoderModel.weightCountMoeLayer != WEIGHT_COUNT_MOE_LAYER) {
        return false;
    }
    return true;
}

TEST(ernieMoeDecoderModel, DecoderModel)
{
    GlobalMockObject::verify();

    const std::string param =
        "{\"normEps\": 1e-05, \"normType\": 0, \"numAttentionHeadsPerRank\": 8, "
        "\"hiddenSizePerAttentionHead\": 128, \"numHiddenLayers\": 4, "
        "\"numKeyValueHeadsPerRank\": 1, \"useQKNorm\": true, "
        "\"isUnpadInputs\": true, \"isFA\": false, \"isBF16\": true, "
        "\"mlpLinearQuantType\": [[1, -1, 0, -1], [1, -1, 0, -1], [1, -1, 0, -1], [1, -1, 1, -1]], "
        "\"moeLinearQuantType\": [[-1, -1, -1, -1], [-1, -1, -1, -1], [-1, -1, -1, -1], [0, 1, -1, 1]], "
        "\"mlpLinearTransposeType\": [[1, -1, 1, -1], [1, -1, 1, -1], [1, -1, 1, -1], [1, -1, 1, -1]], "
        "\"moeLinearTransposeType\": [[-1, -1, -1, -1], [-1, -1, -1, -1], [-1, -1, -1, -1], [1, 0, -1, 1]], "
        "\"lmHeadTransposeType\": 1, \"enableInitQuant\": true, \"enableSwiGLU\": true, "
        "\"hasSharedExpert\": false, \"rank\": 1, \"expertParallelDegree\": 0, \"numOfExperts\": 64, "
        "\"numOfDeviceExperts\": 64, \"numOfGroups\": 1, \"routingMethod\": \"topkFused\", "
        "\"processLogits\": \"normScaling\", \"firstKDenseReplace\": 3, \"numOfSharedExperts\": 1, "
        "\"numOfSelectedExperts\": [8], \"numOfSelectedGroups\": 1, \"topkGroups\": 1, \"enableFusedTopk\": false, "
        "\"worldSize\": 4, \"rankTableFile\": \"\", \"enableAddNorm\": false, \"normHasBias\": false, "
        "\"hasAttnTp\": true, \"attnTpRank\": 1, \"attnTpSize\": 4, \"attnTpDomain\": \"\", \"hasAttnDp\": false, "
        "\"attnDpRank\": 0, \"attnDpSize\": 1, \"attnDpDomain\": \"2\", \"hasMlpTp\": true, "
        "\"mlpTpRank\": 1, \"mlpTpSize\": 4, \"mlpTpDomain\": \"\", \"isPrefill\": false, \"backend\": \"hccl\"}";
    atb_speed::ernie_moe::DecoderModel decoderModel(param);

    EXPECT_TRUE(CheckErnieWeightCountPerLayer(decoderModel));
}
} // namespace atb_speed