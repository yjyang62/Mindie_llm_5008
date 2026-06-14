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
#include "models/glm41v/model/decoder_model.h"

namespace atb_speed {

bool CheckGlm41vWeightCountPerLayer(const atb_speed::glm41v::DecoderModel &decoderModel)
{
    constexpr int GLM4_WEIGHT_COUNT_PER_LAYER = 52;
    if (decoderModel.weightCountPerLayer != GLM4_WEIGHT_COUNT_PER_LAYER) {
        return false;
    }
    return true;
}

TEST(Glm41vDecoderModelTest, DecoderModel)
{
    GlobalMockObject::verify();

    const std::string param =
        "{\"enableAddNorm\": false, \"normEps\": 1e-05, \"normType\": 0, "
        "\"numAttentionHeadsPerRank\": 16, \"hiddenSizePerAttentionHead\": 128, \"numHiddenLayers\": 1, "
        "\"numKeyValueHeadsPerRank\": 1, \"isFA\": false, \"isBF16\": true, "
        "\"packQuantType\": [[1, 1]], \"quantGroupSize\": 0, "
        "\"linearQuantType\": [[0, -1, -1, 0, 0, -1, 0]], "
        "\"linearTransposeType\": [[1, -1, -1, 1, 1, -1, 1]], "
        "\"lmHeadTransposeType\": 1, "
        "\"isUnpadInputs\": true, "
        "\"skipWordEmbedding\": false, "
        "\"isLmHeadParallel\": false, "
        "\"enableSwiGLU\": false, "
        "\"rank\": 0, \"worldSize\": 2, \"backend\": \"hccl\", "
        "\"positionEmbeddingType\": 0, \"linearHasBias\": [[true, false, false, false]], "
        "\"isPrefill\": false, \"enableLcoc\": false}";
    atb_speed::glm41v::DecoderModel decoderModel(param);
    EXPECT_TRUE(CheckGlm41vWeightCountPerLayer(decoderModel));
    decoderModel.ConstructInTensorMap();
    MOCKER(atb::CreateOperation<atb::GraphParam>).expects(atLeast(1))
    .with(any(), any()).will(returnValue(0));
    atb::Operation *op = nullptr;
    atb::Status ret = decoderModel.CreateLayerOperation(&op, 0);
    EXPECT_EQ(ret, atb::NO_ERROR);
}

} // namespace atb_speed