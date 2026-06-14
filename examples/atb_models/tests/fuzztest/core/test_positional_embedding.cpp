/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 * MindIE is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

#include <vector>
#include <gtest/gtest.h>
#include "atb_speed/log.h"
#include "operations/fusion/embedding/positional_embedding.h"
#include "secodeFuzz.h"
#include "../utils/fuzz_utils.h"

namespace atb_speed {
TEST(CommonPositionalEmbeddingFusionDTFuzz, RotaryPositionEmbedding)
{
    std::srand(time(NULL));
    ATB_SPEED_LOG_DEBUG("begin====================");
    std::string fuzzName = "CommonPositionalEmbeddingFusionDTFuzz";
    DT_FUZZ_START(0, 10000, const_cast<char*>(fuzzName.c_str()), 0) {
        uint32_t fuzzIndex = 0;
        int RotaryTypes[] = {0, 1, 2};
        int HeadNums[] = {1, 16, 24, 48};
        int HeadDims[] = {128, 256};
        int KvHeadNums[] = {1, 16, 24, 48};
        atb_speed::common::RotaryPositionEmbeddingParam param;
        param.rotaryType = static_cast<atb_speed::common::RotaryType>(
            *(unsigned int *) DT_SetGetNumberEnum(
                &g_Element[fuzzIndex++], 0, RotaryTypes, 3));
        param.isFA = FuzzUtil::GetRandomBool(fuzzIndex);
        param.headNum = *(int *) DT_SetGetNumberEnum(
            &g_Element[fuzzIndex++], 1, HeadNums, 4);
        param.headDim = *(int *) DT_SetGetNumberEnum(
            &g_Element[fuzzIndex++], 1, HeadDims, 2);
        param.kvHeadNum = *(int *) DT_SetGetNumberEnum(
            &g_Element[fuzzIndex++], 1, KvHeadNums, 4);

        atb::Node PositionalEmbeddingNode;
        RotaryPositionEmbedding(param, &PositionalEmbeddingNode.operation);
    }
    DT_FUZZ_END()
    SUCCEED();
}
}