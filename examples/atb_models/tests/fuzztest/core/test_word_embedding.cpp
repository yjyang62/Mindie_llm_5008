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

#include <gtest/gtest.h>
#include <atb/atb_infer.h>
#include "atb_speed/log.h"
#include "operations/fusion/embedding/word_embedding.h"
#include "../utils/fuzz_utils.h"
#include "secodeFuzz.h"

namespace atb_speed {
TEST(WordEmbeddingDTFuzz, WordEmbedding)
{
    std::srand(time(NULL));
    ATB_SPEED_LOG_DEBUG("begin====================");
    std::string fuzzName = "WordEmbeddingDTFuzzWordEmbedding";

    DT_FUZZ_START(0, 10000, const_cast<char*>(fuzzName.c_str()), 0) {
        uint32_t fuzzIndex = 0;

        int worldSize = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 1, 1, 8);
        std::vector<std::string> backendEnumTable = {"lccl", "hccl"};

        atb_speed::common::WordEmbeddingParam param;
        param.unpadInputs = FuzzUtil::GetRandomBool(fuzzIndex);
        param.tensorParallelInfo.rank = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 0, 0, worldSize);
        param.tensorParallelInfo.worldSize = worldSize;
        param.tensorParallelInfo.backend = backendEnumTable[*(int *) DT_SetGetNumberRange( \
            &g_Element[fuzzIndex++], 0, 0, 1)];

        atb::Node wordEmbeddingNode;
        WordEmbedding(param, &wordEmbeddingNode.operation);
    }
    DT_FUZZ_END()
    SUCCEED();
}
}