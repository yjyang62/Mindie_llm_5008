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
#include "operations/fusion/linear/linear.h"
#include "../utils/fuzz_utils.h"
#include "secodeFuzz.h"

namespace atb_speed {
TEST(LinearDTFuzz, FusionLinear)
{
    std::srand(time(NULL));
    ATB_SPEED_LOG_DEBUG("begin====================");
    std::string fuzzName = "LinearDTFuzzFusionLinear";
    DT_FUZZ_START(0, 10000, const_cast<char*>(fuzzName.c_str()), 0) {
        uint32_t fuzzIndex = 0;
        int LinearQuantTypeEnumTable[] = {0, 1, 2, 3, 4, 5};
        int TransposeTypeEnumTable[] = {-1, 0, 1};
        atb_speed::common::FusionLinearParam param;
        param.quantType = static_cast<atb_speed::common::LinearQuantType>(
            *(unsigned int *) DT_SetGetNumberEnum(
                &g_Element[fuzzIndex++], 0, LinearQuantTypeEnumTable, 6)); // 6 is size of quant type
        param.isBF16 = FuzzUtil::GetRandomBool(fuzzIndex);;
        param.hasBias = FuzzUtil::GetRandomBool(fuzzIndex);;
        param.transposeType = static_cast<atb_speed::common::TransposeType>(
            *(int *) DT_SetGetNumberEnum(
                &g_Element[fuzzIndex++], -1, TransposeTypeEnumTable, 3)); // 3 is size of transpose type

        atb::Node linearNode;
        FusionLinear(param, &linearNode.operation);
    }
    DT_FUZZ_END()
    SUCCEED();
}
}