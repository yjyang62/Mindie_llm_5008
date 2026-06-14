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

#include "fuzz_utils.h"
#include <climits>
#include <cfloat>
#include "secodeFuzz.h"

namespace atb_speed {
const int ACL_DATA_RANDOM_NUM = 28;
const int ACL_FORMAT_RANDOM_NUM = 36;
const int DIMNUM_RANDOM_NUM = 9;
const int BOOL_RANDOM_NUM = 2;
const int SINGLE_DIM_RANDOM_NUM = 2000;
aclDataType FuzzUtil::GetRandomAclDataType(int input)
{
    return aclDataType(input % ACL_DATA_RANDOM_NUM);
}

aclFormat FuzzUtil::GetRandomAclFormat(int input)
{
    return aclFormat(input % ACL_FORMAT_RANDOM_NUM);
}

uint64_t FuzzUtil::GetRandomDimNum(uint32_t input)
{
    return input % DIMNUM_RANDOM_NUM;
}

bool FuzzUtil::GetRandomBool(uint32_t &fuzzIndex)
{
    u16 randomNum = *(u16 *) DT_SetGetU16(&g_Element[fuzzIndex++], 0);
    return randomNum % BOOL_RANDOM_NUM;
}

void FuzzUtil::GetRandomModelType(std::vector<std::vector<int>> &modelType, int len, int numLayers, int quantType)
{
    if (quantType == 0) {
        return ;
    }
    std::vector<int> layerType;
    for (int i = 0; i < len; ++i) {
        layerType.push_back(std::rand() % quantType - 1);
    }

    for (int layerId = 0; layerId < numLayers; ++layerId) {
        modelType.push_back(layerType);
    }
}
}