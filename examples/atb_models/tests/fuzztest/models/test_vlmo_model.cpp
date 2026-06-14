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
#include <pybind11/embed.h>
#include <pybind11/stl.h>
#include "atb_speed/log.h"
#include "nlohmann/json.hpp"
#include "../utils/fuzz_utils.h"
#include "secodeFuzz.h"

namespace atb_speed {
void vlmoGetFuzzParam(uint32_t &fuzzIndex, nlohmann::json &parameter)
{
    parameter["rmsNormEps"] = (static_cast<double>(std::rand())) / RAND_MAX;
    parameter["headNum"] = *(int *)DT_SetGetNumberRange(&g_Element[fuzzIndex++], 1, 1, 12);  // 12 最大值
    parameter["dk"] = *(int *)DT_SetGetNumberRange(&g_Element[fuzzIndex++], 1, 1, 12);       // 12 最大值
    parameter["layerNum"] = *(int *)DT_SetGetNumberRange(&g_Element[fuzzIndex++], 1, 1, 12); // 12 最大值
    int worldSize = *(int *)DT_SetGetNumberRange(&g_Element[fuzzIndex++], 1, 1, 8);          // 8 最大值
    parameter["rank"] = *(int *)DT_SetGetNumberRange(&g_Element[fuzzIndex++], 0, 0, worldSize);
    parameter["rankSize"] = parameter["rank"];
    std::vector<std::string> backendEnumTable = { "lccl", "hccl" };
    parameter["backend"] = backendEnumTable[*(int *)DT_SetGetNumberRange(&g_Element[fuzzIndex++], 0, 0, 1)];
    parameter["maxTextLen"] = *(int *)DT_SetGetNumberRange(&g_Element[fuzzIndex++], 1, 1, 40);   // 40 最大值
    parameter["vlLayerIndex"] = *(int *)DT_SetGetNumberRange(&g_Element[fuzzIndex++], 1, 1, 12); // 12 最大值
}

void vlmoGetAclFuzzParam(uint32_t &fuzzIndex, nlohmann::json &parameter)
{
    int seqLenSize = *(int *)DT_SetGetNumberRange(&g_Element[fuzzIndex++], 1, 1, 20); // 20 最大值
    std::vector<int> modelSeqLen;
    for (int i = 0; i < seqLenSize; ++i) {
        modelSeqLen.push_back(std::rand() % 100); // 100 is max len of seqlen
    }
    parameter["seqLen"] = modelSeqLen;
    int tokenOffsetSize = *(int *)DT_SetGetNumberRange(&g_Element[fuzzIndex++], 1, 1, 20); // 20 最大值
    std::vector<int> tokenOffsetLen;
    for (int i = 0; i < tokenOffsetSize; ++i) {
        tokenOffsetLen.push_back(std::rand() % 100); // 100 is max len of token offset
    }
    parameter["tokenOffset"] = tokenOffsetLen;
}


TEST(VlmoModelDTFuzz, VlmoModel)
{
    std::srand(time(NULL));
    ATB_SPEED_LOG_DEBUG("begin====================");
    std::string fuzzName = "VlmoModelDTFuzzVlmoModel";

    pybind11::module sys = pybind11::module_::import("sys");
    sys.attr("path").attr("append")("tests/fuzztest/core/python");
    pybind11::module modelModule = pybind11::module_::import("base_fuzz");

    DT_FUZZ_START(0, 1000, const_cast<char *>(fuzzName.c_str()), 0)
    {
        uint32_t fuzzIndex = 0;

        nlohmann::json parameter, aclParameter;
        vlmoGetFuzzParam(fuzzIndex, parameter);
        vlmoGetAclFuzzParam(fuzzIndex, aclParameter);

        std::string parameter_string = parameter.dump();
        std::string acl_parameter_string = aclParameter.dump();

        try {
            pybind11::object vlmoFuzz = modelModule.attr("BaseFuzz");
            pybind11::object vlmoFuzzIns = vlmoFuzz("vlmo_FlashAttentionModel");

            pybind11::object ret = vlmoFuzzIns.attr("set_param")(parameter_string);
            if (ret.cast<int>() == 0) {
                vlmoFuzzIns.attr("set_weight")();
                vlmoFuzzIns.attr("set_kv_cache")();
                vlmoFuzzIns.attr("execute")(acl_parameter_string);
            }
        } catch (const std::exception &e) {
            std::cerr << e.what() << std::endl;
        }
    }
    DT_FUZZ_END()
    SUCCEED();
}
}