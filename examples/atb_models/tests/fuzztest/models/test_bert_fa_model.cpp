/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
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
#include <pybind11/embed.h>
#include <pybind11/stl.h>

#include "atb_speed/log.h"
#include "nlohmann/json.hpp"
#include "../utils/fuzz_utils.h"
#include "secodeFuzz.h"

namespace atb_speed {
void bertGetFuzzParam(uint32_t &fuzzIndex, nlohmann::json &parameter)
{
    int dk = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 64, 64, 128);
    parameter["dk"] = dk;
    std::vector<int> geluApproximate = {-1, 0, 1};
    parameter["geluApproximate"] = geluApproximate[*(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 0, 1, 2)];
    int headNum = 16;
    parameter["headNum"] = headNum;
    float layerNormEps = float(std::rand()) / RAND_MAX;
    parameter["layerNormEps"] = layerNormEps;
    std::vector<int> layerNormImplMode = {0, 1, 2};
    parameter["layerNormImplMode"] = layerNormImplMode[*(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 0, 1, 2)];
    int layerNum = 2;
    parameter["layerNum"] = layerNum;
    int rankSize = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 1, 1, 1);
    parameter["rank"] = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 0, 0, rankSize);
    parameter["rankSize"] = rankSize;
    parameter["enableFasterGelu"] = FuzzUtil::GetRandomBool(fuzzIndex);
    parameter["enableAclNNAttn"] = FuzzUtil::GetRandomBool(fuzzIndex);
    parameter["enableAclNNMatmul"] = FuzzUtil::GetRandomBool(fuzzIndex);
}

void bertGetAclParam(nlohmann::json &aclParameter)
{
    aclParameter["tokenOffset"] = {512, 512};
    aclParameter["seqLen"] = {512, 512};
}

TEST(BertModelDTFuzz, FlashAttentionModel)
{
    std::srand(time(nullptr));
    ATB_SPEED_LOG_DEBUG("begin====================");
    std::string fuzzName = "BertModelDTFuzzFlashAttentionModel";
    pybind11::module sys = pybind11::module_::import("sys");
    sys.attr("path").attr("append")("tests/fuzztest/core/python");
    pybind11::module modelModule = pybind11::module_::import("bert_fuzz");

    DT_FUZZ_START(0, 1000, const_cast<char*>(fuzzName.c_str()), 0) {
        uint32_t fuzzIndex = 0;

        nlohmann::json parameter, aclParameter;
        bertGetFuzzParam(fuzzIndex, parameter);
        bertGetAclParam(aclParameter);

        std::string parameter_string;
        std::string acl_parameter_string;
        if (*(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 1, 0, 1)) {
            parameter_string = parameter.dump();
            acl_parameter_string = aclParameter.dump();
        } else {
            parameter_string = "fake_parameter_string";
            acl_parameter_string = "fake_acl_parameter_string";
        }

        try{
            pybind11::object bertFuzz = modelModule.attr("BertFuzz");
            pybind11::object bertFuzzIns = bertFuzz("bert_EncoderModel", int(parameter["layerNum"]));

            pybind11::object ret = bertFuzzIns.attr("set_param")(parameter_string);
            if (ret.cast<int>() == 0) {
                bertFuzzIns.attr("set_weight")();
                bertFuzzIns.attr("execute_fa")(acl_parameter_string, bool(parameter["enableAclNNAttn"]));
            }
        } catch (const std::exception& e) {
            std::cerr << e.what() << std::endl;
        }
    }
    DT_FUZZ_END()
    SUCCEED();
}
}
