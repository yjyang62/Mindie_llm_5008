/**
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
#include <pybind11/embed.h>
#include <pybind11/stl.h>
#include "atb_speed/log.h"
#include "nlohmann/json.hpp"
#include "../utils/fuzz_utils.h"
#include "secodeFuzz.h"
#include <random>

namespace atb_speed {
unsigned int GetRandomNumber()
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<unsigned int> dis(0, 65535);  // 0 ~ 65535
    return dis(gen);
}
void minicpmUpdateLayerFuzzParam(int layerNum, nlohmann::json &parameter)
{
    std::vector<int> layerPackQuantType = {
        std::rand() % 12,
        std::rand() % 12
    };
    std::vector<std::vector<int>> modelPackQuantType;
    for (int layerId = 0; layerId < layerNum; layerId++) {
        modelPackQuantType.push_back(layerPackQuantType);
    }
    parameter["packQuantType"] = modelPackQuantType;

    std::vector<int> layerLinearQuantType = {
        std::rand() % 3 - 1,
        std::rand() % 3 - 1,
        std::rand() % 3 - 1,
        std::rand() % 3 - 1,
        std::rand() % 3 - 1,
        std::rand() % 3 - 1,
        std::rand() % 3 - 1
    };
    std::vector<std::vector<int>> modelLinearQuantType;
    for (int layerId = 0; layerId < layerNum; layerId++) {
        modelLinearQuantType.push_back(layerLinearQuantType);
    }
    parameter["linearQuantType"] = modelLinearQuantType;

    std::vector<int> layerLinearTransposeType = {
        std::rand() % 3 - 1,
        std::rand() % 3 - 1,
        std::rand() % 3 - 1,
        std::rand() % 3 - 1,
        std::rand() % 3 - 1,
        std::rand() % 3 - 1,
        std::rand() % 3 - 1
    };
    std::vector<std::vector<int>> modelLinearTransposeType;
    for (int layerId = 0; layerId < layerNum; layerId++) {
        modelLinearTransposeType.push_back(layerLinearTransposeType);
    }
    parameter["linearTransposeType"] = modelLinearTransposeType;
}

void minicpmGetFuzzParam(uint32_t &fuzzIndex, nlohmann::json &parameter)
{
    parameter["rmsNormEps"] = float(std::rand()) / RAND_MAX; // 使用DT_SETGETFLOAT会导致null
    parameter["scaleEmb"] = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 1, 1, 18); // min: 1, max: 18
    parameter["scaleDepth"] = float(std::rand()) / RAND_MAX;
    parameter["dimModelBase"] = *(int *) DT_SetGetNumberRange( \
        &g_Element[fuzzIndex++], 128, 128, 512); // min: 128, max: 512
    int numHiddenLayers = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 1, 1, 200); // min: 1, max: 200
    parameter["numHiddenLayers"] = numHiddenLayers;
    parameter["hiddenSize"] = *(int *) DT_SetGetNumberRange( \
        &g_Element[fuzzIndex++], 128, 128, 8106); // min: 128, max: 8106
    int numAttentionHeads = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 1, 1, 8); // min: 1, max: 8
    parameter["numAttentionHeads"] = numAttentionHeads;
    parameter["numKeyValueHeads"] = round(numAttentionHeads / \
        *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 1, 1, numAttentionHeads));
    int vocabSize = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 1, 1, 8); // min: 1, max: 8
    parameter["vocabSize"] =vocabSize;
    parameter["isGQA"] = FuzzUtil::GetRandomBool(fuzzIndex);
    parameter["isQuant"] = FuzzUtil::GetRandomBool(fuzzIndex);
    parameter["isPrefill"] = FuzzUtil::GetRandomBool(fuzzIndex);
    parameter["enableSpeculate"] = FuzzUtil::GetRandomBool(fuzzIndex);
    minicpmUpdateLayerFuzzParam(numHiddenLayers, parameter);
}

void minicpmGetAclFuzzParam(uint32_t &fuzzIndex, nlohmann::json &parameter, bool supportSpeculate)
{
    parameter["seqLength"] = FuzzUtil::GetRandomBool(fuzzIndex);
}


TEST(MiniCPMModelDTFuzz, MiniCPMModel)
{
    std::srand(time(NULL));
    ATB_SPEED_LOG_DEBUG("begin====================");
    std::string fuzzName = "MiniCPMModelDTFuzzMiniCPMModel";
    pybind11::module sys = pybind11::module_::import("sys");
    sys.attr("path").attr("append")("tests/fuzztest/core/python");
    pybind11::module modelModule = pybind11::module_::import("minicpm_edge_fuzz");
    DT_FUZZ_START(0, 1000, const_cast<char*>(fuzzName.c_str()), 0) {
        uint32_t fuzzIndex = 0;
        nlohmann::json parameter, aclParameter;
        minicpmGetFuzzParam(fuzzIndex, parameter);
        minicpmGetAclFuzzParam(fuzzIndex, aclParameter, parameter["enableSpeculate"]);
        std::string parameter_string = GetRandomNumber() % 10 ? parameter.dump(): "None";  // "None" (1/10)
        std::string acl_parameter_string = GetRandomNumber() % 20 ? aclParameter.dump() : "None";  // "None" (1/20)

        try {
            pybind11::object minicpmFuzz = modelModule.attr("BaseFuzz");
            pybind11::object minicpmFuzzIns = minicpmFuzz("minicpm_DecoderModelEdge");
            pybind11::object ret = minicpmFuzzIns.attr("set_param")(parameter_string);
            if (ret.cast<int>() == 0) {
                minicpmFuzzIns.attr("set_weight")();
                minicpmFuzzIns.attr("set_kv_cache")();
                minicpmFuzzIns.attr("execute")(acl_parameter_string, int(parameter["enableSpeculate"]));
            }
        } catch (const std::exception& e) {
            std::cerr << e.what() << std::endl;
        }
    }
    DT_FUZZ_END()
    SUCCEED();
}
}
