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
#include "models/qwen/layer/moe_decoder_layer.h"
#include "models/qwen/model/moe_decoder_model.h"
#include "nlohmann/json.hpp"
#include "../utils/fuzz_utils.h"
#include "secodeFuzz.h"

namespace atb_speed {

void MoeQwenGetParamType(int numHiddenLayers, nlohmann::json &parameter)
{
    std::vector<int> layerPackQuantType = {
        std::rand() % 12,
        std::rand() % 12
    };
    std::vector<std::vector<int>> modelPackQuantType;
    for (int layerId = 0; layerId < numHiddenLayers; layerId++) {
        modelPackQuantType.push_back(layerPackQuantType);
    }
    parameter["packQuantType"] = modelPackQuantType;

    std::vector<int> LinearType = {
        std::rand() % 3 - 1,
        std::rand() % 3 - 1,
        std::rand() % 3 - 1,
        std::rand() % 3 - 1,
        std::rand() % 3 - 1,
        std::rand() % 3 - 1,
        std::rand() % 3 - 1
    };
    std::vector<std::vector<int>> modelLinearQuantType;
    for (int layerId = 0; layerId < numHiddenLayers; layerId++) {
        modelLinearQuantType.push_back(LinearType);
    }
    parameter["linearQuantType"] = modelLinearQuantType;
    std::vector<std::vector<int>> modelLinearTransposeType;
    for (int layerId = 0; layerId < numHiddenLayers; layerId++) {
        modelLinearTransposeType.push_back(LinearType);
    }
    parameter["linearTransposeType"] = modelLinearTransposeType;
    LinearType.pop_back();
    std::vector<std::vector<int>> contain_six_vect;
    for (int layerId = 0; layerId < numHiddenLayers; layerId++) {
        contain_six_vect.push_back(LinearType);
    }
    parameter["attnLinearQuantType"] = contain_six_vect;
    parameter["attnLinearTransposeType"] = contain_six_vect;
    LinearType.pop_back();
    LinearType.pop_back();
    std::vector<std::vector<int>> contain_four_vect;
    for (int layerId = 0; layerId < numHiddenLayers; layerId++) {
        contain_four_vect.push_back(LinearType);
    }
    parameter["mlpLinearQuantType"] = contain_four_vect;
    parameter["moeLinearQuantType"] = contain_four_vect;
    parameter["mlpLinearTransposeType"] = contain_four_vect;
    parameter["moeLinearTransposeType"] = contain_four_vect;
}

nlohmann::json MoeQwenGetParam(int numHiddenLayers, uint32_t &fuzzIndex)
{
    nlohmann::json parameter;

    parameter["isFA"] = FuzzUtil::GetRandomBool(fuzzIndex);
    parameter["isPrefill"] = FuzzUtil::GetRandomBool(fuzzIndex);
    parameter["isBF16"] = FuzzUtil::GetRandomBool(fuzzIndex);
    parameter["withEmbedding"] = FuzzUtil::GetRandomBool(fuzzIndex);
    parameter["isEmbeddingParallel"] = FuzzUtil::GetRandomBool(fuzzIndex);
    parameter["isLmHeadParallel"] = FuzzUtil::GetRandomBool(fuzzIndex);
    parameter["lmHeadTransposeType"] = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 1, 1, 1);
    parameter["supportSwiGLU"] = FuzzUtil::GetRandomBool(fuzzIndex);
    parameter["rmsNormEps"] = float(std::rand()) / RAND_MAX;
    int numAttentionHeadsPerRank = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 1, 1, 8);
    parameter["numAttentionHeadsPerRank"] = numAttentionHeadsPerRank;
    int hiddenSizeLowerBound = 128;
    int hiddenSizeUpperBound = 1024;
    parameter["hiddenSizePerAttentionHead"] =
            *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++],
                                          hiddenSizeLowerBound,
                                          hiddenSizeLowerBound,
                                          hiddenSizeUpperBound);
    parameter["numKeyValueHeadsPerRank"] = round(numAttentionHeadsPerRank /
            *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 1, 1, numAttentionHeadsPerRank));
    parameter["numHiddenLayers"] = numHiddenLayers;
    parameter["supportLcoc"] = FuzzUtil::GetRandomBool(fuzzIndex);
    int worldSize = numAttentionHeadsPerRank;
    parameter["rank"] = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 0, 0, worldSize);
    parameter["worldSize"] = worldSize;
    parameter["enableLogN"] = FuzzUtil::GetRandomBool(fuzzIndex);
    std::vector<std::string> backendEnumTable = {"lccl", "hccl"};
    parameter["backend"] = backendEnumTable[*(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 0, 0, 1)];
    int numOfExperts = numAttentionHeadsPerRank;
    parameter["numOfExperts"] = numOfExperts;
    parameter["expertParallelDegree"] = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 1, 1, 1);
    parameter["maskStartIdx"] = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 0, 0, 1);
    parameter["numOfSelectedExperts"] = {};
    std::vector<std::string> rankTableFileEnumTable = {"/root/.bashrc", ""};
    parameter["rankTableFile"] = rankTableFileEnumTable[ \
        *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 0, 0, 1)];
    parameter["routingMethod"] = backendEnumTable[*(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 0, 0, 1)];
    parameter["isUnpadInputs"] = FuzzUtil::GetRandomBool(fuzzIndex);
    parameter["normType"] = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 1, 1, 1);
    parameter["isYarn"] = FuzzUtil::GetRandomBool(fuzzIndex);
    parameter["mscale"] = FuzzUtil::GetRandomBool(fuzzIndex);
    parameter["enableQScale"] = FuzzUtil::GetRandomBool(fuzzIndex);
    parameter["enableLora"] = FuzzUtil::GetRandomBool(fuzzIndex);

    MoeQwenGetParamType(numHiddenLayers, parameter);

    return parameter;
}

nlohmann::json MoeQwenGetAclParam(int numHiddenLayers, uint32_t &fuzzIndex)
{
    nlohmann::json aclParameter;

    int layerTokenOffset = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 1, 1, 8);
    std::vector<int> modelTokenOffset;
    for (int layerId = 0; layerId < numHiddenLayers; layerId++) {
        modelTokenOffset.push_back(layerTokenOffset);
    }
    aclParameter["tokenOffset"] = modelTokenOffset;

    int layerSeqLen = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 1, 1, 8);
    std::vector<int> modelSeqLen;
    for (int layerId = 0; layerId < numHiddenLayers; layerId++) {
        modelSeqLen.push_back(layerSeqLen);
    }
    aclParameter["seqLen"] = modelSeqLen;

    return aclParameter;
}

TEST(MoeQwenModelDTFuzz, DecoderModel)
{
    std::srand(time(NULL));
    ATB_SPEED_LOG_DEBUG("begin====================");
    std::string fuzzName = "MoeQwenModelDTFuzzDecoderModel";
    pybind11::module sys = pybind11::module_::import("sys");
    sys.attr("path").attr("append")("tests/fuzztest/core/python");
    pybind11::module modelModule = pybind11::module_::import("base_fuzz");
    DT_FUZZ_START(0, 1000, const_cast<char*>(fuzzName.c_str()), 0) {
        uint32_t fuzzIndex = 0;
        int numHiddenLayers = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 1, 1, 200);

        nlohmann::json parameter, aclParameter;
        parameter = MoeQwenGetParam(numHiddenLayers, fuzzIndex);
        aclParameter = MoeQwenGetAclParam(numHiddenLayers, fuzzIndex);

        std::string parameter_string = parameter.dump();
        std::string acl_parameter_string = aclParameter.dump();

        try {
            pybind11::object llamaFuzz = modelModule.attr("BaseFuzz");
            pybind11::object llamaFuzzIns = llamaFuzz("qwen_MoeDecoderModel");

            pybind11::object ret = llamaFuzzIns.attr("set_param")(parameter_string);
            if (ret.cast<int>() == 0) {
                llamaFuzzIns.attr("set_weight")();
                llamaFuzzIns.attr("set_kv_cache")();
                llamaFuzzIns.attr("execute")(acl_parameter_string);
            }
        } catch (const std::exception& e) {
            std::cerr << e.what() << std::endl;
        }
    }
    DT_FUZZ_END()
    SUCCEED();
}

}
