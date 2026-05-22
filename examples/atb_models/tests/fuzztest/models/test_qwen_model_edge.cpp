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

namespace atb_speed {
void qwenUpdateLayerFuzzParamEdge(int layerNum, nlohmann::json &parameter)
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
        std::rand() % 2,
        std::rand() % 2,
        std::rand() % 2,
        std::rand() % 2,
        std::rand() % 2,
        std::rand() % 2,
        std::rand() % 2
    };
    std::vector<std::vector<int>> modelLinearTransposeType;
    for (int layerId = 0; layerId < layerNum; layerId++) {
        modelLinearTransposeType.push_back(layerLinearTransposeType);
    }
    parameter["linearTransposeType"] = modelLinearTransposeType;
}

void qwenGetFuzzParamEdge(uint32_t &fuzzIndex, nlohmann::json &parameter)
{
    parameter["isFA"] = FuzzUtil::GetRandomBool(fuzzIndex);
    parameter["isBF16"] = FuzzUtil::GetRandomBool(fuzzIndex);
    parameter["withEmbedding"] = FuzzUtil::GetRandomBool(fuzzIndex);
    parameter["isEmbeddingParallel"] = FuzzUtil::GetRandomBool(fuzzIndex);
    parameter["isLmHeadParallel"] = FuzzUtil::GetRandomBool(fuzzIndex);
    parameter["lmHeadTransposeType"] = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 1, 1, 1);
    parameter["supportSwiGLU"] = FuzzUtil::GetRandomBool(fuzzIndex);
    parameter["rmsNormEps"] = float(std::rand()) / RAND_MAX;
    parameter["numAttentionHeadsPerRank"] = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 1, 1, 32);
    parameter["hiddenSizePerAttentionHead"] = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 1, 1, 128);
    int numHiddenLayers = 28;
    parameter["numHiddenLayers"] = numHiddenLayers;
    parameter["numKeyValueHeadsPerRank"] = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 1, 1, 32);
    parameter["rank"] = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 0, 0, 7);
    int worldSize = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 1, 1, 8); // min: 1, max: 8
    parameter["worldSize"] = worldSize;
    std::vector<std::string> backendEnumTable = {"lccl", "hccl"};
    parameter["backend"] = backendEnumTable[*(int *) DT_SetGetNumberRange(
        &g_Element[fuzzIndex++], 0, 0, 1)];
    parameter["kvQuant"] = FuzzUtil::GetRandomBool(fuzzIndex);
    parameter["hiddenSize"] = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 1, 1, 5120);
    parameter["vocabSize"] = 151936;
    parameter["isQuant"] = FuzzUtil::GetRandomBool(fuzzIndex);
    parameter["useQKNorm"] = FuzzUtil::GetRandomBool(fuzzIndex); // True for Qwen3, False for others
    parameter["isPrefill"] = FuzzUtil::GetRandomBool(fuzzIndex);
    parameter["supportLcoc"] = FuzzUtil::GetRandomBool(fuzzIndex);
    parameter["enableFA3"] = FuzzUtil::GetRandomBool(fuzzIndex);
    parameter["enableQScale"] = FuzzUtil::GetRandomBool(fuzzIndex);
    parameter["head_dim"] = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 1, 1, 128);
    parameter["quantGroupSize"] = 0;
    parameter["enableAddNorm"] = FuzzUtil::GetRandomBool(fuzzIndex);
    qwenUpdateLayerFuzzParamEdge(numHiddenLayers, parameter);
}

void qwenGetAclFuzzParamEdge(uint32_t &fuzzIndex, nlohmann::json &parameter)
{
    int seqLenSize = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 1, 1, 20); // min: 1, max: 20
    std::vector<int> modelSeqLen;
    for (int i = 0; i < seqLenSize; ++i) {
        modelSeqLen.push_back(std::rand() % 100); // 100 is max len of seqlen
    }
    parameter["seqLen"] = modelSeqLen;
    int tokenOffsetSize = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 1, 1, 20); // min: 1, max: 20
    std::vector<int> tokenOffsetLen;
    for (int i = 0; i < tokenOffsetSize; ++i) {
        tokenOffsetLen.push_back(std::rand() % 100); // 100 is max len of token offset
    }
    parameter["tokenOffset"] = tokenOffsetLen;
}

TEST(QwenModelDTFuzz, QwenModelEdge)
{
    std::srand(time(NULL));
    std::string fuzzName = "QwenModelDTFuzzModel";

    pybind11::module sys = pybind11::module_::import("sys");
    sys.attr("path").attr("append")("tests/fuzztest/core/python");
    pybind11::module modelModule = pybind11::module_::import("qwen_edge_fuzz");
    DT_FUZZ_START(0, 1000, const_cast<char*>(fuzzName.c_str()), 0) {
        uint32_t fuzzIndex = 0;

        nlohmann::json parameter, aclParameter;
        qwenGetFuzzParamEdge(fuzzIndex, parameter);
        qwenGetAclFuzzParamEdge(fuzzIndex, aclParameter);

        std::string parameter_string = parameter.dump();
        std::string acl_parameter_string = aclParameter.dump();

        try {
            pybind11::object qwenFuzz = modelModule.attr("QwenEdgeFuzz");
            pybind11::object qwenFuzzIns = qwenFuzz("qwen_DecoderModelEdge");
            pybind11::object ret = qwenFuzzIns.attr("set_param")(parameter_string);

            if (ret.cast<int>() == 0) {
                qwenFuzzIns.attr("set_weight")();
                qwenFuzzIns.attr("set_kv_cache")();
                qwenFuzzIns.attr("execute")(acl_parameter_string);
            }
        } catch (const std::exception& e) {
            std::cerr << e.what() << std::endl;
        }
    }
    DT_FUZZ_END()
    SUCCEED();
}
}
