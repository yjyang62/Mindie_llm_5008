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
#include <atb/atb_infer.h>
#include "atb_speed/log.h"
#include "nlohmann/json.hpp"
#include "models/deepseekv2/layer/decoder_layer.h"
#include "models/deepseekv2/model/mtp_decoder_model.h"
#include "../utils/fuzz_utils.h"
#include "secodeFuzz.h"

namespace atb_speed {
void SetDeepseekv2MtpIntParam(uint32_t &fuzzIndex, nlohmann::json &parameter, nlohmann::json &aclParameter)
{
    int numAttentionHeadsPerRank = *(int *) DT_SetGetNumberRange(
        &g_Element[fuzzIndex++], 1, 1, 8); // 8 is numattentionheads of per rank
    parameter["numAttentionHeadsPerRank"] = numAttentionHeadsPerRank;
    parameter["hiddenSizePerAttentionHead"] = *(int *) DT_SetGetNumberRange( \
        &g_Element[fuzzIndex++], 128, 128, 1024); // 128, 1024 is hidden size of per head
    parameter["numKeyValueHeadsPerRank"] = round(numAttentionHeadsPerRank / \
        *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 1, 1, numAttentionHeadsPerRank));
    parameter["numOfExperts"] = *(int *) DT_SetGetNumberRange(
        &g_Element[fuzzIndex++], 1, 1, 8); // 8 is num of Experts
    parameter["numOfGroups"] = *(int *) DT_SetGetNumberRange(
        &g_Element[fuzzIndex++], 1, 1, 8); // 8 is num of Groups
    parameter["firstKDenseReplace"] = *(int *) DT_SetGetNumberRange(
        &g_Element[fuzzIndex++], 1, 1, 8); // 8 is num of firstKDenseReplace
    parameter["numOfSharedExperts"] = *(int *) DT_SetGetNumberRange(
        &g_Element[fuzzIndex++], 1, 1, 8); // 8 is num of SharedExperts
    int worldSize = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 8, 1, 8); // 8 is worldSize
    parameter["rank"] = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 7, 0, worldSize-1);
    parameter["expertParallelDegree"] = *(int *) DT_SetGetNumberRange(
        &g_Element[fuzzIndex++], 64/worldSize, 64/worldSize, 64/worldSize); // 64 is worldSize * expertParallelDegree
    parameter["worldSize"] = worldSize;
    parameter["normType"] = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 0, 1, 1);
    parameter["maskStartIdx"] = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 1, 1, 8); // 8 is maskStartIdx
    parameter["qLoraRank"] = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 1, 1, 1536); // 1536 is qLoraRank
    parameter["kvLoraRank"] = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 1, 1, 512); // 512 is kvLoraRank
    parameter["headNum"] = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 8, 1, 128); // 128 is headNum
    parameter["qkNopeHeadDim"] = *(int *) DT_SetGetNumberRange(
        &g_Element[fuzzIndex++], 1, 1, 128); // 128 is qkNopeHeadDim
    parameter["qkRopeHeadDim"] = *(int *) DT_SetGetNumberRange(
        &g_Element[fuzzIndex++], 1, 1, 64); // 64 is qkRopeHeadDim
}

void SetDeepseekv2MtpStringParam(uint32_t &fuzzIndex, nlohmann::json &parameter, nlohmann::json &aclParameter)
{
    std::vector<std::string> routingMethodEnumTable = {"noAuxTc", ""};
    parameter["routingMethod"] = routingMethodEnumTable[ \
        *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 0, 0, 1)];
    std::vector<std::string> processLogitsEnumTable = {"normScaling", ""};
    parameter["processLogits"] = processLogitsEnumTable[ \
        *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 0, 0, 1)];
    std::vector<std::string> domainEnumTable = {"", ""};
    parameter["domain"] = domainEnumTable[ \
        *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 0, 0, 1)];
    parameter["numOfDeviceExperts"] = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 0, 1, 1);
    parameter["nSharedExperts"] = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 0, 1, 1);
    parameter["routedScalingFactor"] = float(std::rand()) / RAND_MAX;
    parameter["quantGroupSize"] = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 0, 1, 1);
    parameter["numOfRedundantExpert"] = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 0, 1, 1);
    parameter["moePackQuantType"] = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 0, 1, 1);
    parameter["mlpNormQuantType"] = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 0, 1, 1);
    parameter["topkGroups"] = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 0, 1, 1);
}

void SetDeepseekv2MtpJsonParam(uint32_t &fuzzIndex, nlohmann::json &parameter, nlohmann::json &aclParameter)
{
    std::vector<std::string> backendEnumTable = {"lccl", "hccl"};
    parameter["backend"] = backendEnumTable[*(int *) DT_SetGetNumberRange( \
        &g_Element[fuzzIndex++], 0, 0, 1)];
    std::vector<std::string> rankTableFileEnumTable = {"", ""};
    parameter["rankTableFile"] = rankTableFileEnumTable[ \
        *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 0, 0, 1)];

    parameter["deviceexpert"] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
     20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46,
      47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71};
    
    std::vector<int> modelrankIds;
    for (int layerId = 0; layerId < 8; layerId++) {
        modelrankIds.push_back(layerId);
    }
    std::vector<int> modelrankId;
    for (int layerId = 0; layerId < 1; layerId++) {
        modelrankId.push_back(layerId);
    }

    using json =nlohmann::json;
    json wordEmbedTp;
    wordEmbedTp["groupId"] = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 0, 0, 1);
    wordEmbedTp["rankIds"] = modelrankIds;
    wordEmbedTp["rank"] = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 0, 1, 1);
    wordEmbedTp["bufferSize"] = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 0, 1, 512);
    wordEmbedTp["backend"] = backendEnumTable[*(int *) DT_SetGetNumberRange( \
        &g_Element[fuzzIndex++], 0, 0, 1)];

    json wordEmbedDp;
    wordEmbedDp["groupId"] = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 0, 0, 1);
    wordEmbedDp["rankIds"] = modelrankId;
    wordEmbedDp["rank"] = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 0, 1, 1);
    wordEmbedDp["bufferSize"] = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 0, 1, 512);
    wordEmbedDp["backend"] = backendEnumTable[*(int *) DT_SetGetNumberRange( \
        &g_Element[fuzzIndex++], 0, 0, 1)];

    json attnTp;
    attnTp["groupId"] = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 0, 0, 1);
    attnTp["rankIds"] = modelrankIds;
    attnTp["rank"] = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 0, 1, 1);
    attnTp["bufferSize"] = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 0, 1, 512);
    attnTp["backend"] = backendEnumTable[*(int *) DT_SetGetNumberRange( \
        &g_Element[fuzzIndex++], 0, 0, 1)];

    json attnDp;
    attnDp["groupId"] = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 0, 0, 1);
    attnDp["rankIds"] = modelrankId;
    attnDp["rank"] = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 0, 1, 1);
    attnDp["bufferSize"] = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 0, 1, 512);
    attnDp["backend"] = backendEnumTable[*(int *) DT_SetGetNumberRange( \
        &g_Element[fuzzIndex++], 0, 0, 1)];

    json attnInnerSp;
    attnInnerSp["groupId"] = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 0, 0, 1);
    attnInnerSp["rankIds"] = modelrankId;
    attnInnerSp["rank"] = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 0, 1, 1);
    attnInnerSp["bufferSize"] = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 0, 1, 512);
    attnInnerSp["backend"] = backendEnumTable[*(int *) DT_SetGetNumberRange( \
        &g_Element[fuzzIndex++], 0, 0, 1)];

    json attnOProjTp;
    attnOProjTp["groupId"] = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 0, 0, 1);
    attnOProjTp["rankIds"] = modelrankId;
    attnOProjTp["rank"] = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 0, 1, 1);
    attnOProjTp["bufferSize"] = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 0, 1, 512);
    attnOProjTp["backend"] = backendEnumTable[*(int *) DT_SetGetNumberRange( \
        &g_Element[fuzzIndex++], 0, 0, 1)];

    json attnOProjDp;
    attnOProjDp["groupId"] = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 0, 0, 1);
    attnOProjDp["rankIds"] = modelrankId;
    attnOProjDp["rank"] = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 0, 1, 1);
    attnOProjDp["bufferSize"] = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 0, 1, 512);
    attnOProjDp["backend"] = backendEnumTable[*(int *) DT_SetGetNumberRange( \
        &g_Element[fuzzIndex++], 0, 0, 1)];

    json mlpTp;
    mlpTp["groupId"] = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 0, 0, 1);
    mlpTp["rankIds"] = modelrankIds;
    mlpTp["rank"] = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 0, 1, 1);
    mlpTp["bufferSize"] = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 0, 1, 512);
    mlpTp["backend"] = backendEnumTable[*(int *) DT_SetGetNumberRange( \
        &g_Element[fuzzIndex++], 0, 0, 1)];

    json mlpDp;
    mlpDp["groupId"] = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 0, 0, 1);
    mlpDp["rankIds"] = modelrankId;
    mlpDp["rank"] = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 0, 1, 1);
    mlpDp["bufferSize"] = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 0, 1, 512);
    mlpDp["backend"] = backendEnumTable[*(int *) DT_SetGetNumberRange( \
        &g_Element[fuzzIndex++], 0, 0, 1)];

    json moeTp;
    moeTp["groupId"] = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 0, 0, 1);
    moeTp["rankIds"] = modelrankIds;
    moeTp["rank"] = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 0, 1, 1);
    moeTp["bufferSize"] = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 0, 1, 512);
    moeTp["backend"] = backendEnumTable[*(int *) DT_SetGetNumberRange( \
        &g_Element[fuzzIndex++], 0, 0, 1)];

    json moeEp;
    moeEp["groupId"] = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 0, 0, 1);
    moeEp["rankIds"] = modelrankId;
    moeEp["rank"] = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 0, 1, 1);
    moeEp["bufferSize"] = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 0, 1, 512);
    moeEp["backend"] = backendEnumTable[*(int *) DT_SetGetNumberRange( \
        &g_Element[fuzzIndex++], 0, 0, 1)];

    json lmHeadTp;
    lmHeadTp["groupId"] = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 0, 0, 1);
    lmHeadTp["rankIds"] = modelrankIds;
    lmHeadTp["rank"] = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 0, 1, 1);
    lmHeadTp["bufferSize"] = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 0, 1, 512);
    lmHeadTp["backend"] = backendEnumTable[*(int *) DT_SetGetNumberRange( \
        &g_Element[fuzzIndex++], 0, 0, 1)];

    json lmHeadDp;
    lmHeadDp["groupId"] = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 0, 0, 1);
    lmHeadDp["rankIds"] = modelrankId;
    lmHeadDp["rank"] = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 0, 1, 1);
    lmHeadDp["bufferSize"] = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 0, 1, 512);
    lmHeadDp["backend"] = backendEnumTable[*(int *) DT_SetGetNumberRange( \
        &g_Element[fuzzIndex++], 0, 0, 1)];


    json mapping;
    mapping["worldSize"] = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 8, 1, 8);
    mapping["rank"] = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 0, 1, 7);
    mapping["rankTableFile"] = rankTableFileEnumTable[ \
        *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 0, 0, 1)];
    mapping["localWorldSize"] = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 8, 1, 8);
    mapping["lcclCommDomainLowerBound"] = 0;
    mapping["lcclCommDomainUpperBound"] = 64;
    mapping["enableQkvdownDp"] = FuzzUtil::GetRandomBool(fuzzIndex);
    mapping["wordEmbedTp"] = wordEmbedTp;
    mapping["wordEmbedDp"] = wordEmbedDp;
    mapping["attnTp"] = attnTp;
    mapping["attnDp"] = attnDp;
    mapping["attnInnerSp"] = attnInnerSp;
    mapping["attnOProjTp"] = attnOProjTp;
    mapping["attnOProjDp"] = attnOProjDp;
    mapping["mlpTp"] = mlpTp;
    mapping["mlpDp"] = mlpDp;
    mapping["moeTp"] = moeTp;
    mapping["moeEp"] = moeEp;
    mapping["lmHeadTp"] = lmHeadTp;
    mapping["lmHeadDp"] = lmHeadDp;
    parameter["mapping"] = mapping;
}

void SetDeepseekv2MtpVectorParam(uint32_t &fuzzIndex, nlohmann::json &parameter, nlohmann::json &aclParameter)
{
    int numHiddenLayers = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 1, 1, 30);
    parameter["numHiddenLayers"] = numHiddenLayers;
    int layerNumOfSelectedExperts = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 1, 1, 8);
    atb::SVector<int32_t> modelNumOfSelectedExperts;
    for (int layerId = 0; layerId < numHiddenLayers; layerId++) {
        modelNumOfSelectedExperts.push_back(layerNumOfSelectedExperts);
    }
    parameter["numOfSelectedExperts"] = modelNumOfSelectedExperts;

    int layertopkGroups = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 1, 1, 8);
    atb::SVector<int32_t> modeltopkGroups;
    for (int layerId = 0; layerId < numHiddenLayers; layerId++) {
        modeltopkGroups.push_back(layertopkGroups);
    }
    parameter["topkGroups"] = modeltopkGroups;
}

void SetDeepseekv2MtpSVectorpackParam(uint32_t &fuzzIndex, nlohmann::json &parameter, nlohmann::json &aclParameter)
{
    int numHiddenLayers = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 1, 1, 30);
    parameter["numHiddenLayers"] = numHiddenLayers;

    std::vector<int> layerPackQuantType = {
        std::rand() % 3 - 1,
        std::rand() % 3 - 1
    };
    std::vector<std::vector<int>> modelPackQuantType;
    for (int layerId = 0; layerId < numHiddenLayers; layerId++) {
        modelPackQuantType.push_back(layerPackQuantType);
    }
    parameter["packQuantType"] = modelPackQuantType;

    std::vector<bool> layerisAntiOutlier = {
        false,
        false
    };
    std::vector<std::vector<bool>> modelisAntiOutlier;
    for (int layerId = 0; layerId < numHiddenLayers; layerId++) {
        modelisAntiOutlier.push_back(layerisAntiOutlier);
    }
    parameter["isAntiOutlier"] = modelisAntiOutlier;
}

void SetDeepseekv2MtpSVectorQuantTypeParam(uint32_t &fuzzIndex, nlohmann::json &parameter, nlohmann::json &aclParameter)
{
    int numHiddenLayers = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 1, 1, 30);
    parameter["numHiddenLayers"] = numHiddenLayers;

    std::vector<int> layerattnLinearQuantType = {
        std::rand() % 3 - 1,
        std::rand() % 3 - 1,
        std::rand() % 3 - 1,
        std::rand() % 3 - 1,
        std::rand() % 3 - 1,
        std::rand() % 3 - 1
    };
    std::vector<std::vector<int>> modelattnLinearQuantType;
    for (int layerId = 0; layerId < numHiddenLayers; layerId++) {
        modelattnLinearQuantType.push_back(layerattnLinearQuantType);
    }
    parameter["attnLinearQuantType"] = modelattnLinearQuantType;

    std::vector<int> layermlpLinearQuantType = {
        std::rand() % 3 - 1,
        std::rand() % 3 - 1,
        std::rand() % 3 - 1,
        std::rand() % 3 - 1
    };
    std::vector<std::vector<int>> modelmlpLinearQuantType;
    for (int layerId = 0; layerId < numHiddenLayers; layerId++) {
        modelmlpLinearQuantType.push_back(layermlpLinearQuantType);
    }
    parameter["mlpLinearQuantType"] = modelmlpLinearQuantType;

    std::vector<int> layermoeLinearQuantType = {
        std::rand() % 3 - 1,
        std::rand() % 3 - 1,
        std::rand() % 3 - 1,
        std::rand() % 3 - 1
    };
    std::vector<std::vector<int>> modelmoeLinearQuantType;
    for (int layerId = 0; layerId < numHiddenLayers; layerId++) {
        modelmoeLinearQuantType.push_back(layermoeLinearQuantType);
    }
    parameter["moeLinearQuantType"] = modelmoeLinearQuantType;
}

void SetDeepseekv2MtpSVectorTransposeTypeParam(
    uint32_t &fuzzIndex, nlohmann::json &parameter, nlohmann::json &aclParameter)
{
    int numHiddenLayers = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 1, 1, 30);
    parameter["numHiddenLayers"] = numHiddenLayers;

    std::vector<int> layerattnLinearTransposeType = {
        std::rand() % 3 - 1,
        std::rand() % 3 - 1,
        std::rand() % 3 - 1,
        std::rand() % 3 - 1,
        std::rand() % 3 - 1,
        std::rand() % 3 - 1
    };
    std::vector<std::vector<int>> modelattnLinearTransposeType;
    for (int layerId = 0; layerId < numHiddenLayers; layerId++) {
        modelattnLinearTransposeType.push_back(layerattnLinearTransposeType);
    }
    parameter["attnLinearTransposeType"] = modelattnLinearTransposeType;

    std::vector<int> layermlpLinearTransposeType = {
        std::rand() % 3 - 1,
        std::rand() % 3 - 1,
        std::rand() % 3 - 1,
        std::rand() % 3 - 1
    };
    std::vector<std::vector<int>> modelmlpLinearTransposeType;
    for (int layerId = 0; layerId < numHiddenLayers; layerId++) {
        modelmlpLinearTransposeType.push_back(layermlpLinearTransposeType);
    }
    parameter["mlpLinearTransposeType"] = modelmlpLinearTransposeType;

    std::vector<int> layermoeLinearTransposeType = {
        std::rand() % 3 - 1,
        std::rand() % 3 - 1,
        std::rand() % 3 - 1,
        std::rand() % 3 - 1
    };
    std::vector<std::vector<int>> modelmoeLinearTransposeType;
    for (int layerId = 0; layerId < numHiddenLayers; layerId++) {
        modelmoeLinearTransposeType.push_back(layermoeLinearTransposeType);
    }
    parameter["moeLinearTransposeType"] = modelmoeLinearTransposeType;
}

TEST(Deepseekv2MtpModelDTFuzz, Deepseekv2MtpModel)
{
    std::srand(time(NULL));
    std::string fuzzName = "Deepseekv2MtpModelDTFuzzDeepseekv2MtpModel";

    pybind11::module sys = pybind11::module_::import("sys");
    sys.attr("path").attr("append")("tests/fuzztest/core/python");
    pybind11::module modelModule = pybind11::module_::import("deepseekv2_fuzz");

    DT_FUZZ_START(0, 10000, const_cast<char*>(fuzzName.c_str()), 0) {
        uint32_t fuzzIndex = 0;

        nlohmann::json parameter, aclParameter;
        parameter["isFA"] = FuzzUtil::GetRandomBool(fuzzIndex);
        parameter["isPrefill"] = FuzzUtil::GetRandomBool(fuzzIndex);
        parameter["isBF16"] = FuzzUtil::GetRandomBool(fuzzIndex);
        parameter["isEmbeddingParallel"] = FuzzUtil::GetRandomBool(fuzzIndex);
        parameter["isLmHeadParallel"] = FuzzUtil::GetRandomBool(fuzzIndex);
        parameter["lmHeadTransposeType"] = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 1, 1, 1);
        parameter["enableSwiGLU"] = FuzzUtil::GetRandomBool(fuzzIndex);
        parameter["supportLcoc"] = FuzzUtil::GetRandomBool(fuzzIndex);
        parameter["normEps"] = float(std::rand()) / RAND_MAX; // 使用DT_SETGETFLOAT会导致null
        parameter["hasSharedExpert"] = FuzzUtil::GetRandomBool(fuzzIndex);
        parameter["hasSharedExpertGate"] = FuzzUtil::GetRandomBool(fuzzIndex);
        parameter["routedScalingFactor"] = float(std::rand()) / RAND_MAX;
        parameter["softmaxScale"] = float(std::rand()) / RAND_MAX;
        parameter["isUnpadInputs"] = FuzzUtil::GetRandomBool(fuzzIndex);
        parameter["qkvHasBias"] = FuzzUtil::GetRandomBool(fuzzIndex);
        parameter["enableSwiGLUQuantForSharedExperts"] = FuzzUtil::GetRandomBool(fuzzIndex);
        parameter["enableGMMSwigluQuant"] = FuzzUtil::GetRandomBool(fuzzIndex);
        parameter["enableInitQuant"] = FuzzUtil::GetRandomBool(fuzzIndex);
        parameter["finalStateOut"] = FuzzUtil::GetRandomBool(fuzzIndex);
        parameter["enableFusedTopk"] = FuzzUtil::GetRandomBool(fuzzIndex);
        parameter["enableSwigluQuant"] = FuzzUtil::GetRandomBool(fuzzIndex);
        parameter["enableMlaPreprocess"] = FuzzUtil::GetRandomBool(fuzzIndex);
        parameter["enableGatherPreNorm"] = FuzzUtil::GetRandomBool(fuzzIndex);
        parameter["attnOprojPrefetch"] = FuzzUtil::GetRandomBool(fuzzIndex);
        parameter["isNzCache"] = FuzzUtil::GetRandomBool(fuzzIndex);
        parameter["enableExtraOprojTp"] = FuzzUtil::GetRandomBool(fuzzIndex);
        parameter["hasP2DWeight"] = FuzzUtil::GetRandomBool(fuzzIndex);
        parameter["enableAllToAllMC2"] = FuzzUtil::GetRandomBool(fuzzIndex);
        parameter["enableLoadBalance"] = FuzzUtil::GetRandomBool(fuzzIndex);
        parameter["enableEPWB"] = FuzzUtil::GetRandomBool(fuzzIndex);
        parameter["enableExpertCumSumOutput"] = FuzzUtil::GetRandomBool(fuzzIndex);
        parameter["isMlpFullTP"] = FuzzUtil::GetRandomBool(fuzzIndex);
        parameter["maskfree"] = FuzzUtil::GetRandomBool(fuzzIndex);
        parameter["enableQkvdownDp"] = FuzzUtil::GetRandomBool(fuzzIndex);
        parameter["enableSharedExpertDp"] = FuzzUtil::GetRandomBool(fuzzIndex);
        parameter["enableGatingDp"] = FuzzUtil::GetRandomBool(fuzzIndex);
        parameter["enableInfNan"] = FuzzUtil::GetRandomBool(fuzzIndex);
        parameter["enableCVOverlap"] = FuzzUtil::GetRandomBool(fuzzIndex);
        parameter["enableDpOut"] = FuzzUtil::GetRandomBool(fuzzIndex);
        parameter["lmHeadLocalTp"] = FuzzUtil::GetRandomBool(fuzzIndex);
        parameter["hasBias"] = FuzzUtil::GetRandomBool(fuzzIndex);
        parameter["enableGreedyPostProcessing "] = FuzzUtil::GetRandomBool(fuzzIndex);

        SetDeepseekv2MtpIntParam(fuzzIndex, parameter, aclParameter);
        SetDeepseekv2MtpStringParam(fuzzIndex, parameter, aclParameter);
        SetDeepseekv2MtpJsonParam(fuzzIndex, parameter, aclParameter);
        SetDeepseekv2MtpVectorParam(fuzzIndex, parameter, aclParameter);
        SetDeepseekv2MtpSVectorpackParam(fuzzIndex, parameter, aclParameter);
        SetDeepseekv2MtpSVectorQuantTypeParam(fuzzIndex, parameter, aclParameter);
        SetDeepseekv2MtpSVectorTransposeTypeParam(fuzzIndex, parameter, aclParameter);

        std::string parameter_string = parameter.dump();
        std::string acl_parameter_string = aclParameter.dump();

        try {
            pybind11::object deepseekv2MtpFuzz = modelModule.attr("BaseFuzz");
            pybind11::object deepseekv2MtpFuzzIns = deepseekv2MtpFuzz("deepseekV2_MtpDecoderModel");
            
            pybind11::object ret = deepseekv2MtpFuzzIns.attr("set_param")(parameter_string);
            if (ret.cast<int>() == 0) {
                deepseekv2MtpFuzzIns.attr("set_weight")();
                deepseekv2MtpFuzzIns.attr("set_kv_cache")();
                deepseekv2MtpFuzzIns.attr("execute")(acl_parameter_string);
            }
        } catch (const std::exception& e) {
            std::cerr << e.what() << std::endl;
        }
    }
    DT_FUZZ_END()
    SUCCEED();
}
}