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

#include "operations/fusion/utils.h"
#include <gtest/gtest.h>
#include <mockcpp/mockcpp.hpp>
#include "atb_speed/utils/operation_util.h"
#include "atb_speed/base/event_manager.h"
#include "atb_speed/utils/singleton.h"
#include "atb_speed/base/model.h"
#include <unordered_map>
#include <vector>
#include <string>

using namespace atb_speed;
using namespace atb_speed::common;
using namespace atb;

namespace atb_speed {
namespace common {

class DapUtilsTest : public testing::Test {
protected:
    void SetUp() override
    {
        GetSingleton<DapManager>().SetRole(DapRole::UNDEFINED_ROLE);
        GetSingleton<CommOpCounter>().Reset();
    }

    void TearDown() override
    {
        GetSingleton<DapManager>().SetRole(DapRole::UNDEFINED_ROLE);
        GetSingleton<CommOpCounter>().Reset();
    }
};

// ------------------------------ DapManager  ------------------------------
TEST_F(DapUtilsTest, DapManager_SetAndGetRole)
{
    GlobalMockObject::verify(); // 清空mock
    DapManager& manager = GetSingleton<DapManager>();
    
    manager.SetRole(DapRole::PRECEDER);
    EXPECT_EQ(manager.GetRole(), DapRole::PRECEDER);
    
    manager.SetRole(DapRole::SUCCESSOR);
    EXPECT_EQ(manager.GetRole(), DapRole::SUCCESSOR);
    
    manager.SetRole(DapRole::UNDEFINED_ROLE);
    EXPECT_EQ(manager.GetRole(), DapRole::UNDEFINED_ROLE);
}

TEST_F(DapUtilsTest, DapManager_GetSuccessorSuffix)
{
    GlobalMockObject::verify(); // 清空mock
    DapManager& manager = GetSingleton<DapManager>();
    EXPECT_EQ(manager.GetSuccessorSuffix(), "_successor");
}

TEST_F(DapUtilsTest, DapManager_GetStreamId)
{
    GlobalMockObject::verify(); // 清空mock
    DapManager& manager = GetSingleton<DapManager>();
    
    manager.SetRole(DapRole::PRECEDER);
    EXPECT_EQ(manager.GetStreamId(), 0u);
    
    manager.SetRole(DapRole::SUCCESSOR);
    EXPECT_EQ(manager.GetStreamId(), 1u);
    
    manager.SetRole(DapRole::UNDEFINED_ROLE);
    EXPECT_EQ(manager.GetStreamId(), 0u);
}

// ------------------------------ CommOpCounter  ------------------------------
TEST_F(DapUtilsTest, CommOpCounter_IncrementAndGetCount)
{
    GlobalMockObject::verify(); // 清空mock
    CommOpCounter& counter = GetSingleton<CommOpCounter>();
    DapManager& manager = GetSingleton<DapManager>();
    
    manager.SetRole(DapRole::PRECEDER);
    EXPECT_EQ(counter.GetCount(), 0);
    EXPECT_EQ(counter.Increment(), 1);
    EXPECT_EQ(counter.GetCount(), 1);
    EXPECT_EQ(counter.Increment(), 2);
    EXPECT_EQ(counter.GetCount(), 2);
    
    manager.SetRole(DapRole::SUCCESSOR);
    EXPECT_EQ(counter.GetCount(), 0);
    EXPECT_EQ(counter.Increment(), 1);
    EXPECT_EQ(counter.GetCount(), 1);
    
    manager.SetRole(DapRole::PRECEDER);
    EXPECT_EQ(counter.GetCount(), 2);
}

TEST_F(DapUtilsTest, CommOpCounter_Reset)
{
    GlobalMockObject::verify(); // 清空mock
    CommOpCounter& counter = GetSingleton<CommOpCounter>();
    DapManager& manager = GetSingleton<DapManager>();
    
    manager.SetRole(DapRole::PRECEDER);
    counter.Increment();
    counter.Increment();
    
    manager.SetRole(DapRole::SUCCESSOR);
    counter.Increment();
    
    counter.Reset();
    
    manager.SetRole(DapRole::PRECEDER);
    EXPECT_EQ(counter.GetCount(), 0);
    
    manager.SetRole(DapRole::SUCCESSOR);
    EXPECT_EQ(counter.GetCount(), 0);
}

// ------------------------------ AssignTensorIdx  ------------------------------
TEST_F(DapUtilsTest, AssignTensorIdx_WithStartIdx)
{
    GlobalMockObject::verify(); // 清空mock
    std::map<std::string, std::vector<std::string>> tensorCandidates = {
        {"input", {"tensor1", "tensor2"}},
        {"output", {"tensor3"}}
    };
    std::map<std::string, uint32_t> tensorMap;
    uint32_t tensorIdx = 10;
    
    AssignTensorIdx(tensorCandidates, "input", tensorIdx, tensorMap);
    EXPECT_EQ(tensorMap["tensor1"], 10u);
    EXPECT_EQ(tensorMap["tensor2"], 11u);
    EXPECT_EQ(tensorIdx, 12u);
    
    AssignTensorIdx(tensorCandidates, "invalid_key", tensorIdx, tensorMap);
    EXPECT_EQ(tensorMap.size(), 2u);
    EXPECT_EQ(tensorIdx, 12u);
}

TEST_F(DapUtilsTest, AssignTensorIdx_WithoutStartIdx)
{
    GlobalMockObject::verify(); // 清空mock
    std::map<std::string, std::vector<std::string>> tensorCandidates = {
        {"input", {"tensor1", "tensor2"}},
        {"output", {"tensor3"}}
    };
    std::map<std::string, uint32_t> tensorMap = {{"init", 0}};
    
    AssignTensorIdx(tensorCandidates, "output", tensorMap);
    EXPECT_EQ(tensorMap["tensor3"], 1u);
    
    AssignTensorIdx(tensorCandidates, "invalid_key", tensorMap);
    EXPECT_EQ(tensorMap.size(), 2u);
}

// ------------------------------ AddTensorToList  ------------------------------
TEST_F(DapUtilsTest, AddTensorToList_VectorString)
{
    GlobalMockObject::verify(); // 清空mock
    std::map<std::string, std::vector<std::string>> tensorCandidates = {
        {"input", {"tensor1", "tensor2"}},
        {"output", {"tensor3"}}
    };
    std::vector<std::string> tensorList = {"init"};
    
    AddTensorToList(tensorCandidates, "input", tensorList);
    EXPECT_EQ(tensorList, std::vector<std::string>({"init", "tensor1", "tensor2"}));
    
    AddTensorToList(tensorCandidates, "invalid_key", tensorList);
    EXPECT_EQ(tensorList.size(), 3u);
}

TEST_F(DapUtilsTest, AddTensorToList_SVectorString)
{
    GlobalMockObject::verify(); // 清空mock
    std::map<std::string, SVector<std::string>> tensorCandidates = {
        {"input", {"tensor1", "tensor2"}},
        {"output", {"tensor3"}}
    };
    SVector<std::string> tensorList = {"init"};
    
    AddTensorToList(tensorCandidates, "output", tensorList);
    EXPECT_EQ(tensorList.size(), 2u);
    EXPECT_EQ(tensorList[1], "tensor3");
}

// ------------------------------ GetTensorMap  ------------------------------
TEST_F(DapUtilsTest, GetTensorMap)
{
    GlobalMockObject::verify(); // 清空mock
    std::vector<std::string> inTensorList = {"in1", "in2"};
    std::vector<std::string> outTensorList = {"out1"};
    std::vector<std::string> intermediateTensorList = {"mid1", "mid2"};
    
    auto tensorMap = GetTensorMap(inTensorList, outTensorList, intermediateTensorList);
    
    EXPECT_EQ(tensorMap["in1"], 0u);
    EXPECT_EQ(tensorMap["in2"], 1u);
    EXPECT_EQ(tensorMap["out1"], 2u);
    EXPECT_EQ(tensorMap["mid1"], 3u);
    EXPECT_EQ(tensorMap["mid2"], 4u);
    EXPECT_EQ(tensorMap.size(), 5u);
}

// ------------------------------ GetTensorIdx  ------------------------------
TEST_F(DapUtilsTest, GetTensorIdx_ExistAndNotExist)
{
    GlobalMockObject::verify(); // 清空mock
    std::map<std::string, uint32_t> tensorMap = {{"tensor1", 0}, {"tensor2", 1}};
    
    EXPECT_EQ(GetTensorIdx(tensorMap, "tensor1"), 0u);
    
    EXPECT_EQ(GetTensorIdx(tensorMap, "invalid_tensor"), UINT32_MAX);
}

// ------------------------------ GetTensorIdxList  ------------------------------
TEST_F(DapUtilsTest, GetTensorIdxList)
{
    GlobalMockObject::verify(); // 清空mock
    std::map<std::string, uint32_t> tensorMap = {{"tensor1", 0}, {"tensor2", 1}, {"tensor3", 2}};
    std::vector<std::string> tensorNames = {"tensor2", "tensor1", "invalid", "tensor3"};
    
    auto idxList = GetTensorIdxList(tensorMap, tensorNames);
    
    EXPECT_EQ(idxList.size(), 4u);
    EXPECT_EQ(idxList[0], 1u);
    EXPECT_EQ(idxList[1], 0u);
    EXPECT_EQ(idxList[2], UINT32_MAX);
    EXPECT_EQ(idxList[3], 2u);

// ------------------------------ CheckAntiOutlier  ------------------------------
    EXPECT_TRUE(CheckAntiOutlier(ALL_W8A8SC_ANTI));
    EXPECT_TRUE(CheckAntiOutlier(ALL_W4A16_ANTI));
    
    EXPECT_FALSE(CheckAntiOutlier(ALL_FP));
    EXPECT_FALSE(CheckAntiOutlier(ALL_W8A8));
    EXPECT_FALSE(CheckAntiOutlier(PACK_QUANT_UNDEFINED));
}

// ------------------------------ CheckPack  ------------------------------
TEST_F(DapUtilsTest, CheckPack_PackableQuantType)
{
    GlobalMockObject::verify(); // 清空mock
    std::vector<int> linearDescs = {1, 1, 1};
    std::vector<int> linearIndex = {0, 1, 2};
    
    EXPECT_TRUE(CheckPack(ALL_FP, linearDescs, linearIndex));
    EXPECT_TRUE(CheckPack(ALL_W8A16, linearDescs, linearIndex));
    EXPECT_TRUE(CheckPack(ALL_W4A8_ANTI, linearDescs, linearIndex));
    
    EXPECT_FALSE(CheckPack(MIX_W8A8, linearDescs, linearIndex));
    EXPECT_FALSE(CheckPack(MIX_W4A16, linearDescs, linearIndex));
}

TEST_F(DapUtilsTest, CheckPack_UndefinedQuantType_SameDescs)
{
    GlobalMockObject::verify(); // 清空mock
    std::vector<int> linearDescs = {2, 2, 2};
    std::vector<int> linearIndex = {0, 1, 2};
    
    EXPECT_TRUE(CheckPack(PACK_QUANT_UNDEFINED, linearDescs, linearIndex));
}

TEST_F(DapUtilsTest, CheckPack_UndefinedQuantType_DifferentDescs)
{
    GlobalMockObject::verify(); // 清空mock
    std::vector<int> linearDescs = {2, 3, 2};
    std::vector<int> linearIndex = {0, 1, 2};
    
    EXPECT_FALSE(CheckPack(PACK_QUANT_UNDEFINED, linearDescs, linearIndex));
}

TEST_F(DapUtilsTest, CheckPack_InvalidLinearIndex)
{
    GlobalMockObject::verify(); // 清空mock
    std::vector<int> linearDescs = {2, 2};
    std::vector<int> linearIndex = {0, 5};
    
    EXPECT_TRUE(CheckPack(PACK_QUANT_UNDEFINED, linearDescs, linearIndex));
}

// ------------------------------ CheckParamVectorSize  ------------------------------
TEST_F(DapUtilsTest, CheckParamVectorSize)
{
    GlobalMockObject::verify(); // 清空mock
    std::vector<int> vec1 = {1, 2, 3};
    std::vector<int> vec2 = {1};
    
    EXPECT_EQ(CheckParamVectorSize(vec1, 3), NO_ERROR);
    EXPECT_EQ(CheckParamVectorSize(vec1, 2), NO_ERROR);
    
    EXPECT_EQ(CheckParamVectorSize(vec2, 2), ERROR_INVALID_PARAM);
}

// ------------------------------ ConvertQuantTypeToPackType  ------------------------------
TEST_F(DapUtilsTest, ConvertQuantTypeToPackType)
{
    GlobalMockObject::verify(); // 清空mock
    EXPECT_EQ(ConvertQuantTypeToPackType("float"), ALL_FP);
    EXPECT_EQ(ConvertQuantTypeToPackType("w8a8"), ALL_W8A8);
    EXPECT_EQ(ConvertQuantTypeToPackType("w8a8sc"), ALL_W8A8SC);
    EXPECT_EQ(ConvertQuantTypeToPackType("w8a8_dynamic"), ALL_W8A8_DYNAMIC);
    EXPECT_EQ(ConvertQuantTypeToPackType(""), PACK_QUANT_UNDEFINED);
    
    EXPECT_EQ(ConvertQuantTypeToPackType("invalid_type"), PACK_QUANT_UNDEFINED);
    EXPECT_EQ(ConvertQuantTypeToPackType("w4a4"), PACK_QUANT_UNDEFINED);
}

// ------------------------------ AddDapEventsAfterComm （Mock EventManager） ------------------------------
TEST_F(DapUtilsTest, AddDapEventsAfterComm_Successor)
{
    GlobalMockObject::verify(); // 清空mock
    DapManager& manager = GetSingleton<DapManager>();
    manager.SetRole(DapRole::SUCCESSOR);
    CommOpCounter& counter = GetSingleton<CommOpCounter>();
    counter.Reset();
    
    Model::Graph opGraph;
    EXPECT_EQ(AddDapEventsAfterComm(opGraph), NO_ERROR);
    EXPECT_EQ(opGraph.nodes.size(), 2u); // RecordEvent + WaitEvent
    EXPECT_EQ(counter.GetCount(), 1);
}

} // namespace common
} // namespace atb_speed