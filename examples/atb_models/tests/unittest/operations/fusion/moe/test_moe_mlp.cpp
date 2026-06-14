/*
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
#include <mockcpp/mockcpp.hpp>
#include <atb/atb_infer.h>
#include "operations/aclnn/utils/utils.h"
#include "operations/fusion/utils.h"
#include "operations/fusion/moe/moe_mlp.h"
#include "operations/fusion/moe/integrated_gmm.h"
#include "operations/fusion/moe/ep/expert_filter.h"

namespace atb_speed {
namespace common {

TEST(moeMlpTest, CreateMoeMlpOperationDistribute)
{
    GlobalMockObject::verify(); // 清空mock
    MoeMlpParam param;
    atb::Operation *op = nullptr;
    param.hasMoeEp = true;
    param.enableMoeDistribute = true;
    param.enableDispatchCombineV2 = true;
    param.moeLinearQuantType = {false, false, false, false};
    param.supportSwiGLU = true;
    MOCKER(atb_speed::common::GetTensorIdx).expects(atLeast(1)).with(any(), any()).will(returnValue(0));
    MOCKER(atb_speed::common::AddTensorToList<std::vector<std::string>>).expects(atLeast(1)).with(any(), any());
    MOCKER(atb::CreateOperation<atb::GraphParam>).expects(atLeast(1)).with(any(), any()).will(returnValue(0));
    MOCKER(atb_speed::common::AddDapEventsBeforeComm<atb::GraphParam>).expects(atLeast(1)).with(any()).will(returnValue(0));
    MOCKER(atb_speed::common::AddDapEventsAfterComm<atb::GraphParam>).expects(atLeast(1)).with(any()).will(returnValue(0));
    MOCKER(atb_speed::common::CreateIntegratedGmmOperation).expects(atLeast(1)).with(any(), any()).will(returnValue(0));
    EXPECT_EQ(atb_speed::common::CreateMoeMlpOperation(param, &op), 0);
    param.enableDispatchCombineV2 = false;
    EXPECT_EQ(atb_speed::common::CreateMoeMlpOperation(param, &op), 0);
    param.supportSwiGLU = false;
    EXPECT_EQ(atb_speed::common::CreateMoeMlpOperation(param, &op), 0);
}

TEST(moeMlpTest, CreateMoeMlpOperationFusedRouting)
{
    GlobalMockObject::verify(); // 清空mock
    MoeMlpParam param;
    atb::Operation *op = nullptr;
    param.hasMoeEp = false;
    param.enableMoeDistribute = false;
    param.enableDispatchCombineV2 = false;
    param.moeLinearQuantType = {false, false, false, false};
    param.supportSwiGLU = true;
    param.enableFusedRouting = true;
    MOCKER(atb_speed::common::GetTensorIdx).expects(atLeast(1)).with(any(), any()).will(returnValue(0));
    MOCKER(atb_speed::common::AddTensorToList<std::vector<std::string>>).expects(atLeast(1)).with(any(), any());
    MOCKER(atb::CreateOperation<atb::GraphParam>).expects(atLeast(1)).with(any(), any()).will(returnValue(0));
    MOCKER(atb_speed::common::CreateIntegratedGmmOperation).expects(atLeast(1)).with(any(), any()).will(returnValue(0));
    EXPECT_EQ(atb_speed::common::CreateMoeMlpOperation(param, &op), 0);
    param.hasMoeEp = true;
    MOCKER(atb_speed::common::CreateExpertFilterOperation).expects(atLeast(1)).with(any(), any()).will(returnValue(0));
    EXPECT_EQ(atb_speed::common::CreateMoeMlpOperation(param, &op), 0);
    param.supportSwiGLU = false;
    EXPECT_EQ(atb_speed::common::CreateMoeMlpOperation(param, &op), 0);
    param.enableInitRoutingCutoff = true;
    param.enableInitQuant = true;
    param.packQuantType = atb_speed::common::PackQuantType::ALL_W8A8_DYNAMIC;
    MOCKER(atb_speed::common::Is310P).expects(atLeast(1)).will(returnValue(true));
    EXPECT_EQ(atb_speed::common::CreateMoeMlpOperation(param, &op), 0);
}

TEST(moeMlpTest, CreateMoeMlpOperationDefault)
{
    GlobalMockObject::verify(); // 清空mock
    MoeMlpParam param;
    atb::Operation *op = nullptr;
    param.hasMoeEp = false;
    param.enableMoeDistribute = false;
    param.enableDispatchCombineV2 = false;
    param.moeLinearQuantType = {false, false, false, false};
    param.supportSwiGLU = true;
    MOCKER(atb_speed::common::GetTensorIdx).expects(atLeast(1)).with(any(), any()).will(returnValue(0));
    MOCKER(atb_speed::common::AddTensorToList<std::vector<std::string>>).expects(atLeast(1)).with(any(), any());
    MOCKER(atb::CreateOperation<atb::GraphParam>).expects(atLeast(1)).with(any(), any()).will(returnValue(0));
    MOCKER(atb_speed::common::CreateIntegratedGmmOperation).expects(atLeast(1)).with(any(), any()).will(returnValue(0));
    EXPECT_EQ(atb_speed::common::CreateMoeMlpOperation(param, &op), 0);
    param.hasMoeEp = true;
    MOCKER(atb_speed::common::CreateExpertFilterOperation).expects(atLeast(1)).with(any(), any()).will(returnValue(0));
    EXPECT_EQ(atb_speed::common::CreateMoeMlpOperation(param, &op), 0);
    param.packQuantType = atb_speed::common::PackQuantType::ALL_W8A8_DYNAMIC;
    EXPECT_EQ(atb_speed::common::CreateMoeMlpOperation(param, &op), 0);
    param.supportSwiGLU = false;
    EXPECT_EQ(atb_speed::common::CreateMoeMlpOperation(param, &op), 0);
}

} // namespace common
} // namespace atb_speed
