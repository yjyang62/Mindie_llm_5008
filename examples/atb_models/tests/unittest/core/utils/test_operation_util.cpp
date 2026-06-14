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
#include "atb_speed/utils/operation_util.h"

namespace atb_speed {
namespace test {

TEST(OperationUtilTest, CreateOperation)
{
    GlobalMockObject::verify();
    auto testFn = [](atb::infer::ActivationParam &param, atb::Operation **op) -> atb::Status {
        CREATE_OPERATION(param, op);
        return atb::NO_ERROR;
    };

    atb::infer::ActivationParam param;
    atb::Operation *op = nullptr;
    EXPECT_EQ(testFn(param, &op), atb::NO_ERROR);
    MOCKER(atb::CreateOperation<atb::infer::ActivationParam>).stubs().with(any(), any()).will(returnValue(1));
    EXPECT_EQ(testFn(param, &op), 1);
}

TEST(OperationUtilTest, CheckOperationStatusReturn)
{
    GlobalMockObject::verify();

    EXPECT_EQ([]() -> atb::Status {CHECK_OPERATION_STATUS_RETURN(0); return atb::NO_ERROR;}(), atb::NO_ERROR);
    EXPECT_EQ([]() -> atb::Status {CHECK_OPERATION_STATUS_RETURN(1); return atb::NO_ERROR;}(), 1);
}

TEST(OperationUtilTest, CheckParamLt)
{
    GlobalMockObject::verify();
    auto testFn = [](int a, int b) -> atb::Status {
        CHECK_PARAM_LT(a, b);
        return atb::NO_ERROR;
    };

    EXPECT_EQ(testFn(12, 13), atb::NO_ERROR);
    EXPECT_EQ(testFn(16, 15), atb::ERROR_INVALID_PARAM);
}

TEST(OperationUtilTest, CheckParamRt)
{
    GlobalMockObject::verify();
    auto testFn = [](int a, int b) -> atb::Status {
        CHECK_PARAM_GT(a, b);
        return atb::NO_ERROR;
    };

    EXPECT_EQ(testFn(23, 22), atb::NO_ERROR);
    EXPECT_EQ(testFn(25, 26), atb::ERROR_INVALID_PARAM);
}

TEST(OperationUtilTest, CheckParamNe)
{
    GlobalMockObject::verify();
    auto testFn = [](int a, int b) -> atb::Status {
        CHECK_PARAM_NE(a, b);
        return atb::NO_ERROR;
    };

    EXPECT_EQ(testFn(33, 32), atb::NO_ERROR);
    EXPECT_EQ(testFn(45, 45), atb::ERROR_INVALID_PARAM);
}

TEST(OperationUtilTest, CheckTensordescDimnum)
{
    GlobalMockObject::verify();
    auto testFn = [](int a) -> atb::Status {
        CHECK_TENSORDESC_DIMNUM_VALID(a);
        return atb::NO_ERROR;
    };

    EXPECT_EQ(testFn(4), atb::NO_ERROR);
    EXPECT_EQ(testFn(55), atb::ERROR_INVALID_PARAM);
}

}
}

