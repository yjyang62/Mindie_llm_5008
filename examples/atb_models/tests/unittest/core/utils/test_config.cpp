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
#include <cstdlib>
#include <gtest/gtest.h>
#include <mockcpp/mockcpp.hpp>
#include "atb_speed/utils/config.h"

namespace atb_speed {
namespace test {

TEST(ConfigTest, Config)
{
    GlobalMockObject::verify();
    Config config;

    EXPECT_EQ(config.IsTorchTensorFormatCast(), true);
    EXPECT_EQ(config.IsConvertNCHWToND(), true);
    EXPECT_EQ(config.IsLayerInternalTensorReuse(), true);
    EXPECT_EQ(config.IsUseTilingCopyStream(), false);
}

TEST(ConfigTest, IsEnable)
{
    GlobalMockObject::verify();
    Config config;
    EXPECT_EQ(config.IsEnable("TEST_ENV"), false);
    setenv("TEST_ENV", "1", 1);
    EXPECT_EQ(config.IsEnable("TEST_ENV"), true);
}

}
}