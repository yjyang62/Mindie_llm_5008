/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
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
#define private public
#include "dcmi_wrapper.h"
#include "threadpool_monitor.h"

using namespace mindie_llm;

class DCMIWrapperTest : public testing::Test {
protected:
    void SetUp() override
    {
        wrapper = &DCMIWrapper::GetInstance();
    }

    void TearDown() override
    {
        wrapper->Finalize();
    }

    DCMIWrapper* wrapper;
};

// 单例测试
TEST_F(DCMIWrapperTest, SingletonPattern)
{
    DCMIWrapper& instance1 = DCMIWrapper::GetInstance();
    DCMIWrapper& instance2 = DCMIWrapper::GetInstance();
    EXPECT_EQ(&instance1, &instance2);
}

// 初始状态测试
TEST_F(DCMIWrapperTest, InitialState)
{
    EXPECT_FALSE(wrapper->IsInitialized());
    EXPECT_EQ(wrapper->handle_, nullptr);
    EXPECT_TRUE(wrapper->funcCache_.empty());
}

// 重复初始化测试
TEST_F(DCMIWrapperTest, InitializeWhenAlreadyInitialized)
{
    wrapper->initialized_ = true;
    EXPECT_TRUE(wrapper->Initialize());
}

// 清理功能测试
TEST_F(DCMIWrapperTest, CleanUp)
{
    wrapper->funcCache_["test"] = (void*)0x5678;

    wrapper->CleanUp();
    EXPECT_EQ(wrapper->handle_, nullptr);
    EXPECT_TRUE(wrapper->funcCache_.empty());
}

// Finalize测试
TEST_F(DCMIWrapperTest, Finalize)
{
    wrapper->initialized_ = true;
    wrapper->Finalize();
    EXPECT_FALSE(wrapper->IsInitialized());
}

// 未初始化时获取函数
TEST_F(DCMIWrapperTest, GetFunctionWhenNotInitialized)
{
    auto func = wrapper->GetFunction<int(*)()>("dcmi_init");
    EXPECT_EQ(func, nullptr);
}

// 获取函数指针测试
TEST_F(DCMIWrapperTest, GetFunction)
{
    wrapper->initialized_ = true;
    wrapper->handle_ = nullptr;
    wrapper->funcCache_["test_func"] = reinterpret_cast<void*>(0x1234);

    EXPECT_NO_THROW(wrapper->GetFunction<int(*)()>("test_func"));
    wrapper->funcCache_.clear();
}