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
 
#pragma once

#include <gtest/gtest.h>
#define private public
#include "data_type.h"
#include "llm_manager.h"
#include "llm_manager_impl.h"

namespace mindie_llm {
struct SpyResponseInfo {
    TensorMap outputs;
    bool isFinal;
};

class LlmManagerTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override
    {
        void ClearResponseSpy();
        ClearResponseSpy();
    }

public:
    static std::map<std::string, SpyResponseInfo> responseSpyMap;
};
} // namespace mindie_llm