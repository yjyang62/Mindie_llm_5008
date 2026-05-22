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

#include <thread>
#include <gtest/gtest.h>
#include "mockcpp/mockcpp.hpp"
#include "common_util.h"
#include "dt_tools.h"
#include "config_manager.h"
#include "base_config_manager.h"

using Json = nlohmann::json;
using namespace mindie_llm;

namespace mindie_llm {

class ConfigManagerTestLayerWiseDisaggregated : public testing::Test {
public:
    static void SetUpTestSuite() {}

    static void TearDownTestSuite() {}

    void SetUp()
    {
        jsonPath = GetCwdDirectory() + "/conf/config.json";
    }

    void TearDown()
    {
        GlobalMockObject::verify();
    }

    std::string GetCwdDirectory()
    {
        char buffer[1024];

        if (getcwd(buffer, sizeof(buffer)) == nullptr) {
            std::cerr << "Error getting current directory: " << strerror(errno) << std::endl;
            return "";
        }

        char* temp = strdup(buffer);
        std::string result(temp);
        free(temp);
        return result;
    }

    std::string jsonPath;
};


TEST_F(ConfigManagerTestLayerWiseDisaggregated, TestLayerwiseDisaggregatedErrorProofingVerification)
{
    std::string jsonPath = GetCwdDirectory() + "/conf/config.json";
    EXPECT_EQ(ConfigManager::CreateInstance(jsonPath), true);
}

} // namespace mindie_llm
