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
#include <cstdlib>
#include "operations/aclnn/core/acl_nn_global_cache.h"

namespace atb_speed {
namespace test {

using atb_speed::common::AclNNGlobalCache;
using atb_speed::common::AclNNOpCache;

bool IsVariankPackEqual(const AclNNGlobalCache&, const atb::VariantPack&)
{
    return true;
}

class AclNNGlobalCacheTest : public ::testing::Test {
protected:
    void SetUp() override
    {
        const char* mindieEnv = std::getenv("MINDIE_ACLNN_CACHE_GLOBAL_COUNT");
        if (mindieEnv) mindieEnvValue = mindieEnv;
        unsetenv("MINDIE_ACLNN_CACHE_GLOBAL_COUNT");
    }
    
    void TearDown() override
    {
        if (!mindieEnvValue.empty()) {
            setenv("MINDIE_ACLNN_CACHE_GLOBAL_COUNT", mindieEnvValue.c_str(), 1);
        } else {
            unsetenv("MINDIE_ACLNN_CACHE_GLOBAL_COUNT");
        }
    }
    
private:
    std::string atbEnvValue;
    std::string mindieEnvValue;
};

TEST_F(AclNNGlobalCacheTest, DefaultConstructor)
{
    AclNNGlobalCache cache;
    EXPECT_EQ(16, cache.globalCacheCountMax_);
}

TEST_F(AclNNGlobalCacheTest, EnvironmentVariableSetting)
{
    setenv("MINDIE_ACLNN_CACHE_GLOBAL_COUNT", "20", 1);
    AclNNGlobalCache cache2;
    EXPECT_EQ(20, cache2.globalCacheCountMax_);

    setenv("MINDIE_ACLNN_CACHE_GLOBAL_COUNT", "invalid", 1);
    AclNNGlobalCache cache3;
    EXPECT_EQ(16, cache3.globalCacheCountMax_);

    setenv("MINDIE_ACLNN_CACHE_GLOBAL_COUNT", "150", 1);
    AclNNGlobalCache cache4;
    EXPECT_EQ(16, cache4.globalCacheCountMax_);
}

TEST_F(AclNNGlobalCacheTest, GetNonExistentCache)
{
    AclNNGlobalCache cache;
    atb::VariantPack vp;
    
    auto result = cache.GetGlobalCache("test_op", vp);
    EXPECT_EQ(nullptr, result);
}

TEST_F(AclNNGlobalCacheTest, AddAndGetRepeatableCache)
{
    AclNNGlobalCache cache;
    atb::VariantPack vp;

    auto testCache = std::make_shared<AclNNOpCache>();
    testCache->executorRepeatable = true;

    auto status = cache.UpdateGlobalCache("test_op", testCache);
    EXPECT_EQ(atb::NO_ERROR, status);

    auto result = cache.GetGlobalCache("test_op", vp);
    EXPECT_NE(nullptr, result);
    EXPECT_EQ(testCache, result);
}

TEST_F(AclNNGlobalCacheTest, AddNonRepeatableCache)
{
    AclNNGlobalCache cache;

    auto testCache = std::make_shared<AclNNOpCache>();
    testCache->executorRepeatable = false;

    auto status = cache.UpdateGlobalCache("test_op", testCache);
    EXPECT_EQ(atb::NO_ERROR, status);

    atb::VariantPack vp;
    auto result = cache.GetGlobalCache("test_op", vp);
    EXPECT_EQ(nullptr, result);
}

TEST_F(AclNNGlobalCacheTest, PrintCache)
{
    AclNNGlobalCache cache;

    auto cache1 = std::make_shared<AclNNOpCache>();
    cache1->executorRepeatable = true;
    cache.UpdateGlobalCache("op1", cache1);
    
    auto cache2 = std::make_shared<AclNNOpCache>();
    cache2->executorRepeatable = true;
    cache.UpdateGlobalCache("op2", cache2);

    std::string result = cache.PrintGlobalCache();

    EXPECT_NE(std::string::npos, result.find("op1"));
    EXPECT_NE(std::string::npos, result.find("op2"));
    EXPECT_NE(std::string::npos, result.find("Cache Addr"));
}

TEST_F(AclNNGlobalCacheTest, CacheIsolationByOpName)
{
    AclNNGlobalCache cache;
    atb::VariantPack vp;

    auto cache1 = std::make_shared<AclNNOpCache>();
    cache1->executorRepeatable = true;
    cache.UpdateGlobalCache("op1", cache1);
    
    auto cache2 = std::make_shared<AclNNOpCache>();
    cache2->executorRepeatable = true;
    cache.UpdateGlobalCache("op2", cache2);

    auto result1 = cache.GetGlobalCache("op1", vp);
    EXPECT_EQ(cache1, result1);
    
    auto result2 = cache.GetGlobalCache("op2", vp);
    EXPECT_EQ(cache2, result2);

    auto result3 = cache.GetGlobalCache("op3", vp);
    EXPECT_EQ(nullptr, result3);
}

} // namespace test
} // namespace atb_speed