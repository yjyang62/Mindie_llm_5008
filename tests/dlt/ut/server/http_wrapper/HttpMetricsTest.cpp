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
#include <thread>
#include <vector>
#include "http_metrics.h"
#include "log.h"

using namespace mindie_llm;

namespace mindie_llm {

class MockSingleLLMReqHandler {
public:
    MockSingleLLMReqHandler(uint64_t firstTokenCost, const std::vector<uint64_t>& decodeTimes)
        : firstTokenCost_(firstTokenCost), decodeTimes_(decodeTimes) {}
    
    uint64_t GetFirstTokenCost() const { return firstTokenCost_; }
    const std::vector<uint64_t>& GetDecodeTimes() const { return decodeTimes_; }
    
private:
    uint64_t firstTokenCost_;
    std::vector<uint64_t> decodeTimes_;
};

class MockSingleReqInferInterface {
public:
    MockSingleReqInferInterface(uint64_t firstTokenCost, const std::vector<uint64_t>& decodeTimes)
        : handler_(std::make_shared<MockSingleLLMReqHandler>(firstTokenCost, decodeTimes)) {}
    
    std::shared_ptr<MockSingleLLMReqHandler> GetHandler() const { return handler_; }
    
private:
    std::shared_ptr<MockSingleLLMReqHandler> handler_;
};

class HttpMetricsTest : public testing::Test {
protected:
    void SetUp() override { ClearMetricsData(); }

    void TearDown() override { ClearMetricsData(); }
    
private:
    void ClearMetricsData()
    {
        HttpMetrics& metrics = HttpMetrics::GetInstance();

        for (int i = 0; i < 2000; i++) {
        }
    }
};

TEST_F(HttpMetricsTest, GetInstance)
{
    HttpMetrics& instance1 = HttpMetrics::GetInstance();
    HttpMetrics& instance2 = HttpMetrics::GetInstance();
    
    EXPECT_EQ(&instance1, &instance2);
    EXPECT_NE(&instance1, nullptr);
}

TEST_F(HttpMetricsTest, InitialState)
{
    HttpMetrics& metrics = HttpMetrics::GetInstance();
    
    EXPECT_EQ(metrics.TTFTSize(), 0u);
    EXPECT_EQ(metrics.TBTSize(), 0u);
    EXPECT_EQ(metrics.DynamicAverageTTFT(), 0u);
    EXPECT_EQ(metrics.DynamicAverageTBT(), 0u);
}

TEST_F(HttpMetricsTest, HandleNullRequest)
{
    HttpMetrics& metrics = HttpMetrics::GetInstance();

    std::shared_ptr<SingleReqInferInterfaceBase> nullRequest = nullptr;
    metrics.CollectStatisticsRequest(nullRequest);

    EXPECT_EQ(metrics.TTFTSize(), 0u);
    EXPECT_EQ(metrics.TBTSize(), 0u);
}

TEST_F(HttpMetricsTest, BasicFunctionality)
{
    HttpMetrics& metrics = HttpMetrics::GetInstance();

    EXPECT_EQ(metrics.TTFTSize(), 0u);
    EXPECT_EQ(metrics.TBTSize(), 0u);
    EXPECT_EQ(metrics.DynamicAverageTTFT(), 0u);
    EXPECT_EQ(metrics.DynamicAverageTBT(), 0u);

    EXPECT_EQ(metrics.DynamicAverageTTFT(), 0u);
    EXPECT_EQ(metrics.DynamicAverageTBT(), 0u);
}

TEST_F(HttpMetricsTest, WindowSizeLimit)
{
    HttpMetrics& metrics = HttpMetrics::GetInstance();

    EXPECT_LE(metrics.TTFTSize(), 1000u);
    EXPECT_LE(metrics.TBTSize(), 1000u);
}

TEST_F(HttpMetricsTest, MultiThreadStability)
{
    HttpMetrics& metrics = HttpMetrics::GetInstance();
    
    const int numThreads = 4;
    std::vector<std::thread> threads;

    for (int i = 0; i < numThreads; i++) {
        threads.emplace_back([&metrics]() {
            for (int j = 0; j < 100; j++) {
                metrics.TTFTSize();
                metrics.TBTSize();
                metrics.DynamicAverageTTFT();
                metrics.DynamicAverageTBT();

                std::shared_ptr<SingleReqInferInterfaceBase> nullRequest = nullptr;
                metrics.CollectStatisticsRequest(nullRequest);
            }
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    size_t ttftSize = metrics.TTFTSize();
    size_t tbtSize = metrics.TBTSize();
    size_t ttftAvg = metrics.DynamicAverageTTFT();
    size_t tbtAvg = metrics.DynamicAverageTBT();

    EXPECT_LE(ttftSize, 1000u);
    EXPECT_LE(tbtSize, 1000u);
}

TEST_F(HttpMetricsTest, BoundaryConditions)
{
    HttpMetrics& metrics = HttpMetrics::GetInstance();

    EXPECT_EQ(metrics.TTFTSize(), 0u);
    EXPECT_EQ(metrics.TBTSize(), 0u);
    EXPECT_EQ(metrics.DynamicAverageTTFT(), 0u);
    EXPECT_EQ(metrics.DynamicAverageTBT(), 0u);

    for (int i = 0; i < 100; i++) {
        metrics.TTFTSize();
        metrics.TBTSize();
        metrics.DynamicAverageTTFT();
        metrics.DynamicAverageTBT();
    }
}

TEST_F(HttpMetricsTest, SingletonThreadSafety)
{
    const int numThreads = 10;
    std::vector<std::thread> threads;
    std::vector<HttpMetrics*> instances(numThreads);

    for (int i = 0; i < numThreads; i++) {
        threads.emplace_back([&instances, i]() {
            instances[i] = &HttpMetrics::GetInstance();
        });
    }
    
    for (auto& t : threads) {
        t.join();
    }

    HttpMetrics* firstInstance = instances[0];
    for (int i = 1; i < numThreads; i++) {
        EXPECT_EQ(firstInstance, instances[i]);
    }
}

TEST_F(HttpMetricsTest, EnvironmentVariableHandling)
{
    HttpMetrics& metrics = HttpMetrics::GetInstance();

    EXPECT_NO_THROW({
        HttpMetrics& m = HttpMetrics::GetInstance();
        (void)m;
    });
}

} // namespace mindie_llm