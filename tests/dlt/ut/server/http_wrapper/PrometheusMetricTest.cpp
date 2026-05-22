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
#include <libgen.h>
#include <gtest/gtest.h>
#include <cstdlib>
#include "mockcpp/mockcpp.hpp"
#define private public
#include "prometheus_metrics.h"
#include "env_util.h"
#include "config_manager.h"
#include "config_manager/config_manager_impl.h"
#include "mock_util.h"

using namespace mindie_llm;

MOCKER_CPP_OVERLOAD_EQ(ModelDeployConfig)

class PrometheusMetricsTest : public testing::Test {
protected:
    void SetUp()
    {
        EnvUtil::GetInstance().SetEnvVar("MIES_SERVICE_MONITOR_MODE", "1");
        EnvUtil::GetInstance().SetEnvVar("RANK_TABLE_FILE", GetParentDirectory() + "/../../config_manager/conf/ranktable.json");
        EnvUtil::GetInstance().SetEnvVar("MIES_CONTAINER_IP", "127.0.0.1");
        EnvUtil::GetInstance().SetEnvVar("HOST_IP", "127.0.0.1");
        EnvUtil::GetInstance().SetEnvVar("MINDIE_CHECK_INPUTFILES_PERMISSION", "1");
        ConfigManager::CreateInstance(GetParentDirectory() + "/../../config_manager/conf/config_http.json");
        ModelDeployConfig config;
        config.modelInstanceType = "StandardMock";
        config.modelName = "llama_65b";
        config.modelWeightPath = "../../config_manager/conf";
        modelConfig = {config};
        MOCKER_CPP(&GetModelDeployConfig, const std::vector<ModelDeployConfig>& (*)())
        .stubs()
        .will(returnValue(modelConfig));
    }

    void TearDown()
    {
        EnvUtil::GetInstance().ClearEnvVar("MIES_SERVICE_MONITOR_MODE");
        EnvUtil::GetInstance().ClearEnvVar("RANK_TABLE_FILE");
        EnvUtil::GetInstance().ClearEnvVar("MINDIE_CHECK_INPUTFILES_PERMISSION");
        EnvUtil::GetInstance().ClearEnvVar("MIES_CONTAINER_IP");
        EnvUtil::GetInstance().ClearEnvVar("HOST_IP");
        GlobalMockObject::verify();
    }

    std::string GetParentDirectory()
    {
        char buffer[1024];
        try {
            return std::filesystem::current_path().string();
        } catch (const std::filesystem::filesystem_error& e) {
            std::cerr << "Error getting current directory: " << e.what() << std::endl;
            return "";
        }
    }

    std::vector<ModelDeployConfig> modelConfig;
};

TEST_F(PrometheusMetricsTest, testActivateMode)
{
    std::shared_ptr<PrometheusMetrics> pm = PrometheusMetrics::GetInstance();
    EXPECT_NE(pm, nullptr);
    EXPECT_TRUE(pm->isActivate_);
    pm->TTFTObserve(1);
    pm->TBTObserve(1);
    pm->E2EObserve(1);
    
    pm->RequestNumberCount();
    pm->ResponseNumberCount();
    pm->PrefillThroughputGaugeCollect(1.0f);
    pm->DecodeThroughputGaugeCollect(1.0f);
    pm->totalRequestNum_ = 0;
    pm->failedRequestNum_ = 1;
    pm->FailedRequestRateGaugeCollect();
    pm->totalRequestNum_ = 1;
    pm->FailedRequestRateGaugeCollect();
    pm->RequestInputTokenHistogramCollect(1);
    pm->ResponseOutputTokenHistogramCollect(1);
    pm->RequestInputTokenCount(1);
    pm->ResponseOutputTokenCount(1);
    pm->CacheBlockDataCollect(1, 1, 1, 1);
    
    std::string metricsResult;
    pm->GetMetricsResult(metricsResult);
    
    pm->RadixMatchDataCollect(1, 1);
    pm->PreemptNumCount(1);
    pm->RequestNumsGaugeCollect(1, 1, 1);
    if (pm->collectThread_.joinable()) {
        pm->shutdown_ = true;
        pm->collectThread_.join();
    }
}