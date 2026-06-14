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
#include "environment.h"
#include <fstream>
#include "nlohmann/json.hpp"

namespace dlt::framework {
std::shared_ptr<EnvironmentInfo>
details::CreateAndRegisterEnvironmentInfo(const std::string &name,
                                          std::shared_ptr<details::EnvironmentFactoryBase> factory)
{
    EnvironmentInfoManager &emInstance = EnvironmentInfoManager::GetInstance();
    auto envInfo = std::make_shared<EnvironmentInfo>(name, factory, emInstance);
    emInstance.addEnvironmentInfo(envInfo);
    return envInfo;
}

void EnvironmentInfo::run()
{
    if (!factory_) {
        return;
    }

    BasicEnvironment *env = factory_->CreateEnvironment();
    if (!env) {
        std::cerr << "Failed to create environment for scenario: " << scenarioName_ << std::endl;
        return;
    }

    ::testing::AddGlobalTestEnvironment(env);

    envManager_.setCurrentEnvironment(env);
    envManager_.SetCurrentEnvironmentInfo(this);

    // Run the environment setup
    int ret = ::testing::UnitTest::GetInstance()->Run();
    if (ret != 0) {
        std::cerr << "Environment setup failed, Error code: " << ret << std::endl;
    }

    envManager_.ClearCurrentEnv();
}

EnvironmentInfoManager &EnvironmentInfoManager::GetInstance()
{
    static EnvironmentInfoManager instance;
    return instance;
}

void EnvironmentInfoManager::addEnvironmentInfo(std::shared_ptr<EnvironmentInfo> info)
{
    if (std::find(environmentInfos_.begin(), environmentInfos_.end(), info) != environmentInfos_.end()) {
        std::cerr << "EnvironmentInfo already exists: " << info->GetSenarioName() << std::endl;
        return; // EnvironmentInfo already exists
    }

    environmentInfos_.emplace_back(std::shared_ptr<EnvironmentInfo>(info));
}

void EnvironmentInfoManager::run()
{
    if (environmentInfos_.empty()) {
        return;
    }

    std::for_each(environmentInfos_.begin(), environmentInfos_.end(), [](std::shared_ptr<EnvironmentInfo> envInfo) {
        if (envInfo) {
            std::cout << "Running environment: " << envInfo->GetSenarioName() << std::endl;
            envInfo->run();
            std::cout << "Finished running environment: " << envInfo->GetSenarioName() << std::endl;
        }
    });
}

std::set<std::string> GetEnvs(std::string name)
{
    std::istringstream ss(name);
    std::string env;
    std::set<std::string> envs;
    while (std::getline(ss, env, '|')) {
        if (!env.empty()) {
            envs.insert(env);
        }
    }

    return envs;
}

bool CurrentScenarioMatch(const std::string &scenarioName)
{
    std::set<std::string> envs = GetEnvs(scenarioName);
    EnvironmentInfoManager &manager = EnvironmentInfoManager::GetInstance();
    EnvironmentInfo *currentEnvInfo = manager.GetCurrentEnvironmentInfo();
    if (currentEnvInfo) {
        std::string currentScenarioName = currentEnvInfo->GetSenarioName();
        if (envs.count(currentScenarioName) != 0) {
            return true; // Current scenario matches
        }
    }

    return false; // No match found
}
} // namespace dlt::framework