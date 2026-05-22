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

#ifndef __ENVIRONMENT_H__
#define __ENVIRONMENT_H__

#include "gtest/gtest.h"
#include <memory>
#include <vector>
#include <string>
#include <algorithm>
#include "noncopyable.h"

namespace dlt::framework {

class BasicEnvironment : public ::testing::Environment {
public:
    BasicEnvironment() {}
    virtual ~BasicEnvironment() = default;

    virtual void SetUp() {}
    virtual void TearDown(){};
};

namespace details {

class EnvironmentFactoryBase : private NonCopyable {
public:
    virtual ~EnvironmentFactoryBase() = default;
    virtual BasicEnvironment *CreateEnvironment() = 0;
    void LoadScene(std::string scenarioName, std::string configPath);

protected:
    EnvironmentFactoryBase() = default;
};

template <typename EnvClass> class EnvironmentFactoryImpl : public EnvironmentFactoryBase {
public:
    virtual BasicEnvironment *CreateEnvironment() override { return new EnvClass(); }
};

} // namespace details

class EnvironmentInfoManager : public NonCopyable {
public:
    static EnvironmentInfoManager &GetInstance();

    void addEnvironmentInfo(std::shared_ptr<class EnvironmentInfo> info);
    void run();
    void setCurrentEnvironment(BasicEnvironment *env) { current_env_ = env; }
    void SetCurrentEnvironmentInfo(class EnvironmentInfo *info) { currentEnvironmentInfo_ = info; }
    void ClearCurrentEnv()
    {
        current_env_ = nullptr;
        currentEnvironmentInfo_ = nullptr;
    }
    BasicEnvironment *GetCurrentEnvironment() const { return current_env_; }
    class EnvironmentInfo *GetCurrentEnvironmentInfo() const { return currentEnvironmentInfo_; }

private:
    EnvironmentInfoManager() = default;
    ~EnvironmentInfoManager() = default;

    std::vector<std::shared_ptr<class EnvironmentInfo>> environmentInfos_;
    BasicEnvironment *current_env_ = nullptr;
    class EnvironmentInfo *currentEnvironmentInfo_ = nullptr;
};

class EnvironmentInfo : public NonCopyable {
public:
    EnvironmentInfo(const std::string &name, std::shared_ptr<details::EnvironmentFactoryBase> factory,
                    EnvironmentInfoManager &em)
        : scenarioName_(name), factory_(factory), envManager_(em){};
    ~EnvironmentInfo(){};

    void run();
    std::string GetSenarioName() const { return scenarioName_; }

private:
    std::string scenarioName_;
    std::shared_ptr<details::EnvironmentFactoryBase> factory_;
    EnvironmentInfoManager &envManager_;
};

namespace details {

std::shared_ptr<EnvironmentInfo> CreateAndRegisterEnvironmentInfo(const std::string &name,
                                                                  std::shared_ptr<EnvironmentFactoryBase> factory);

} // namespace details

bool CurrentScenarioMatch(const std::string &scenarioName);

} // namespace dlt::framework

#define REQUIRE_SCENARIO(scenarioName)                                                                                 \
    do {                                                                                                               \
        if (!dlt::framework::CurrentScenarioMatch(#scenarioName)) {                                                    \
            return;                                                                                                    \
        }                                                                                                              \
    } while (0);

#endif // __ENVIRONMENT_H__