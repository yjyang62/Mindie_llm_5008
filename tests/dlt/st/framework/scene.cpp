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
 
#include "scene.h"
#include <fstream>
#include <cstdio>
#include <iostream>
#include "nlohmann/json.hpp"
#include "environment.h"

namespace dlt::framework {
using Json = nlohmann::json;

Scene &Scene::GetInstance()
{
    static Scene instance;
    return instance;
}

bool Scene::LoadScene()
{
    try {
        std::string configFileName = GetWorkPath(__FILE__) + "config/scene.json";
        std::ifstream configFile(configFileName);
        if (!configFile.is_open()) {
            std::cerr << "Failed to open config file: " << configFileName << std::endl;
            return false;
        }

        Json configJson;
        configFile >> configJson;
        if (configJson.is_null()) {
            std::cerr << "Config file is empty or invalid: " << configFileName << std::endl;
            return false;
        }

        // std::cout << "Loaded scene configuration from: " << configJson["scenarios"].dump() << std::endl;
        for (const auto &scene : configJson["scenarios"].items()) {
            const std::string &scenarioName = scene.key();
            const auto &scenarioConfig = scene.value();
            std::cout << "Loaded scenario: " << scenarioName << std::endl;
            LoadNode(scenarioConfig["p_instances_num"].get<int>(), "P");
            LoadNode(scenarioConfig["d_instances_num"].get<int>(), "D");

            CreateAndRegisterEnvironmentInfo(scenarioName,
                                             std::make_shared<details::EnvironmentFactoryImpl<BasicEnvironment>>());
        }
        return true;
    } catch (const std::exception &e) {
        std::cerr << "Error loading scene configuration: " << e.what() << std::endl;
        return false;
    }
}

std::string Scene::GetWorkPath(const std::string file) const
{
    size_t last_slash = file.find_last_of("/\\");
    if (last_slash != std::string::npos) {
        return file.substr(0, last_slash) + "/../";
    }
    return "./";
}

void Scene::LoadNode(int nodeCount, std::string nodeType)
{
    for (int i = 0; i < nodeCount; ++i) {
        std::string nodeName = nodeType + "-node-" + std::to_string(i + 1);

        // 调用unshare脚本启动对应的server程序
        std::string cmd =
            GetWorkPath(__FILE__) + "utils/unshare.sh -n " + nodeName;
        std::cout << "Loading node: " << nodeName << ", cmd: " << cmd << std::endl;

        FILE *pipe = popen(cmd.c_str(), "r");
        if (pipe) {
            char buffer[128];
            while (fgets(buffer, sizeof(buffer), pipe)) {
                std::cout << buffer; // 逐行读取输出
            }
            pclose(pipe);
        }
    }
}
} // namespace dlt::framework