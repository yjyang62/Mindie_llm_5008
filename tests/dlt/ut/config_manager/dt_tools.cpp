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

#include "dt_tools.h"

namespace mindie_llm {
void HandleNestedJson(OrderedJson &j, const std::vector<std::string> &path, const OrderedJson &value)
{
    if (path.empty()) {
        std::cerr << "Path cannot be empty." << std::endl;
        return;
    }

    OrderedJson *current = &j;
    size_t pathSize = path.size();

    for (size_t i = 0; i < pathSize - 1; ++i) {
        if (!current->contains(path[i])) {
            (*current)[path[i]] = OrderedJson::object();
        }
        current = &(*current)[path[i]];

        if (!current->is_object()) {
            std::cerr << "Expected an object at path: " << path[i] << std::endl;
            return;
        }
    }

    (*current)[path.back()] = value;
}

void ModifyTargetField(const OrderedJson &oldData, OrderedJson &newData,
                       const std::unordered_map<std::string, OrderedJson> &updates)
{
    for (auto it = oldData.begin(); it != oldData.end(); ++it) {
        newData[it.key()] = it.value();
    }

    for (const auto &[key, value] : updates) {
        if (value.is_object()) {
            for (auto &[nestedKey, nestedValue] : value.items()) {
                try {
                    std::vector<std::string> path = {key, nestedKey};
                    HandleNestedJson(newData, path, nestedValue);
                } catch (const std::exception &e) {
                    std::cerr << "Failed to update nested key " << key << "-" << value << ":" << e.what() << std::endl;
                    return;
                }
            }
        } else {
            if (!newData.contains(key)) {
                std::cerr << "key not found: " << key;
                return;
            }
            newData[key] = value;
        }
    }
}

void UpdateConfigJson(const std::string &oldFilePath, const std::string &newFilePath,
                      const std::unordered_map<std::string, OrderedJson> &updates)
{
    std::ifstream oldFd(oldFilePath);
    if (!oldFd.is_open()) {
        std::cerr << "Failed to open source config json file." << std::endl;
        return;
    }

    OrderedJson oldData;
    try {
        oldFd >> oldData;
    } catch (const std::exception &e) {
        std::cerr << "Failed to parse source json file." << e.what() << std::endl;
        return;
    }
    oldFd.close();

    OrderedJson newData = OrderedJson::object();
    ModifyTargetField(oldData, newData, updates);

    std::ofstream newFd(newFilePath);
    if (!newFd.is_open()) {
        std::cerr << "Failed to open modified config json file." << std::endl;
        return;
    }

    size_t indent = 4;
    try {
        newFd << newData.dump(indent);
    } catch (const std::exception &e) {
        std::cerr << "Failed to write modified config json file." << e.what() << std::endl;
        return;
    }
    newFd.close();

    mode_t fileMode = 0640;
    chmod(newFilePath.c_str(), fileMode);
}

void HandleParamContainer(std::vector<OrderedJson> &testArr, std::map<std::string, OrderedJson::array_t> &paramMap)
{
    OrderedJson tmpJsonObj;

    for (const auto &[key, paramArr] : paramMap) {
        if (paramArr.empty()) {
            testArr.clear();
            return;
        }

        for (size_t i = 0; i < paramArr.size(); ++i) {
            tmpJsonObj[key] = paramArr[i];
            testArr.push_back(tmpJsonObj);
            tmpJsonObj = {};
        }
    }
}

std::string GetCwdDirectory()
{
    char buffer[1024];

    if (getcwd(buffer, sizeof(buffer)) == nullptr) {
        std::cerr << "Error getting current directory: " << strerror(errno) << std::endl;
        return "";
    }

    char *temp = strdup(buffer);
    std::string result(temp);
    free(temp);
    return result;
}
} // namespace mindie_llm