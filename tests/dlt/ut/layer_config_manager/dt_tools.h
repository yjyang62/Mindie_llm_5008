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

#ifndef DT_TOOLS_H
#define DT_TOOLS_H

#include <iostream>
#include <fstream>
#include <unordered_map>
#include <stdexcept>
#include <sys/stat.h>
#include <unistd.h>
#include "nlohmann/json.hpp"

using OrderedJson = nlohmann::ordered_json;

namespace mindie_llm {
void HandleNestedJson(OrderedJson &j, const std::vector<std::string> &path, const OrderedJson &value);
void ModifyTargetField(const OrderedJson &oldData, OrderedJson &newData,
                       const std::unordered_map<std::string, OrderedJson> &updates);
void UpdateConfigJson(const std::string &oldFilePath, const std::string &newFilePath,
                      const std::unordered_map<std::string, OrderedJson> &updates);
void HandleParamContainer(std::vector<OrderedJson> &testArr, std::map<std::string, OrderedJson::array_t> &paramMap);
std::string GetCwdDirectory();
} // namespace mindie_llm

#endif // DT_TOOLS_H