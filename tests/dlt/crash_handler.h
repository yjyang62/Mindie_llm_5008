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
#include <string>

namespace mindie_llm {
namespace test {

// 初始化崩溃信号处理器（注册 SIGSEGV, SIGABRT 等）
void InitCrashHandler();

// 设置当前上下文（例如测试用例名），用于崩溃时诊断
void SetCurrentContext(const std::string& context);

// 获取当前上下文
std::string GetCurrentContext();

} // namespace test
} // namespace mindie_llm
