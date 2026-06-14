/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
 * MindIE is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

#ifndef ATB_TORCH_CONFIG_H
#define ATB_TORCH_CONFIG_H
#include <string>
#include <set>
#include <torch/extension.h>

namespace atb_torch {
class Config {
public:
    static Config &Instance();
    bool IsUseTilingCopyStream() const;
    bool IsTaskQueueEnable() const;
    uint64_t GetGlobalWorkspaceSize() const;
    void SetGlobalWorkspaceSize(uint64_t size);
    torch::Tensor &GetGlobalWorkspaceTensor();

private:
    Config();
    ~Config();
    bool IsEnable(const char *env, bool enable = false) const;

private:
    bool isUseTilingCopyStream_ = false;
    bool isTaskQueueEnable_ = false;
    uint64_t globalWorkspaceSize_ = 0;
    torch::Tensor globalWorkspaceTensor_;
};
} // namespace atb_torch
#endif