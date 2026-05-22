/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * MindIE is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

#ifndef ATB_SPEED_UTILS_CONTEXT_H
#define ATB_SPEED_UTILS_CONTEXT_H

#include <torch/torch.h>
#include <torch_npu/csrc/framework/OpCommand.h>
#include <torch/custom_class.h>
#include <torch/script.h>

namespace atb_speed {
class Context : public torch::CustomClassHolder {
public:
    static void EnableCacheWorkspace();
    static void DisableCacheWorkspace();
    static void ResumeHcclComm();
};
} // namespace atb_speed
#endif