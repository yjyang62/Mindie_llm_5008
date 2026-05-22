/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
 * MindIE is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

#ifndef ATB_TORCH_BASEOPERATION_H
#define ATB_TORCH_BASEOPERATION_H
#include <string>
#include <unordered_map>
#include <torch/script.h>
#include <atb/atb_infer.h>
#include "operation.h"
#include "atb_speed/log.h"

namespace atb_torch {
class BaseOperation : public Operation {
public:
    BaseOperation(const std::string &opType, const std::string &opParam, const std::string &opName);
    ~BaseOperation() override;
    std::string GetOpType() const;
    std::string GetOpParam() const;

protected:
    std::string opType_;
    std::string opParam_;
};
} // namespace atb_torch
#endif