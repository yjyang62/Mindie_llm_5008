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

#include "base_operation.h"
#include "operation_factory.h"

namespace atb_torch {
BaseOperation::BaseOperation(const std::string &opType, const std::string &opParam, const std::string &opName)
    : Operation(opName), opType_(opType), opParam_(opParam)
{
    ATB_SPEED_LOG_DEBUG(opName_ << " construct start, opType:" << opType << ", opParam:" << opParam);
    atbOperation_ = OperationFactory::Instance().CreateOperation(opType, opParam);
    CHECK_THROW(atbOperation_ == nullptr, opName_ << "create atb operation fail, please check opParam");

    ATB_SPEED_LOG_DEBUG(opName_ << " construct end");
}

BaseOperation::~BaseOperation() { ATB_SPEED_LOG_DEBUG(opName_ << " disconstruct"); }

std::string BaseOperation::GetOpType() const { return opType_; }

std::string BaseOperation::GetOpParam() const { return opParam_; }
} // namespace atb_torch
