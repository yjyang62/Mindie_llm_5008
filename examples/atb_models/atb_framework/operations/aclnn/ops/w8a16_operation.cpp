/*
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
#include "atb_speed/log.h"
#include "operations/aclnn/utils/utils.h"
#include "w8a16_operation.h"

namespace atb_speed {
namespace common {

W8A16Operation::W8A16Operation(
    const std::string &name,
    AclNNWeightQuantBatchMatmulParam param) : QuantBatchMatmulOperation(name, param), param_(param) {}

atb::Tensor W8A16Operation::PreprocessATBInTensor(atb::Tensor atbTensor, int index)
{
    ATB_SPEED_LOG_DEBUG("W8A16 preprocess ATB in tensor " << index);
    atb::Tensor squeezedATBTensor = SqueezeBatchSeq(atbTensor);
    return squeezedATBTensor;
}

} // namespace common
} // namespace atb_speed