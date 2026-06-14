/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 * MindIE is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */
#include <iostream>
#include <sstream>
#include <memory>
#include "acl/acl.h"
#include "atb_speed/log.h"
#include "atb_speed/utils/check_util.h"
#include "operations/aclnn/utils/utils.h"
#include "w4a16_operation.h"

namespace atb_speed {
namespace common {

W4A16Operation::W4A16Operation(
    const std::string &name,
    AclNNWeightQuantBatchMatmulParam param) : QuantBatchMatmulOperation(name, param), param_(param) {}

atb::Tensor W4A16Operation::PreprocessATBInTensor(atb::Tensor atbTensor, int index)
{
    atb::Tensor squeezedAtbTensor = SqueezeBatchSeq(atbTensor);
    if (index == 1) {  // 1: weight
        squeezedAtbTensor.desc.dtype = ACL_INT4;
        squeezedAtbTensor.desc.shape.dims[DIM1] = CheckIntMulOverFlow(
            squeezedAtbTensor.desc.shape.dims[DIM1], 2);  // 2: 最后一维shape * 2
    }
    return squeezedAtbTensor;
}

} // namespace common
} // namespace atb_speed