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

#include "atb_speed/utils/check_util.h"
#include "operations/aclnn/utils/utils.h"
#include "operations/fusion/infer_shape_functions.h"

namespace atb_speed {
namespace common {

void SqueezeHeadNumHeadDim(const atb::Dims &oldShape, atb::Dims &newShape)
{
    newShape = oldShape;
    if (oldShape.dimNum >= 2) {  // 2: 对输入tensor的后两维进行合并，若维度小于2则不做修改
        newShape.dimNum = oldShape.dimNum - 1;
        newShape.dims[newShape.dimNum - 1] = \
            CheckIntMulOverFlow(oldShape.dims[oldShape.dimNum - 2], oldShape.dims[oldShape.dimNum - 1]);  // 2: index
    }
}

void UnsqueezeHeadNumHeadDim(const atb::Dims &oldShape, atb::Dims &newShape, int32_t headNum, int32_t headDim)
{
    newShape = oldShape;
    if (oldShape.dimNum == 0) {
        return;
    }
    newShape.dimNum = oldShape.dimNum + 1;
    newShape.dims[newShape.dimNum - 2] = headNum;  // -2: headNum
    newShape.dims[newShape.dimNum - 1] =  headDim;  // -1: headDim
}

void UnsqueezeAxis(const atb::Dims &oldShape, atb::Dims &newShape, int32_t axis)
{
    newShape = oldShape;
    newShape.dimNum = oldShape.dimNum + 1;
    newShape.dims[axis] = 1;
    for (uint64_t i = axis + 1; i < std::min(newShape.dimNum, static_cast<uint64_t>(8)); i++) {  // 8: tensor维度上限
        newShape.dims[i] = oldShape.dims[i - 1];
    }
}

void SqueezeBatchAndSeq(const atb::Dims& oldShape, atb::Dims& newShape)
{
    if (oldShape.dimNum == NUM3) { // 3: If input shape is [B, S, N, D], squeeze it to [B*S, N ,D]
        newShape.dimNum = NUM2;
        newShape.dims[DIM0] = CheckIntMulOverFlow(oldShape.dims[DIM0], oldShape.dims[DIM1]);
        newShape.dims[DIM1] = oldShape.dims[DIM2];
    } else {
        newShape = oldShape;
    }
}

void SqueezeBatchAndHiddenSize(const atb::Dims& oldShape, atb::Dims& newShape)
{
    if (oldShape.dimNum == 4) {  // 4: 若输入是[B,S,N,D]，则合并为[BS,ND]
        newShape.dimNum = 2;  // 2: [BS,ND]
        newShape.dims[0] = CheckIntMulOverFlow(oldShape.dims[0], oldShape.dims[1]);  // 0,0,1: [B,S] => [BS]
        newShape.dims[1] = CheckIntMulOverFlow(oldShape.dims[2], oldShape.dims[3]);  // 1,2,3: [N,D] => [ND]
    } else {
        newShape = oldShape;
    }
}

void InternlmV2QKVSplit(
    const atb::Dims& oldShape, atb::Dims& newShape, int32_t headNum, int32_t kvHeadNum, int32_t headDim)
{
    if (kvHeadNum == 0 || headDim == 0) {
        ATB_SPEED_LOG_ERROR("kvHeadNum or headDim is 0 in InternlmV2QKVSplit, "
                       << "reshape failed, newShape remains the same as oldShape");
        newShape = oldShape;
        return;
    }
    newShape.dimNum = 4;  // 4: 新的shape维度为4
    size_t newShapeDimIndex = 0;
    size_t oldShapeDimIndex = 0;
    newShape.dims[newShapeDimIndex++] = oldShape.dims[oldShapeDimIndex++];
    newShape.dims[newShapeDimIndex++] = \
        oldShape.dims[oldShapeDimIndex++] / (CheckIntMulOverFlow(
            (2 + headNum / kvHeadNum), headDim)  // 2: k + v linear
    );
    if ((2 + headNum / kvHeadNum)  // 2: k + v linear
        > std::numeric_limits<int32_t>::max()) {
        newShape = oldShape;
        return;
    }
    newShape.dims[newShapeDimIndex++] = 2 + headNum / kvHeadNum; // 2: k + v linear
    newShape.dims[newShapeDimIndex++] = headDim;
}

} // namespace common
} // namespace atb_speed