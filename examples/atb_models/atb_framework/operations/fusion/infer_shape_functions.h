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

#ifndef ATB_SPEED_MODELS_COMMON_INFER_SHAPE_FUNCTIONS_H
#define ATB_SPEED_MODELS_COMMON_INFER_SHAPE_FUNCTIONS_H

#include <map>
#include <atb/atb_infer.h>
#include "atb_speed/log.h"


namespace atb_speed {
namespace common {

/// If oldShape dimNum is not larger than 2, do nothing. Otherwise, squeeze the shape from [..., headNum, headDim]
/// to [..., headNum * headDim].
void SqueezeHeadNumHeadDim(const atb::Dims &oldShape, atb::Dims &newShape);
/// Unsqueeze shape from [..., headNum * headDim] to [..., headNum, headDim].
void UnsqueezeHeadNumHeadDim(const atb::Dims &oldShape, atb::Dims &newShape, int32_t headNum, int32_t headDim);
/// Unsqueeze shape at `axis`, e.g. [..., x, ...] to [..., 1, x, ...], where x in oldShape is at `axis`.
void UnsqueezeAxis(const atb::Dims &oldShape, atb::Dims &newShape, int32_t axis);
/// If input shape is [B, S, N, D], squeeze it to [B*S, N*D].
void SqueezeBatchAndHiddenSize(const atb::Dims& oldShape, atb::Dims& newShape);
/// Reshape before spliting packed qkv linear for the InterlmV2 model, from [B, S]
/// to [B, S / ((`headNum` / `kvHeadNum` + 2) * `headDim`), `headNum` / `kvHeadNum` + 2, `headDim`]
void InternlmV2QKVSplit(
    const atb::Dims& oldShape, atb::Dims& newShape, int32_t headNum, int32_t kvHeadNum, int32_t headDim);
/// If input shape is [B, S, N, D], squeeze it to [B*S, N ,D]
void SqueezeBatchAndSeq(const atb::Dims& oldShape, atb::Dims& newShape);
} // namespace common
} // namespace atb_speed
#endif