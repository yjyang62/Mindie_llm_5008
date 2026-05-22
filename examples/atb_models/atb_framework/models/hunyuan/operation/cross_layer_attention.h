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

#ifndef ATB_SPEED_MODELS_COMMON_CROSSLAYER_ATTENTION_H
#define ATB_SPEED_MODELS_COMMON_CROSSLAYER_ATTENTION_H

#include <vector>
#include <atb/atb_infer.h>
#include "atb_speed/log.h"
#include "atb_speed/utils/operation_util.h"
#include "operations/fusion/linear/linear.h"
#include "operations/fusion/linear/linear_parallel.h"
#include "operations/fusion/norm/norm_linear.h"
#include "operations/fusion/embedding/positional_embedding.h"

namespace atb_speed {
namespace common {

constexpr uint64_t QKV_PROJ_LINEAR_INDEX = 0;
constexpr uint64_t O_PROJ_LINEAR_INDEX = 1;

template <typename NormParamType>
struct CrossLayerAttentionParam {
    // QKV linear param
    bool isGroupedQueryAttention = false;
    bool isBF16 = false;
    bool splitWithStride = false;
    bool qkvHasBias = false;
    bool skipNorm = false;
    bool normHasBias = false;
    bool enableNormQuantOp = true;
    int quantGroupSize = 0;
    // ClA Param
    int headNum = 128;
    bool isCrossLayer = false;

    int packQuantType = atb_speed::common::PackQuantType::ALL_FP;
    std::vector<int> attnLinearQuantType = {};
    std::vector<int> attnLinearTransposeType = {};
    NormParamType normParamType;
    NormParamType normQuantParamType;
    // rope param
    atb_speed::common::RotaryType rotaryType;
    atb::infer::RopeParam ropeParam;
    // self attention param
    bool enableLogN = false;
    bool isFA = false;
    bool isPrefill = false;
    int headDim = 0;
    atb::infer::SelfAttentionParam selfAttentionParam;
    atb::infer::PagedAttentionParam pageAttentionParam;
    atb::infer::ReshapeAndCacheParam reshapeCacheParm;
    // self out linear param
    bool selfAttnHasBias = false;
    bool supportLcoc = false;
    int denseQuantType = atb_speed::common::PackQuantType::PACK_QUANT_UNDEFINED;
    atb_speed::common::TensorParallelInfo selfOutLinearTensorParallelInfo;
};

template <typename NormParamType>
atb::Status CrossLayerAttention(const CrossLayerAttentionParam<NormParamType> &param, atb::Operation **operation);
} // namespace common
} // namespace atb_speed
#endif