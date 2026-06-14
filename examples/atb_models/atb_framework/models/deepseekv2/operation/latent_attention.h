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

#ifndef ATB_SPEED_MODELS_COMMON_LATENT_ATTENTION_H
#define ATB_SPEED_MODELS_COMMON_LATENT_ATTENTION_H

#include <vector>
#include <map>
#include <atb/atb_infer.h>
#include "atb_speed/log.h"
#include "atb_speed/utils/operation_util.h"
#include "operations/fusion/linear/linear.h"
#include "operations/fusion/linear/linear_parallel.h"
#include "operations/fusion/norm/norm_linear.h"
#include "operations/fusion/embedding/positional_embedding.h"

namespace atb_speed {
namespace common {

constexpr uint64_t Q_PROJ_A_LINEAR_INDEX = 0;
constexpr uint64_t Q_PROJ_B_LINEAR_INDEX = 1;
constexpr uint64_t KV_PROJ_A_LINEAR_INDEX = 2;
constexpr uint64_t KV_PROJ_B_FOR_Q_LINEAR_INDEX = 3;
constexpr uint64_t KV_PROJ_B_FOR_V_LINEAR_INDEX = 4;
constexpr uint64_t O_LINEAR_INDEX = 5;

template <typename NormParamType>
struct LatentAttentionParam {
    // QKV linear param
    bool isGroupedQueryAttention = false;
    bool isBF16 = false;
    bool splitWithStride = false;
    bool qkvHasBias = false;
    bool skipNorm = false;
    bool normHasBias = false;
    bool enableNormQuantOp = true;
    int quantGroupSize = 0;
    // MLA Param
    int qLoraRank = 1536;
    int kvLoraRank = 512;
    int headNum = 128;
    int qkNopeHeadDim = 128;
    int qkRopeHeadDim = 64;
    bool enableMlaPreprocess = false;
    bool enableExtraOprojTp = false;
    bool isNzCache = false;

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
    bool isFA = true;
    bool isPrefill = false;
    int headDim = 0;
    atb::infer::SelfAttentionParam selfAttentionParam;
    atb::infer::RingMLAParam ringMLAParam;
    atb::infer::PagedAttentionParam pageAttentionParam;
    atb::infer::ReshapeAndCacheParam reshapeCacheParm;
    // self out linear param
    bool selfAttnHasBias = false;
    bool enableLcoc = false;
    int denseQuantType = atb_speed::common::PackQuantType::PACK_QUANT_UNDEFINED;
    atb_speed::common::TensorParallelInfo selfOutLinearTensorParallelInfo;
    atb_speed::common::ParallelInfo selfOutLinearInnerTensorParallelInfo;
    // context parallelism
    atb_speed::common::ParallelInfo contextParallelInfo;
    atb_speed::common::ParallelInfo prefixcacheContextParallelInfo;
    // sequence parallelism
    bool hasAttnInnerSp = false;
    int attnSpRank = 0;
    int attnSpSize = 1;
    std::string attnSpDomain = "";
    std::string attnSpRankTableFile = "";
    std::string attnSpBackend = "";
    HcclComm attnSpHcclComm = nullptr;
    bool attnOprojPrefetch = false;
    // h3p qkvdown dp
    int layerId = 0;
    int firstKDenseReplace = 0;
    bool isDenseLayer = false;
    bool enableQkvdownDp = false;
    bool hasAttnComm = false;
    bool hasFfnComm = false;
    int attnTpRank = 0;
    int attnTpSize = 1;
    std::string attnTpBackend = "";
    std::string attnTpDomain = "";
    std::string attnTpRankTableFile = "";
    HcclComm hcclComm = nullptr;
    bool ffnAllGather = false;

    bool enableOutLcocTp = false;
    bool enablePreprocessLcocTp = false;
    int lcocAttnTpRank = 0;
    int lcocAttnTpRankSize = 1;
    std::string lcocAttnTpBackend = "";
    std::string lcocAttnTpDomain = "";
    HcclComm lcocHcclComm = nullptr;
    bool enablePrefixCache = false;
    bool enableFusedMLA = false;
};

template <typename NormParamType>
std::map<std::string, uint32_t> ConstructTensorMap(
    uint32_t &inTensorNum, uint32_t &outTensorNum, uint32_t &internalTensorNum);

template <typename NormParamType>
atb::Status Attention(const LatentAttentionParam<NormParamType> &param, atb::Operation **operation);
} // namespace common
} // namespace atb_speed
#endif