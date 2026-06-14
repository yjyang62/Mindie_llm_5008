/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * MindIE is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

#include <gtest/gtest.h>
#include <mockcpp/mockcpp.hpp>
#include <atb/atb_infer.h>
#include "models/deepseekv2/operation/latent_attention.h"
#include "operations/fusion/linear/linear_parallel.h"
#include "models/base/param/mapping.h"
#include "atb_speed/utils/singleton.h"
#include "atb_speed/base/external_comm_manager.h"
#include "operations/fusion/utils.h"

namespace atb_speed {
bool CheckGraphParamForLatentAttentionCase(const atb::GraphParam &opGraph)
{
atb::SVector<atb::TensorDesc> inTensorDescs, outTensorDescs;
    inTensorDescs.reserve(80);
    inTensorDescs.resize(80);
    outTensorDescs.reserve(1);
    outTensorDescs.resize(1);
    atb::Status ret = opGraph.inferShapeFunc(inTensorDescs, outTensorDescs);
    return true;
}

bool CheckGraphParamForLatentAttentionCase1(const atb::GraphParam &opGraph)
{
    atb::SVector<atb::TensorDesc> inTensorDescs, outTensorDescs;
    inTensorDescs.reserve(80);
    inTensorDescs.resize(80);
    outTensorDescs.reserve(1);
    outTensorDescs.resize(1);
    atb::Dims oldShape, newShape;
    oldShape.dimNum = 4;
    for (size_t i = 0; i < oldShape.dimNum; ++i) {
        oldShape.dims[i] = 2;
    }
    opGraph.nodes[9].inTensorReshapeFuncs[1](oldShape, newShape); // 9: decoder selfAttentionNode
    oldShape.dimNum = 2;
    opGraph.nodes[9].inTensorReshapeFuncs[1](oldShape, newShape); // 9: decoder selfAttentionNode
    return true;
}

void SetLatentAttentionParam(atb_speed::common::LatentAttentionParam<atb::infer::RmsNormParam> &param)
{
    param.attnLinearQuantType.push_back(6);
    param.attnLinearQuantType.push_back(0);
    param.attnLinearQuantType.push_back(0);
    param.attnLinearQuantType.push_back(0);
    param.attnLinearQuantType.push_back(0);
    param.attnLinearQuantType.push_back(0);
    param.attnLinearTransposeType.push_back(0);
    param.attnLinearTransposeType.push_back(0);
    param.attnLinearTransposeType.push_back(0);
    param.attnLinearTransposeType.push_back(0);
    param.attnLinearTransposeType.push_back(0);
    param.attnLinearTransposeType.push_back(0);
}

TEST(dsv2LatentAttentionTest1, latentAttention)
{
    GlobalMockObject::verify(); // 清空mock
    GetSingleton<ExternalCommManager>().lcclCommDomainUpperBound_ = 1;
    atb_speed::common::LatentAttentionParam<atb::infer::RmsNormParam> param;
    SetLatentAttentionParam(param);
    atb::Operation *op = nullptr;
    atb_speed::common::ParallelInfo info;
    info.rankIds = {0, 1};
    info.bufferSize = 500;
    param.selfOutLinearInnerTensorParallelInfo = info;
    param.contextParallelInfo = info;
    param.isPrefill = false;
    MOCKER(atb_speed::common::FusionLinear).expects(atLeast(1))
    .with(any(), any()).will(returnValue(0));
    MOCKER(atb_speed::common::LinearParallel).expects(atLeast(1))
    .with(any(), any()).will(returnValue(0));
    MOCKER(atb::CreateOperation<atb::GraphParam>).expects(atLeast(1))
    .with(checkWith(CheckGraphParamForLatentAttentionCase1), any()).will(returnValue(0));
    MOCKER(atb_speed::common::GetTensorIdx).expects(atLeast(1)).with(any(), any()).will(returnValue(0));
    MOCKER(atb_speed::common::AddTensorToList<std::vector<std::string>>).expects(atLeast(1)).with(any(), any());
    EXPECT_EQ(atb_speed::common::Attention(param, &op), 0);
}

TEST(dsv2LatentAttentionTest4, latentAttention)
{
    GlobalMockObject::verify(); // 清空mock
    GetSingleton<ExternalCommManager>().lcclCommDomainUpperBound_ = 1;
    atb_speed::common::LatentAttentionParam<atb::infer::RmsNormParam> param;
    SetLatentAttentionParam(param);
    atb::Operation *op = nullptr;
    atb_speed::common::ParallelInfo info;
    info.rankIds = {0, 1};
    info.bufferSize = 500;
    param.selfOutLinearInnerTensorParallelInfo = info;
    param.isPrefill = true;
    param.enableFusedMLA = true;
    param.enableMlaPreprocess = false;
    param.rotaryType = atb_speed::common::RotaryType::HALF_ROTARY;
    MOCKER(atb_speed::common::FusionLinear).expects(atLeast(1))
    .with(any(), any()).will(returnValue(0));
    MOCKER(atb_speed::common::LinearParallel).expects(atLeast(1))
    .with(any(), any()).will(returnValue(0));
    MOCKER(atb::CreateOperation<atb::GraphParam>).expects(atLeast(1))
    .with(any(), any()).will(returnValue(0));
    std::map<std::string, std::vector<std::string>>  attnInTensorCandidates;
    MOCKER(atb_speed::common::GetTensorIdx).expects(atLeast(1)).with(any(), any())
    .will(returnValue(0));
    MOCKER(atb_speed::common::AddTensorToList<std::vector<std::string>>).expects(atLeast(1)).with(any(), any());
    EXPECT_EQ(atb_speed::common::Attention(param, &op), 0);
}

TEST(dsv2LatentAttentionTest, latentAttention)
{
    GlobalMockObject::verify(); // 清空mock
    GetSingleton<ExternalCommManager>().lcclCommDomainUpperBound_ = 1;
    atb_speed::common::LatentAttentionParam<atb::infer::RmsNormParam> param;
    SetLatentAttentionParam(param);
    atb::Operation *op = nullptr;
    atb_speed::common::ParallelInfo info;
    info.rankIds = {0, 1};
    info.bufferSize = 500;
    param.selfOutLinearInnerTensorParallelInfo = info;
    param.contextParallelInfo = info;
    param.enablePrefixCache = true;
    param.hasAttnInnerSp = true;
    MOCKER(atb_speed::common::GetTensorIdx).expects(atLeast(1)).with(any(), any())
    .will(returnValue(0));
    MOCKER(atb_speed::common::AddTensorToList<std::vector<std::string>>).expects(atLeast(1)).with(any(), any());
    MOCKER(atb::CreateOperation<atb::GraphParam>).expects(atLeast(1))
    .with(checkWith(CheckGraphParamForLatentAttentionCase), any()).will(returnValue(0));
    EXPECT_EQ(atb_speed::common::Attention(param, &op), 0);
    param.enableQkvdownDp = true;
    param.enableMlaPreprocess = true;
    param.isPrefill = false;
    param.pageAttentionParam.quantType = atb::infer::PagedAttentionParam::QuantType::TYPE_QUANT_QKV_ONLINE;
    EXPECT_EQ(atb_speed::common::Attention(param, &op), 0);
    param.isPrefill = true;
    EXPECT_EQ(atb_speed::common::Attention(param, &op), 0);
    param.isPrefill = true;
    param.enableFusedMLA = true;
    param.enablePrefixCache = false;
    EXPECT_EQ(atb_speed::common::Attention(param, &op), 0);
    param.enablePrefixCache = false;
    param.enableFusedMLA = false;
    param.isPrefill = true;
    param.enableOutLcocTp = true;
    param.pageAttentionParam.quantType = atb::infer::PagedAttentionParam::QuantType::TYPE_DEQUANT_FUSION;
    param.isNzCache = true;
    param.isFA = false;
    param.attnOprojPrefetch = true;
    param.enablePreprocessLcocTp = true;
    param.ffnAllGather = true;
    param.selfAttentionParam.maskType = atb::infer::SelfAttentionParam::MaskType::MASK_TYPE_NORM;
    param.pageAttentionParam.calcType = atb::infer::PagedAttentionParam::CalcType::CALC_TYPE_SPEC;
    param.rotaryType = atb_speed::common::RotaryType::HALF_ROTARY;
    EXPECT_EQ(atb_speed::common::Attention(param, &op), 0);
    param.qLoraRank = 0;
    param.enablePrefixCache = false;
    info.rankIds = {0};
    param.contextParallelInfo = info;
    param.enableFusedMLA = true;
    param.isPrefill = true;
    param.isFA = true;
    EXPECT_EQ(atb_speed::common::Attention(param, &op), 0);
    param.enableFusedMLA = false;
    param.attnLinearQuantType[1] = 6;
    param.packQuantType = atb_speed::common::LinearType::INT;
    EXPECT_EQ(atb_speed::common::Attention(param, &op), 0);
    param.isPrefill = false;
    param.qLoraRank = 18;
    param.isNzCache = false;
    param.enableExtraOprojTp = true;
    info.rankIds = {};
    param.pageAttentionParam.maskType = atb::infer::PagedAttentionParam::MaskType::MASK_TYPE_NORM;
    param.pageAttentionParam.calcType = atb::infer::PagedAttentionParam::CalcType::CALC_TYPE_SPEC;
    EXPECT_EQ(atb_speed::common::Attention(param, &op), 0);
    param.isNzCache = true;
    param.pageAttentionParam.maskType = atb::infer::PagedAttentionParam::MaskType::MASK_TYPE_SPEC;
    EXPECT_EQ(atb_speed::common::Attention(param, &op), 0);
    param.enableFusedMLA = false;
    param.attnLinearQuantType[0] = 0;
    param.enableMlaPreprocess = false;
    param.enablePreprocessLcocTp = false;
    param.enableOutLcocTp = false;
    EXPECT_EQ(atb_speed::common::Attention(param, &op), 0);
    param.enablePreprocessLcocTp = true;
    param.attnLinearQuantType[1] = 6;
    param.attnLinearQuantType[5] = 1;
    EXPECT_EQ(atb_speed::common::Attention(param, &op), 0);
    param.denseQuantType = 18;
    EXPECT_EQ(atb_speed::common::Attention(param, &op), 0);
    param.enablePreprocessLcocTp = false;
    param.isFA = false;
    param.attnLinearQuantType[0] = 1;
    EXPECT_EQ(atb_speed::common::Attention(param, &op), 0);
    param.selfOutLinearInnerTensorParallelInfo = info;
    EXPECT_THROW(atb_speed::common::Attention(param, &op), std::runtime_error);
}
} // namespace atb_speed

