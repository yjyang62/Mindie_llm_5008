/*
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
#include <atb/atb_infer.h>

namespace atb {

template <> Status CreateOperation(const infer::ActivationParam &param, Operation **op)
{
    return NO_ERROR;
}

template <> Status CreateOperation(const infer::AllGatherParam &param, Operation **op)
{
    return NO_ERROR;
}

template <> Status CreateOperation(const infer::AllGatherVParam &param, Operation **op)
{
    return NO_ERROR;
}

template <> Status CreateOperation(const infer::AllReduceParam &param, Operation **op)
{
    return NO_ERROR;
}

template <> Status CreateOperation(const infer::AllToAllParam &param, Operation **op)
{
    return NO_ERROR;
}

template <> Status CreateOperation(const infer::AllToAllVParam &param, Operation **op)
{
    return NO_ERROR;
}

template <> Status CreateOperation(const infer::AllToAllVV2Param &param, Operation **op)
{
    return NO_ERROR;
}

template <> Status CreateOperation(const infer::AsStridedParam &param, Operation **op)
{
    return NO_ERROR;
}

template <> Status CreateOperation(const infer::BlockCopyParam &param, Operation **op)
{
    return NO_ERROR;
}

template <> Status CreateOperation(const infer::BroadcastParam &param, Operation **op)
{
    return NO_ERROR;
}

template <> Status CreateOperation(const infer::CohereLayerNormParam &param, Operation **op)
{
    return NO_ERROR;
}

template <> Status CreateOperation(const infer::CumsumParam &param, Operation **op)
{
    return NO_ERROR;
}

template <> Status CreateOperation(const infer::DynamicNTKParam &param, Operation **op)
{
    return NO_ERROR;
}

template <> Status CreateOperation(const infer::ElewiseParam &param, Operation **op)
{
    return NO_ERROR;
}

template <> Status CreateOperation(const infer::FaUpdateParam &param, Operation **op)
{
    return NO_ERROR;
}

template <> Status CreateOperation(const infer::FillParam &param, Operation **op)
{
    return NO_ERROR;
}

template <> Status CreateOperation(const infer::FusedAddTopkDivParam &param, Operation **op)
{
    return NO_ERROR;
}

template <> Status CreateOperation(const infer::GatherParam &param, Operation **op)
{
    return NO_ERROR;
}

template <> Status CreateOperation(const infer::GatherPreRmsNormParam &param, Operation **op)
{
    return NO_ERROR;
}

template <> Status CreateOperation(const infer::GmmDeqSwigluQuantGmmDeqParam &param, Operation **op)
{
    return NO_ERROR;
}

template <> Status CreateOperation(const infer::GroupedMatmulInplaceAddParam &param, Operation **op)
{
    return NO_ERROR;
}

template <> Status CreateOperation(const infer::GroupedMatmulWithRoutingParam &param, Operation **op)
{
    return NO_ERROR;
}

template <> Status CreateOperation(const infer::IndexAddParam &param, Operation **op)
{
    return NO_ERROR;
}

template <> Status CreateOperation(const infer::KvCacheParam &param, Operation **op)
{
    return NO_ERROR;
}

template <> Status CreateOperation(const infer::LayerNormParam &param, Operation **op)
{
    return NO_ERROR;
}

template <> Status CreateOperation(const infer::LayerNormWithStrideParam &param, Operation **op)
{
    return NO_ERROR;
}

template <> Status CreateOperation(const infer::LinearParam &param, Operation **op)
{
    return NO_ERROR;
}

template <> Status CreateOperation(const infer::LinearParallelParam &param, Operation **op)
{
    return NO_ERROR;
}

template <> Status CreateOperation(const infer::LinearSparseParam &param, Operation **op)
{
    return NO_ERROR;
}

template <> Status CreateOperation(const infer::MlaPreprocessParam &param, Operation **op)
{
    return NO_ERROR;
}

template <> Status CreateOperation(const infer::MmDeqSwigluQuantMmDeqParam &param, Operation **op)
{
    return NO_ERROR;
}

template <> Status CreateOperation(const infer::MultiLatentAttentionParam &param, Operation **op)
{
    return NO_ERROR;
}

template <> Status CreateOperation(const infer::MultinomialParam &param, Operation **op)
{
    return NO_ERROR;
}

template <> Status CreateOperation(const infer::NonzeroParam &param, Operation **op)
{
    return NO_ERROR;
}

template <> Status CreateOperation(const infer::NormRopeReshapeParam &param, Operation **op)
{
    return NO_ERROR;
}

template <> Status CreateOperation(const infer::OnehotParam &param, Operation **op)
{
    return NO_ERROR;
}

template <> Status CreateOperation(const infer::PadParam &param, Operation **op)
{
    return NO_ERROR;
}

template <> Status CreateOperation(const infer::PagedAttentionParam &param, Operation **op)
{
    return NO_ERROR;
}

template <> Status CreateOperation(const infer::PagedCacheLoadParam &param, Operation **op)
{
    return NO_ERROR;
}

template <> Status CreateOperation(const infer::RazorFusionAttentionParam &param, Operation **op)
{
    return NO_ERROR;
}

template <> Status CreateOperation(const infer::RecvParam &param, Operation **op)
{
    return NO_ERROR;
}

template <> Status CreateOperation(const infer::ReduceParam &param, Operation **op)
{
    return NO_ERROR;
}

template <> Status CreateOperation(const infer::ReduceScatterParam &param, Operation **op)
{
    return NO_ERROR;
}

template <> Status CreateOperation(const infer::ReduceScatterVParam &param, Operation **op)
{
    return NO_ERROR;
}

template <> Status CreateOperation(const infer::RelayAttentionParam &param, Operation **op)
{
    return NO_ERROR;
}

template <> Status CreateOperation(const infer::RepeatParam &param, Operation **op)
{
    return NO_ERROR;
}

template <> Status CreateOperation(const infer::ReshapeAndCacheParam &param, Operation **op)
{
    return NO_ERROR;
}

template <> Status CreateOperation(const infer::ReshapeAndCacheOmniParam &param, Operation **op)
{
    return NO_ERROR;
}

template <> Status CreateOperation(const infer::ReshapeAndCacheWithStrideParam &param, Operation **op)
{
    return NO_ERROR;
}

template <> Status CreateOperation(const infer::RingMLAParam &param, Operation **op)
{
    return NO_ERROR;
}

template <> Status CreateOperation(const infer::RmsNormParam &param, Operation **op)
{
    return NO_ERROR;
}

template <> Status CreateOperation(const infer::RmsNormWithStrideParam &param, Operation **op)
{
    return NO_ERROR;
}

template <> Status CreateOperation(const infer::RopeParam &param, Operation **op)
{
    return NO_ERROR;
}

template <> Status CreateOperation(const infer::RopeQConcatParam &param, Operation **op)
{
    return NO_ERROR;
}

template <> Status CreateOperation(const infer::ScatterElementsV2Param &param, Operation **op)
{
    return NO_ERROR;
}

template <> Status CreateOperation(const infer::SelfAttentionParam &param, Operation **op)
{
    return NO_ERROR;
}

template <> Status CreateOperation(const infer::SendParam &param, Operation **op)
{
    return NO_ERROR;
}

template <> Status CreateOperation(const infer::SetValueParam &param, Operation **op)
{
    return NO_ERROR;
}

template <> Status CreateOperation(const infer::SliceParam &param, Operation **op)
{
    return NO_ERROR;
}

template <> Status CreateOperation(const infer::SoftmaxParam &param, Operation **op)
{
    return NO_ERROR;
}

template <> Status CreateOperation(const infer::SortParam &param, Operation **op)
{
    return NO_ERROR;
}

template <> Status CreateOperation(const infer::SplitParam &param, Operation **op)
{
    return NO_ERROR;
}

template <> Status CreateOperation(const infer::SwigluQuantParam &param, Operation **op)
{
    return NO_ERROR;
}

template <> Status CreateOperation(const infer::TransdataParam &param, Operation **op)
{
    return NO_ERROR;
}

template <> Status CreateOperation(const infer::TransposeParam &param, Operation **op)
{
    return NO_ERROR;
}

template <> Status CreateOperation(const infer::UnpadParam &param, Operation **op)
{
    return NO_ERROR;
}

template <> Status CreateOperation(const infer::WhereParam &param, Operation **op)
{
    return NO_ERROR;
}
} // namespace atb