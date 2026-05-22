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
#include <securec.h>
#include "operations/fusion/utils.h"
#include "operations/fusion/parallel_info.h"
#include "operations/aclnn/ops/attn_operation.h"
#include "models/deepseekv2/operation/ring_attention.h"

namespace atb_speed {
namespace common {


template <typename NormParamType>
atb::Status AddSelfAttnNode(
    const LatentAttentionParam<NormParamType> &param, atb::GraphParam &opGraph,
    std::map<std::string, uint32_t> &tensorMap, uint32_t qIdx, uint32_t kvIdx)
{
    atb::Node selfAttentionNode;
    atb::infer::RingMLAParam ringMLAParam = param.ringMLAParam;
    // The first RingAttn computation (with kv0) does not require passing o and lse,
    // while in other cases they should be passed by default.
    ringMLAParam.calcType = kvIdx == 0 ?
                                  atb::infer::RingMLAParam::CalcType::CALC_TYPE_FISRT_RING :
                                  atb::infer::RingMLAParam::CalcType::CALC_TYPE_DEFAULT;
    // Masking is required when computing Qi and KVi, but not needed in other cases.
    ringMLAParam.maskType = qIdx == kvIdx ?
                                  atb::infer::RingMLAParam::MaskType::MASK_TYPE_TRIU :
                                  atb::infer::RingMLAParam::MaskType::NO_MASK;

    bool isQFirst = qIdx < param.contextParallelInfo.rankIds.size();
    bool isKVFirst = kvIdx < param.contextParallelInfo.rankIds.size();
    std::string v = isKVFirst ? "intermediate_v_first" : "intermediate_v_last";
    std::string o = isQFirst ? "intermediate_o_first" : "intermediate_o_last";
    std::string lse = isQFirst ? "intermediate_lse_first" : "intermediate_lse_last";

    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(ringMLAParam, &selfAttentionNode.operation));
    selfAttentionNode.inTensorIds = {
        GetTensorIdx(tensorMap, "intermediate_q_nope_cp"),
        GetTensorIdx(tensorMap, "intermediate_q_rope_cp"),
        GetTensorIdx(tensorMap, "intermediate_k_nope_cp"),
        GetTensorIdx(tensorMap, "intermediate_k_rope_cp"),
        GetTensorIdx(tensorMap, v),
        GetTensorIdx(tensorMap, "in_attention_mask"),
        GetTensorIdx(tensorMap, "in_seq_len_cp"),
    };
    if (ringMLAParam.calcType == atb::infer::RingMLAParam::CalcType::CALC_TYPE_DEFAULT) {
        selfAttentionNode.inTensorIds.push_back(GetTensorIdx(tensorMap, o));
        selfAttentionNode.inTensorIds.push_back(GetTensorIdx(tensorMap, lse));
    }
    selfAttentionNode.outTensorIds = GetTensorIdxList(tensorMap, {o, lse});
    opGraph.nodes.push_back(selfAttentionNode);
    return atb::NO_ERROR;
}

template <typename NormParamType>
atb::Status AddSelfAttnPrefixNode(
    const LatentAttentionParam<NormParamType> &param, atb::GraphParam &opGraph,
    std::map<std::string, uint32_t> &tensorMap, bool isQFirst)
{
    atb::Node selfAttentionPrefixNode;
    atb::infer::RingMLAParam ringMLAParam = param.ringMLAParam;
    ringMLAParam.calcType = atb::infer::RingMLAParam::CalcType::CALC_TYPE_DEFAULT;
    // No Masking when computing Qi and KVcache.
    ringMLAParam.maskType = atb::infer::RingMLAParam::MaskType::NO_MASK;

    std::string o_history = isQFirst ? "intermediate_o_first" : "intermediate_o_last";
    std::string lse_history = isQFirst ? "intermediate_lse_first" : "intermediate_lse_last";

    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(ringMLAParam, &selfAttentionPrefixNode.operation));
    selfAttentionPrefixNode.inTensorIds = {
        GetTensorIdx(tensorMap, "intermediate_q_nope_cp"),
        GetTensorIdx(tensorMap, "intermediate_q_rope_cp"),
        GetTensorIdx(tensorMap, "intermediate_k_nope_history"),
        GetTensorIdx(tensorMap, "rope_k_o_repeat_history"),
        GetTensorIdx(tensorMap, "intermediate_v_mha_history"),
        GetTensorIdx(tensorMap, "in_attention_mask"),
        GetTensorIdx(tensorMap, "in_kv_cache_len"),
        GetTensorIdx(tensorMap, o_history),
        GetTensorIdx(tensorMap, lse_history),
    };
    selfAttentionPrefixNode.outTensorIds = GetTensorIdxList(tensorMap, {o_history, lse_history});
    selfAttentionPrefixNode.inTensorReshapeFuncs.resize(selfAttentionPrefixNode.inTensorIds.size());
    selfAttentionPrefixNode.inTensorReshapeFuncs[2] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = 3; // 3: dim num
        newShape.dims[0] = oldShape.dims[0];
        newShape.dims[1] = param.selfAttentionParam.headNum;
        newShape.dims[2] = param.qkNopeHeadDim; // 2: dim id
    };
    selfAttentionPrefixNode.inTensorReshapeFuncs[4] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = 3; // 3: dim num
        newShape.dims[0] = oldShape.dims[0];
        newShape.dims[1] = param.selfAttentionParam.headNum;
        newShape.dims[2] = param.qkNopeHeadDim; // 2: dim id
    };
    opGraph.nodes.push_back(selfAttentionPrefixNode);
    ATB_SPEED_LOG_DEBUG("MLA SelfAttnPrefixNode calculation success");
    return atb::NO_ERROR;
}


template <typename NormParamType>
atb::Status AddPaEncoderLBNode(
    const LatentAttentionParam<NormParamType> &param, atb::GraphParam &opGraph,
    std::map<std::string, uint32_t> &tensorMap, uint32_t qIdx, uint32_t kvIdx)
{
    if (qIdx < kvIdx) {
        // q_i is only calculated with  kv_0 to kv_i
        return atb::NO_ERROR;
    }

    bool isQFirst = qIdx < param.contextParallelInfo.rankIds.size();
    bool isKVFirst = kvIdx < param.contextParallelInfo.rankIds.size();
    std::string q = isQFirst ? "intermediate_q_first" : "intermediate_q_last";
    std::string kvGatherIdx = isKVFirst ? "in_cp_load_balance_idx_first" : "in_cp_load_balance_idx_last";
    std::string k = isKVFirst ? "intermediate_k_first" : "intermediate_k_last";

    atb::Node splitQNode;
    atb::infer::SplitParam splitQParam;
    splitQParam.splitDim = 2; // 2: dim num
    splitQParam.splitSizes = {param.qkNopeHeadDim, param.qkRopeHeadDim};
    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(splitQParam, &splitQNode.operation));
    splitQNode.inTensorIds = {GetTensorIdx(tensorMap, q)};
    splitQNode.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_q_nope_cp"),
                               GetTensorIdx(tensorMap, "intermediate_q_rope_cp")};
    opGraph.nodes.push_back(splitQNode);

    atb::Node splitKNode;
    atb::infer::SplitParam splitKParam;
    splitKParam.splitDim = 2; // 2: dim num
    splitKParam.splitSizes = {param.qkNopeHeadDim, param.qkRopeHeadDim};
    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(splitKParam, &splitKNode.operation));
    splitKNode.inTensorIds = {GetTensorIdx(tensorMap, k)};
    splitKNode.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_k_nope_cp"),
                              GetTensorIdx(tensorMap, "intermediate_k_rope_cp")};
    opGraph.nodes.push_back(splitKNode);

    CHECK_OPERATION_STATUS_RETURN(AddSelfAttnNode(param, opGraph, tensorMap, qIdx, kvIdx));

    ATB_SPEED_LOG_DEBUG("PA encoder calculation success");
    return atb::NO_ERROR;
}


template <typename NormParamType>
atb::Status AddPaEncoderLBPrefixNode(
    const LatentAttentionParam<NormParamType> &param, atb::GraphParam &opGraph,
    std::map<std::string, uint32_t> &tensorMap, bool isQFirst)
{
    std::string q = isQFirst ? "intermediate_q_first" : "intermediate_q_last";

    atb::Node splitQNode;
    atb::infer::SplitParam splitQParam;
    splitQParam.splitDim = 2; // 2: dim num
    splitQParam.splitSizes = {param.qkNopeHeadDim, param.qkRopeHeadDim};
    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(splitQParam, &splitQNode.operation));
    splitQNode.inTensorIds = {GetTensorIdx(tensorMap, q)};
    splitQNode.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_q_nope_cp"),
                               GetTensorIdx(tensorMap, "intermediate_q_rope_cp")};
    opGraph.nodes.push_back(splitQNode);

    CHECK_OPERATION_STATUS_RETURN(AddSelfAttnPrefixNode(param, opGraph, tensorMap, isQFirst));

    ATB_SPEED_LOG_DEBUG("PA history encoder calculation success");
    return atb::NO_ERROR;
}


atb::Status AddQGatherNode(atb::GraphParam &opGraph, std::map<std::string, uint32_t> &tensorMap)
{
    atb::Node gatherQFirstNode;
    atb::infer::GatherParam gatherQFirstParam;
    atb::CreateOperation(gatherQFirstParam, &gatherQFirstNode.operation);
    gatherQFirstNode.inTensorIds =
        atb_speed::common::GetTensorIdxList(tensorMap, {"intermediate_q", "in_cp_load_balance_idx_first"});
    gatherQFirstNode.outTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, {"intermediate_q_first"});
    opGraph.nodes.push_back(gatherQFirstNode);

    atb::Node gatherQLastNode;
    atb::infer::GatherParam gatherQLastParam;
    atb::CreateOperation(gatherQLastParam, &gatherQLastNode.operation);
    gatherQLastNode.inTensorIds =
        atb_speed::common::GetTensorIdxList(tensorMap, {"intermediate_q", "in_cp_load_balance_idx_last"});
    gatherQLastNode.outTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, {"intermediate_q_last"});
    opGraph.nodes.push_back(gatherQLastNode);
    ATB_SPEED_LOG_DEBUG("MLA gather_q calculation success");

    return atb::NO_ERROR;
}


template <typename NormParamType>
atb::Status AddKVSliceNode(
    const LatentAttentionParam<NormParamType> &param, atb::GraphParam &opGraph,
    std::map<std::string, uint32_t> &tensorMap, uint32_t ringTimes)
{
    atb::Node sliceKNode;
    atb::infer::SliceParam sliceKParam;
    sliceKParam.offsets = {ringTimes, 0, 0, 0};
    sliceKParam.size = {1, -1, -1, -1};
    atb::CreateOperation(sliceKParam, &sliceKNode.operation);
    sliceKNode.inTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, {"intermediate_k_mha"});
    sliceKNode.outTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, {"intermediate_k_mha_cp"});
    sliceKNode.inTensorReshapeFuncs.resize(sliceKNode.inTensorIds.size());
    sliceKNode.inTensorReshapeFuncs[0] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        // [cp*bs,n/tp,d+dr] -> [cp,bs,n/tp,d+dr]
        newShape.dimNum = 4; // 4: dim num
        newShape.dims[0] = param.contextParallelInfo.rankIds.size();
        newShape.dims[1] = oldShape.dims[0] / param.contextParallelInfo.rankIds.size();
        newShape.dims[2] = oldShape.dims[1]; // 2: dim id
        newShape.dims[3] = oldShape.dims[2]; // 2, 3: dim id
    };
    opGraph.nodes.push_back(sliceKNode);

    atb::Node sliceVNode;
    atb::infer::SliceParam sliceVParam;
    sliceVParam.offsets = {ringTimes, 0, 0, 0};
    sliceVParam.size = {1, -1, -1, -1};
    atb::CreateOperation(sliceVParam, &sliceVNode.operation);
    sliceVNode.inTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, {"intermediate_v_mha"});
    sliceVNode.outTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, {"intermediate_v_mha_cp"});
    sliceVNode.inTensorReshapeFuncs.resize(sliceVNode.inTensorIds.size());
    sliceVNode.inTensorReshapeFuncs[0] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        // [cp*bs,n/tp*d] -> [cp,bs,n/tp,d]
        newShape.dimNum = 4; // 4: dim num
        newShape.dims[0] = param.contextParallelInfo.rankIds.size();
        newShape.dims[1] = oldShape.dims[0] / param.contextParallelInfo.rankIds.size();
        newShape.dims[2] = param.selfAttentionParam.headNum; // 2: dim id
        newShape.dims[3] = param.qkNopeHeadDim; // 3: dim id
    };
    opGraph.nodes.push_back(sliceVNode);

    return atb::NO_ERROR;
}


atb::Status AddKVGatherNode(atb::GraphParam &opGraph, std::map<std::string, uint32_t> &tensorMap,
    bool isKVFirst)
{
    std::string kvGatherIdx = isKVFirst ? "in_cp_load_balance_idx_first" : "in_cp_load_balance_idx_last";
    std::string k = isKVFirst ? "intermediate_k_first" : "intermediate_k_last";
    std::string v = isKVFirst ? "intermediate_v_first" : "intermediate_v_last";

    atb::Node gatherKNode;
    atb::infer::GatherParam gatherKParam;
    atb::CreateOperation(gatherKParam, &gatherKNode.operation);
    gatherKNode.inTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, {"intermediate_k_mha_cp", kvGatherIdx});
    gatherKNode.outTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, {k});
    gatherKNode.inTensorReshapeFuncs.resize(gatherKNode.inTensorIds.size());
    gatherKNode.inTensorReshapeFuncs[0] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = 3; // 3: dim num
        newShape.dims[0] = oldShape.dims[0] * oldShape.dims[1];
        newShape.dims[1] = oldShape.dims[2]; // 2: dim id
        newShape.dims[2] = oldShape.dims[3]; // 2, 3: dim id
    };
    opGraph.nodes.push_back(gatherKNode);

    atb::Node gatherVNode;
    atb::infer::GatherParam gatherVParam;
    atb::CreateOperation(gatherVParam, &gatherVNode.operation);
    gatherVNode.inTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, {"intermediate_v_mha_cp", kvGatherIdx});
    gatherVNode.outTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, {v});
    gatherVNode.inTensorReshapeFuncs.resize(gatherVNode.inTensorIds.size());
    gatherVNode.inTensorReshapeFuncs[0] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = 3; // 3: dim num
        newShape.dims[0] = oldShape.dims[0] * oldShape.dims[1];
        newShape.dims[1] = oldShape.dims[2]; // 2: dim id
        newShape.dims[2] = oldShape.dims[3]; // 2, 3: dim id
    };
    opGraph.nodes.push_back(gatherVNode);
    ATB_SPEED_LOG_DEBUG("MLA gather_kv calculation success");

    return atb::NO_ERROR;
}


template <typename NormParamType>
atb::Status AddRingAttentionLB(
    const LatentAttentionParam<NormParamType> &param, atb::GraphParam &opGraph,
    std::map<std::string, uint32_t> &tensorMap)
{
    CHECK_OPERATION_STATUS_RETURN(AddQGatherNode(opGraph, tensorMap));

    for (uint32_t ringTimes = 0; ringTimes < param.contextParallelInfo.rankIds.size(); ringTimes++) {
        CHECK_OPERATION_STATUS_RETURN(AddKVSliceNode(param, opGraph, tensorMap, ringTimes));

        // Retrieve the first half and the second half of the KV pairs.
        CHECK_OPERATION_STATUS_RETURN(AddKVGatherNode(opGraph, tensorMap, true));   // K0
        CHECK_OPERATION_STATUS_RETURN(AddKVGatherNode(opGraph, tensorMap, false));  // K3
        
        // rank=0, Q=[q0,q3], KV=[kv0,kv3,kv1,kv2]
        //  ring=0, Q=[q0,q3], KV=[kv0,kv3] ---> q0*kv0; q0*kv3; q3*kv0; q3*kv3
        //  ring=1, Q=[q0,q3], KV=[kv1,kv2] ---> q0*kv1; q0*kv2; q3*kv1; q3*kv2
        uint32_t qFirstIdx = param.contextParallelInfo.rank;
        uint32_t qLastIdx = param.contextParallelInfo.rankIds.size() * 2 - 1 - param.contextParallelInfo.rank;
        uint32_t kvFirstIdx = ringTimes;
        uint32_t kvLastIdx = param.contextParallelInfo.rankIds.size() * 2 - 1 - ringTimes;

        // The calculation of Q and KV for each segment.
        CHECK_OPERATION_STATUS_RETURN(AddPaEncoderLBNode(param, opGraph, tensorMap, qFirstIdx, kvFirstIdx));  // Q0*K0
        CHECK_OPERATION_STATUS_RETURN(AddPaEncoderLBNode(param, opGraph, tensorMap, qFirstIdx, kvLastIdx));   // Q0*K3
        CHECK_OPERATION_STATUS_RETURN(AddPaEncoderLBNode(param, opGraph, tensorMap, qLastIdx, kvFirstIdx));   // Q3*K0
        CHECK_OPERATION_STATUS_RETURN(AddPaEncoderLBNode(param, opGraph, tensorMap, qLastIdx, kvLastIdx));    // Q3*K3
    }

    // Prefixcache: Q * kvcache
    if (param.enablePrefixCache) {
        CHECK_OPERATION_STATUS_RETURN(AddPaEncoderLBPrefixNode(param, opGraph, tensorMap, true));  // Q0 * kv cache
        CHECK_OPERATION_STATUS_RETURN(AddPaEncoderLBPrefixNode(param, opGraph, tensorMap, false));   // Q3 * kv cache
    }

    atb::Node catNode;
    atb::infer::ConcatParam catParam;
    catParam.concatDim = 0;
    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(catParam, &catNode.operation));
    catNode.inTensorIds = {GetTensorIdxList(tensorMap, {"intermediate_o_first", "intermediate_o_last"})};
    catNode.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_o_concat")};
    opGraph.nodes.push_back(catNode);

    atb::Node gatherONode;
    atb::infer::GatherParam gatherOParam;
    atb::CreateOperation(gatherOParam, &gatherONode.operation);
    gatherONode.inTensorIds =
        atb_speed::common::GetTensorIdxList(tensorMap, {"intermediate_o_concat", "in_cp_o_recover_idx"});
    gatherONode.outTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, {"intermediate_self_attention"});
    opGraph.nodes.push_back(gatherONode);

    return atb::NO_ERROR;
}

template atb::Status AddRingAttentionLB(
    const LatentAttentionParam<atb::infer::RmsNormParam> &param, atb::GraphParam &opGraph,
    std::map<std::string, uint32_t> &tensorMap);

} // namespace common
} // namespace atb_speed