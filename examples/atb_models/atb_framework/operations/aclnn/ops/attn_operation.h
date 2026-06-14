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

#ifndef ATTN_OPERATION_H
#define ATTN_OPERATION_H
#include "operations/aclnn/core/acl_nn_operation.h"
#include "atb_speed/utils/operation_util.h"
#include "cstring"
namespace atb_speed {
namespace common {
struct AclNNAttnParam {
    /// A flag indicating whether the model use mask
    bool hasMask = false;
    /// A flag indicating whether the model use FA
    bool isFA = false;
    /// A flag indicating whether the model prefills
    bool isPrefill = false;
    /// A flag indicating whether the model is kvcache int8 compressed
    bool hasKVQuant = false;
    /// A flag indicating whether the model has kvcache compressed offset weight
    bool hasQuantOffset = false;
    /// enable Prefix Attn
    bool enablePrefixAttn = false;
    /// the number of head
    int64_t headNum = 0;
    /// the number of kvHead
    int64_t kvHeadNum = 0;
    /// the number of headDim
    int64_t headDim = 0;
    /// represent high performance/accuracy, dafault 1 (high performance)
    int64_t innerPrecise = 1;
    /// max number of tokens in each block page attention stored in KV cache
    int64_t blockSize = 128;
    std::string inputLayoutPA = "BSND";
};

/// This class defines an operator that calculates the attention including FA and PA.
///
/// This class makes uses of `aclnnFusedInferAttentionScoreV2GetWorkspaceSize` and
/// `aclnnFusedInferAttentionScoreV2` from AscendCL Api.
///
/// Inputs to the operator:
/// Name                      | Dtype                       | Shape                               |
/// --------------------------|-----------------------------|-------------------------------------|
/// input                     | *                           | [batchsize, headNum, dim]           |
/// query                     | float16, bfloat16 or int8   | [batchsize, headNum, dim]           |
/// key                       | float16, bfloat16 or int8   | [blocknum, blocksize, headNum, dim] |
/// value                     | float16, bfloat16 or int8   | [blocknum, blocksize, headNum, dim] |
/// actualSeqLengthsOptional  | int64                       | [bs]                                |
/// blockTableOptional        | float16, bfloat16 or float32| [bs,blocknum]                       |
/// antiquantScaleOptional    | float16, bfloat16 or float32| [bs,dim]                            |
/// antiquantOffsetOptional   | float16, bfloat16 or float32| [bs,dim]                            |
///
/// Outputs of the operator:
/// Name                      | Dtype                       | Shape                               |
/// --------------------------|-----------------------------|-------------------------------------|
/// output                    | float16, bfloat16 or int8   | [batchsize, headNum, dim]           |
///
/// Example:
/// \code
/// enum TensorIdx : uint32_t {
///    QUERY,
///    KEY,
///    VALUE,
///    SEQ_LEN,
///    BLOCK_TABLE,
///    DEQUANT_SCALE,
///    DEQUANT_OFFSET,
///    OUT,
///};
///
/// atb::Node &attnNode = opGraph.nodes.at(nodeId++);
/// attnNode.operation = new atb_speed::common::AttnOperation("AttentionNode");
/// attnNode.inTensorIds = {QUERY, KEY, VALUE, SEQ_LEN, BLOCK_TABLE, DEQUANT_SCALE, DEQUANT_OFFSET};
/// attnNode.outTensorIds = {OUT};
/// \endcode

class AttnOperation : public AclNNOperation {
public:
    explicit AttnOperation(const std::string &name, AclNNAttnParam param);
    ~AttnOperation() override;
    atb::Status InferShape(const atb::SVector<atb::TensorDesc> &inTensorDescs,
                           atb::SVector<atb::TensorDesc> &outTensorDescs) const override;
    uint32_t GetInputNum() const override;
    uint32_t GetOutputNum() const override;

protected:
    int SetAclNNWorkspaceExecutor() override;
    int ExecuteAclNNOp(uint8_t *workspace, aclrtStream &stream) override;
    atb::Status CreateAclNNInTensorVariantPack(const atb::VariantPack &variantPack) override;
    atb::Status CreateAclNNOutTensorVariantPack(const atb::VariantPack &variantPack) override;

private:
    int ProcessSeqLengthTensor(atb::Tensor &tensor);

private:
    aclTensor *tensorsOfValue[1]{nullptr};
    aclTensor *tensorsOfKey[1]{nullptr};
    AclNNAttnParam param_;
    std::string opName_;
};
} // namespace common
} // namespace atb_speed
#endif
