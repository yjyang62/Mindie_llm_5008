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

#ifndef ATB_SPEED_PLUGIN_ACLNN_MOE_TOPK_SOFTMAX_OPERATION_H
#define ATB_SPEED_PLUGIN_ACLNN_MOE_TOPK_SOFTMAX_OPERATION_H
#include "operations/aclnn/core/acl_nn_operation.h"
#include "atb_speed/utils/operation_util.h"

namespace atb_speed {
namespace common {

struct MoeTopkSoftmaxParam {
    /// The number of experts selected for each token
    int64_t topkNum = 2;
};

/// This class defines an operator that first applies softmax to each row of the input, and then
/// selects the top k greatest value.
///
/// This class makes uses of `aclnnMoeGatingTopKSoftmaxGetWorkspaceSize` and `aclnnMoeGatingTopKSoftmax`
/// from the AscendCL API.
///
/// Inputs to the operator:
/// Name            | Dtype                 | Shape |
/// ----------------|-----------------------|-------|
/// input           | float16 or bfloat16   | [m,e] |
///
/// Outputs of the operator:
/// Name            | Dtype                 | Shape |
/// ----------------|-----------------------|-------|
/// output          | float16 or bfloat16   | [m,k] |
/// expertIdx       | int32                 | [m,k] |
/// rowIdx          | int32                 | [m,k] |
///
/// Example:
/// \code
/// enum TensorIdx : uint32_t {
///     IN_INPUT = 0,
///     OUT,
///     OUT_EXPERTIDX,
///     OUT_ROWIDX
/// };
///
/// atb::Node &topKNode = opGraph.nodes.at(nodeId++);
/// atb_speed::common::MoeTopkSoftmaxParam moeTopkSoftmaxParam;
/// moeTopkSoftmaxParam.topkNum = int64_t(param.num.at(0));
/// topKNode.operation = new atb_speed::common::MoeTopkSoftmaxOperation("MoeTopkSoftmaxOperation", moeTopkSoftmaxParam);
/// topKNode.inTensorIds = {INPUT};
/// topKNode.outTensorIds = {OUT,
///                          OUT_EXPERTIDX,
///                          OUT_ROWIDX};
///
/// \endcode

class MoeTopkSoftmaxOperation : public AclNNOperation {
public:
    explicit MoeTopkSoftmaxOperation(const std::string &name, MoeTopkSoftmaxParam param);
    ~MoeTopkSoftmaxOperation() override;
    atb::Status InferShape(const atb::SVector<atb::TensorDesc> &inTensorDescs,
                           atb::SVector<atb::TensorDesc> &outTensorDescs) const override;
    uint32_t GetInputNum() const override;
    uint32_t GetOutputNum() const override;

private:
    int SetAclNNWorkspaceExecutor() override;
    int ExecuteAclNNOp(uint8_t *workspace, aclrtStream &stream) override;

    MoeTopkSoftmaxParam param_;
};

}  // namespace common
}  // namespace atb_speed
#endif  // ATB_SPEED_PLUGIN_ACLNN_MOE_TOPK_SOFTMAX_OPERATION_H