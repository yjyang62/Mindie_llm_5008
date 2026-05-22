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

#ifndef ATB_SPEED_PLUGIN_ACLNN_W16A16SC_OPERATION_H
#define ATB_SPEED_PLUGIN_ACLNN_W16A16SC_OPERATION_H
#include "operations/aclnn/core/acl_nn_operation.h"
#include "operations/aclnn/core/acl_nn_operation_cache.h"
#include "aclnnop/aclnn_matmul_compress.h"

namespace atb_speed {
namespace common {

/// A struct defines `W16A16SCOperation`'s parameter.
struct AclNNW16A16SCParam {
    /// A flag indicating whether the matmul operation includes a bias tensor.
    bool hasBias = true;
};

/// This class defines a matrix operation combines the matmul and add bias operation.
///
/// This class makes use of `aclnnMatmulCompressGetWorkspaceSize` and `aclnnMatmulCompress` from the AscendCL API.
///
/// Operation's Inputs:
/// Name            | Dtype                       | Shape | Description |
/// ----------------|-----------------------------|-------|-------------|
/// input           | FLOAT16                     | [m,k] | |
/// weight          | FLOAT16                     | [x] | |
/// bias            | FLOAT                       | [n] | |
/// compressIndex   | INT8                        | [\] | |indexes for non-zero data|
///
/// Operations's Outputs:
/// Name   | Dtype                              | Shape |
/// -------|------------------------------------|-------|
/// out    | FLOAT16                            | [m,n] |


class W16A16SCOperation : public AclNNOperation {
public:
    explicit W16A16SCOperation(const std::string &name, AclNNW16A16SCParam param);
    ~W16A16SCOperation() override;
    atb::Status InferShape(const atb::SVector<atb::TensorDesc> &inTensorDescs,
        atb::SVector<atb::TensorDesc> &outTensorDescs) const override;
    uint32_t GetInputNum() const override;
    uint32_t GetOutputNum() const override;

private:
    int SetAclNNWorkspaceExecutor() override;
    int ExecuteAclNNOp(uint8_t *workspace, aclrtStream &stream) override;
    atb::Status CreateAclNNInTensorVariantPack(const atb::VariantPack &variantPack) override;
    atb::Status CreateAclNNOutTensorVariantPack(const atb::VariantPack &variantPack) override;

    atb::Dims GetWeightStorageShape(const atb::TensorDesc atbTensorDesc) const;

    AclNNW16A16SCParam param_;
};
}  // namespace common
}  // namespace atb_speed
#endif  // ATB_SPEED_PUBLIC_ACLNN_W16A16SC_OPERATION_H