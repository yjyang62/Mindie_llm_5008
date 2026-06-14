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

#ifndef ATB_SPEED_PLUGIN_ACLRT_OPERATION_H
#define ATB_SPEED_PLUGIN_ACLRT_OPERATION_H
#include <string>
#include <acl/acl.h>
#include <aclnn/acl_meta.h>
#include <atb/atb_infer.h>
#include <atb/operation_infra.h>

namespace atb_speed {
namespace common {

class AclrtCmoAsyncOperation : public atb::OperationInfra {
public:
    explicit AclrtCmoAsyncOperation(const std::string &opName, size_t dataSize = 0);

    ~AclrtCmoAsyncOperation() override;

    std::string GetName() const override;
    atb::Status InferShape(const atb::SVector<atb::TensorDesc> &inTensorDesc,
                           atb::SVector<atb::TensorDesc> &outTensorDesc) const override;
    uint32_t GetInputNum() const override;
    uint32_t GetOutputNum() const override;

    atb::Status Setup(const atb::VariantPack &variantPack, uint64_t &workspaceSize, atb::Context *context) override;

    atb::Status Execute(const atb::VariantPack &variantPack, uint8_t *workspace, uint64_t workspaceSize,
                        atb::Context *context) override;

private:

    aclError CheckAcl(aclError ret) const;
    std::string opName_;
    size_t dataSize_;
};
} // namespace common
} // namespace atb_speed
#endif