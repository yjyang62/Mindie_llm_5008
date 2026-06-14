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

#ifndef ATB_SPEED_PLUGIN_ACLNN_NN_OPERATION_H
#define ATB_SPEED_PLUGIN_ACLNN_NN_OPERATION_H
#include <atb/operation_infra.h>
#include "acl_nn_operation_cache.h"
#include "atb_speed/utils/operation_util.h"

namespace atb_speed {
namespace common {

/// An class that inherited from `atb::Operation` class. An `atb::Operation` class defines a series of
/// interfaces required for operation preparation and execution.
class AclNNOperation : public atb::OperationInfra {
public:
    /// Class constructor.
    ///
    /// Initialize an `AclNNOpCache` pointer for the object's local cache (`aclnnOpCache_`) and set `opName`.
    /// \param opName The name of the AclNN operation.
    explicit AclNNOperation(const std::string &opName);
    ~AclNNOperation() override;
    /// Return the AclNN operation's name.
    /// \return The object's `opName`.
    std::string GetName() const override;
    /// Preparations before operation execution.
    ///
    /// This function calls `UpdateAclNNOpCache` to update `aclnnOpCache_`
    /// and calculate the memory space that needs to be allocated during the operation execution process.
    /// \param variantPack Operation's input and output tensor info.
    /// \param workspaceSize The size of the work space.
    /// \param context The context in which operation's preparation is performed.
    /// \return A status code that indicates whether the setup process was successful.
    atb::Status Setup(const atb::VariantPack &variantPack, uint64_t &workspaceSize, atb::Context *context) override;
    /// Operation execution process.
    ///
    /// Call `GetExecuteStream` from `context`. Call `UpdateAclNNVariantPack` to update tensor's device data.
    /// Execute the operation.
    /// \param variantPack Operation's input and output tensor info.
    /// \param workspace A pointer the memory address allocated by the operation.
    /// \param workspaceSize The size of the work space.
    /// \param context The context in which operation's preparation is performed.
    /// \return A status code that indicates whether the execute process was successful.
    atb::Status Execute(const atb::VariantPack &variantPack, uint8_t *workspace, uint64_t workspaceSize,
                        atb::Context *context) override;
    /// Release all occupied resources, particularly those stored in `aclnnOpCache_`.
    void DestroyOperation() const;

protected:
    /// Prepare the operation's input tensors.
    ///
    /// \param variantPack An `atb::VariantPack` object containing tensor info passed through ATB framework.
    /// \return A status code that indicates whether variantPack was created successfully.
    virtual atb::Status CreateAclNNInTensorVariantPack(const atb::VariantPack &variantPack);
    /// Prepare the operation's output tensors.
    ///
    /// \param variantPack An `atb::VariantPack` object containing tensor info passed through ATB framework.
    /// \return A status code that indicates whether variantPack was created successfully.
    virtual atb::Status CreateAclNNOutTensorVariantPack(const atb::VariantPack &variantPack);
    /// Call AclNN operation's first phase API to get work space size and `aclOpExecutor`.
    ///
    /// \return The return value of AclNN's first phase API.
    virtual int SetAclNNWorkspaceExecutor() = 0;
    /// Call AclNN operation's second phase API to execute the operation.
    ///
    /// \return The return value of AclNN's second phase API.
    virtual int ExecuteAclNNOp(uint8_t *workspace, aclrtStream &stream) = 0;

    /// An `AclNNOpCache` object that can be reused within the current operation object.
    std::shared_ptr<AclNNOpCache> aclnnOpCache_ = nullptr;
    /// A human identifiable name for the operation's name.
    std::string opName_;
private:
/// Create the operation's local cache (`aclnnOpCache_`).
    ///
    /// Create the operation's input tensor and output tensor by calling `CreateAclNNVariantPack`.
    /// Call `SetAclNNWorkspaceExecutor` to get work space size and `aclOpExecutor`.
    /// Call `aclSetAclOpExecutorRepeatable` to make `aclOpExecutor` reusable.
    /// \param variantPack Operation's input and output tensor info passed from ATB framework.
    /// \return A status code that indicates whether `aclnnOpCache_` was successfully created.
    atb::Status CreateAclNNOpCache(const atb::VariantPack &variantPack);
    /// Verify if the local cache or global cache is hit. If neither is hit, create a new instance
    /// by calling `CreateAclNNOpCache`, then update both the `ExecutorManager` and `AclNNGlobalCache`.
    /// \param variantPack Operation's input and output tensor info.
    /// \return A status code that indicates whether `aclnnOpCache_` was successfully updated.
    atb::Status UpdateAclNNOpCache(const atb::VariantPack &variantPack);
    /// Prepare the operation's input tensors and output tensors.
    ///
    /// This function calls `CreateAclNNInTensorVariantPack` and `CreateAclNNOutTensorVariantPack`.
    /// \param variantPack An `atb::VariantPack` object containing tensor info passed through ATB framework.
    /// \return A status code that indicates whether variantPack was created successfully.
    atb::Status CreateAclNNVariantPack(const atb::VariantPack &variantPack);
};
} // namespace common
} // namespace atb_speed
#endif