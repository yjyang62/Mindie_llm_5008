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

#ifndef ATB_SPEED_PLUGIN_ACLNN_NN_OPERATION_LOCAL_CACHE_H
#define ATB_SPEED_PLUGIN_ACLNN_NN_OPERATION_LOCAL_CACHE_H
#include "acl_nn_tensor.h"

namespace atb_speed {
namespace common {

/// Information about input and output tensors of an AclNN operation.
struct AclNNVariantPack {
    /// A container stores an AclNN operation's in tensor in order.
    /// Each `AclNNTensor` object contains one `aclTensor`.
    atb::SVector<std::shared_ptr<AclNNTensor>> aclInTensors;
    /// A container stores an AclNN operation's out tensor in order.
    /// Each `AclNNTensor` object contains one `aclTensor`.
    atb::SVector<std::shared_ptr<AclNNTensor>> aclOutTensors;
    /// A container stores an AclNN operation's input `aclTensorList` in order.
    /// Each `aclTensorList` object may contain multiple `aclTensor`.
    atb::SVector<aclTensorList *> aclInTensorList;
    /// A container stores an AclNN operation's output `aclTensorList` in order.
    /// Each `aclTensorList` object may contain multiple `aclTensor`.
    atb::SVector<aclTensorList *> aclOutTensorList;
};

/// AclNNOpCache stores information of an operation that can be reused between operations.
struct AclNNOpCache {
    /// Information about input and output tensors of an AclNN operation.
    AclNNVariantPack aclnnVariantPack;
    /// AclNN operation's executor, which contains the operator computation process.
    aclOpExecutor *aclExecutor = nullptr;
    /// An indicator shows whether the `aclOpExecutor` is repeatable.
    bool executorRepeatable = false;
    /// Size of the workspace to be allocated on the device.
    uint64_t workspaceSize;
    /// Update the device memory address in `aclTensor` objects when the device memory changes.
    ///
    /// \param variantPack Information about input and output tensors of an AclNN operation.
    /// \return A status code that indicates whether the update operation was successful.
    atb::Status UpdateAclNNVariantPack(const atb::VariantPack &variantPack);
    /// Destroy resources allocated in `AclNNOpCache`.
    ///
    /// Destroy `aclOpExecutor` if it's repeatable and has no reference.
    /// Destroy `aclTensor` and `aclTensorList` if `aclOpExecutor` is destroyed.
    void Destroy();
};

} // namespace common
} // namespace atb_speed
#endif