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

#ifndef ATB_SPEED_PLUGIN_ACLNN_NN_TENSOR_H
#define ATB_SPEED_PLUGIN_ACLNN_NN_TENSOR_H
#include <string>
#include <acl/acl.h>
#include <aclnn/acl_meta.h>
#include <atb/atb_infer.h>

namespace atb_speed {
namespace common {

/// A struct contains the tensor's host data.
///
/// This struct stores the tensor's host data in an int array format.
/// Host data is treat as an operation parameter and used during the setup phase to create the `aclOpExecutor`.
struct AclNNIntArray {
    /// This struct is created by calling `aclCreateIntArray` and should be destroyed by calling `aclDestroyIntArray`.
    /// It is used to create the `aclOpExecutor`.
    aclIntArray* intArray = nullptr;
    /// Data used to create the `aclIntArray*`. It is copied from atb::Tensor's hostData.
    std::vector<int64_t> data = {};
    /// The size of `data` in bytes.
    std::vector<int32_t> dataOri = {};
    uint64_t dataSize = 0;
};

/// A class contains tensor information.
///
/// AclNN operations and ATB operations organize tensor in different format.
/// This class stores the information necessary for easy conversion and tensor usage.
class AclNNTensor {
public:
    /// An const value to indicate that the `tensorListidx` is invalid.
    static const int64_t notInTensorList = -1;

    /// Tensor passed through the ATB framework.
    atb::Tensor atbTensor;
    /// The stride of each dimension in the tensor's view shape. Used when creating `aclTensor`.
    atb::SVector<int64_t> strides = {};
    /// Tensor passed into the AclNN operation.
    aclTensor *tensor = nullptr;
    /// An AclNNIntArray object contain tensor's host data in the int array format.
    AclNNIntArray intArrayHostData;
    /// The index of the tensor in the tensor list. Used when `aclTensor` is passed into `aclTensorList`.
    int tensorListidx = notInTensorList;
    /// The index of the tensor in `aclOpExecutor`'s parameter list.
    int tensorIdx = -1;
    /// An indicator that shows whether the tensor's device data needs to be updated in the execution.
    bool needUpdateTensorDataPtr = false;
};

} // namespace common
} // namespace atb_speed
#endif