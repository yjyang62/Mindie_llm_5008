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

#ifndef ATB_SPEED_PLUGIN_UTILS_H
#define ATB_SPEED_PLUGIN_UTILS_H
#include "operations/aclnn/core/acl_nn_operation_cache.h"
#include "operations/aclnn/core/acl_nn_tensor.h"

namespace atb_speed {
namespace common {

/// Constant to represent index
const int DIM0 = 0;
const int DIM1 = 1;
const int DIM2 = 2;
const int DIM3 = 3;
/// Constant to represent number
const int NUM1 = 1;
const int NUM2 = 2;
const int NUM3 = 3;
const int NUM4 = 4;
const int NUM5 = 5;
const int NUM6 = 6;
const int NUM7 = 7;
const int NUM8 = 8;
const int NUM9 = 9;
const int NUM10 = 10;

/// Calculate stride along each dimension when copying a tensor.
///
/// \param tensorDims The size of each axis in a tensor.
/// \return The number of steps needed in memory to move to the next element
///     along each dimension when copying a tensor.
atb::SVector<int64_t> GetCopyTensorStride(atb::Dims &tensorDims);

/// Calculate stride along each dimension when transposing a tensor.
///
/// \param tensorDims The size of each axis in a tensor.
/// \return The number of steps needed in memory to move to the next element
///     along each dimension when transposing a tensor.
atb::SVector<int64_t> GetTransposeTensorStride(atb::Dims &tensorDims);

/// Call `aclCreateTensor` API to create `aclTensor`.
///
/// \param viewDims The dimension of tensor's view shape.
/// \param storageDims The dimension of tensor's storage shape.
/// \param atbTensor The tensor passed through ATB framework.
/// \param aclnnTensor A pointer to an `AclNNTensor` object whose `tensor` attribute is updated
///     using the return value of `aclCreateTensor`.
/// \return A status code that indicates whether `aclTensor` has been created.
atb::Status CallAclCreateTensor(atb::Dims &viewDims, atb::Dims &storageDims, atb::Tensor &atbTensor,
                                const std::shared_ptr<AclNNTensor> &aclnnTensor);

/// Calling `aclrtGetSocName` API to get the hardware info.
///
/// \return A bool value that indicates whether the hardware is A2.
bool IsA2();

/// Calling `aclrtGetSocName` API to get the hardware info.
///
/// \return A bool value that indicates whether the hardware is A3.
bool IsA3();

/// Calling `aclrtGetSocName` API to get the hardware info.
///
/// \return A bool value that indicates whether the hardware is 310P.
bool Is310P();

/// Reshape a tensor by squeezing batch size axis and seq len axis if the tensor's shape has two dimensions.
///
/// \param atbTensor An `atb::Tensor` object whose tensor shape requires reshaping.
/// \return The `atb::Tensor` after reshaping.
atb::Tensor SqueezeBatchSeq(atb::Tensor atbTensor);

/// Check whether `aclnnVariantPack` and `atbVariantPack` are the same, except for tensors' device data.
///
/// Two variant packs, `aclnnVariantPack` and `atbVariantPack`, are considered the same if they have the same
/// number of tensors, and each corresponding tensor in the variant packs
/// has identical data type，format，shape and host data.
/// \param aclnnVariantPack An `AclNNVariantPack` object containing tensor info of existing AclNN operation.
/// \param atbVariantPack An `atb::VariantPack` object containing tensor info passed through ATB framework.
/// \return A boolean value that indicates whether `aclnnVariantPack` and `atbVariantPack` are the same,
/// except for tensors' device data.
bool IsVariankPackEqual(const AclNNVariantPack &aclnnVariantPack, const atb::VariantPack &atbVariantPack);

/// Create a pointer to `AclNNTensor` by configuring it with tensor information extracted from `atbTensor`.
///
/// `needUpdateTensorDataPtr` is set to true, `atbTensor` and `tensorIdx` are updated with input parameters.
/// `strides` is updated by `GetCopyTensorStride`.
/// \param atbTensor Tensor passed through the ATB framework.
/// \param tensorIdx The index of the tensor in `aclOpExecutor`'s parameter list.
/// \param aclnnTensor A pointer to an `AclNNTensor` object whose `tensor` attribute is updated
///     using the return value of `aclCreateTensor`.
/// \param ReshapeTensorDecsFuncPtr A function of reshaping.
/// \param GetStride A function of GetCopyTensorStride.
/// \return The `atb::Tensor` after creating.
atb::Status CreateTensor(atb::Tensor atbTensor, size_t tensorIdx, std::shared_ptr<AclNNTensor> &aclnnTensor,
                         atb::Tensor (*ReshapeTensorDecsFuncPtr)(atb::Tensor) = nullptr,
                         atb::SVector<int64_t> (*GetStrideFuncPtr)(atb::Dims &tensorDims) = &GetCopyTensorStride);
int ConvertTensorToSeqLengths(atb::Tensor &tensor, aclIntArray *&actualSeqLengths);
} // namespace common
} // namespace atb_speed
#endif