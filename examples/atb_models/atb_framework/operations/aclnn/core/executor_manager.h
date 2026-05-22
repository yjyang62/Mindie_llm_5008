/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 * MindIE is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

#ifndef ATB_SPEED_PLUGIN_ACLNN_NN_OPERATION_EXECUTOR_MANAGER_H
#define ATB_SPEED_PLUGIN_ACLNN_NN_OPERATION_EXECUTOR_MANAGER_H

#include <map>
#include <string>
#include <aclnn/acl_meta.h>

namespace atb_speed {
namespace common {

/// A class that manages `aclOpExecutor` objects and corresponding reference number.
///
/// Each `AclNNOpCache` object has a unique `aclOpExecutor` object.
/// Since an `aclOpExecutor` object can be accessed through both local cache and global cache,
/// when destroying an `aclOpExecutor` object, it's important to ensure that it no longer has any references.
class ExecutorManager {
public:
    /// Increase the reference count of the input `executor` by 1.
    /// If the `executor` has no reference before, the reference count is set to 1.
    ///
    /// \param executor An `aclOpExecutor` object whose reference count needs to be increased.
    /// \return The number of references after the increase.
    int IncreaseReference(aclOpExecutor *executor);
    /// Decrease the reference count of the input `executor` by 1.
    /// If the `executor` has no reference after the decrease, it will be removed from `executorCount_`.
    ///
    /// \param executor An `aclOpExecutor` object whose reference count needs to be decreased.
    /// \return The number of references after the decrease.
    int DecreaseReference(aclOpExecutor *executor);
    /// Print a summary of the objects stored in the `executorCount_`.
    ///
    /// The `aclOpExecutor`'s address and the corresponding reference number are printed.
    ///
    /// \return `aclOpExecutor` info.
    std::string PrintExecutorCount();

private:
    /// A map stores `aclOpExecutor` objects.
    ///
    /// Key is an `aclOpExecutor` object's address. Value is it's reference number.
    std::map<aclOpExecutor *, int> executorCount_;
};

} // namespace common
} // namespace atb_speed
#endif