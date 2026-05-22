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

#ifndef ATB_SPEED_PLUGIN_ACLNN_NN_OPERATION_GLOBAL_CACHE_H
#define ATB_SPEED_PLUGIN_ACLNN_NN_OPERATION_GLOBAL_CACHE_H

#include <vector>
#include <string>
#include <map>
#include <atb/atb_infer.h>
#include "acl_nn_operation_cache.h"
#include "acl_nn_operation.h"

namespace atb_speed {
namespace common {

/// Maximum number of objects stored in the global cache by default
const uint16_t DEFAULT_ACLNN_GLOBAL_CACHE_SIZE = 16;
constexpr int32_t DECIMAL = 10;  /// Decimal base

/// A class that manages global cache.
///
/// This class keeps a private container to manage `AclNNOpCache` objects that may be shared between operations.
/// It provides get and update methods for retrieving and modifying the private container's state.
/// There is also a print function for debugging.
class AclNNGlobalCache {
public:
    /// The Class constructor.
    ///
    /// Update `globalCacheCountMax_` with the environment variable `MINDIE_ACLNN_CACHE_GLOBAL_COUNT`.
    explicit AclNNGlobalCache();
    /// Retrieve the cache object on a cache hit.
    ///
    /// A Cache is hit if the `variantPack` is the same, except for tensors' device data.
    ///
    /// \param opName An operations's name.
    /// \param variantPack Information about input and output tensors of an ATB operation.
    /// \return Return a pointer to an `AclNNOpCache` object on a cache hit; otherwise, returns a nullptr.
    std::shared_ptr<AclNNOpCache> GetGlobalCache(std::string opName, atb::VariantPack variantPack);
    /// Add or replace an cache object.
    ///
    /// Locate the global cache list for the current operation using `opName`.
    /// Add the `cache` at the index specifidex by `nextUpdateIndex_`.
    /// If the slot already contains a cache object, replace it.
    /// Cache is not added if it's executor is not repeatable.
    ///
    /// \param opName An operations's name.
    /// \param cache The cache to be added to the `aclnnGlobalCache_` container.
    /// \return A status code that indicates whether the update operation was successful.
    atb::Status UpdateGlobalCache(std::string opName, std::shared_ptr<AclNNOpCache> cache);
    /// Print a summary of the objects stored in the `aclnnGlobalCache_`.
    ///
    /// The operation's name and the corresponding global cache address are printed.
    ///
    /// \return Cache info.
    std::string PrintGlobalCache();

private:
    /// An index maintains a record of the next available cache slot
    int nextUpdateIndex_ = 0;
    /// Maximum number of objects stored in the global cache
    uint16_t globalCacheCountMax_ = 16;
    /// A map stores `AclNNOpCache` objects.
    ///
    /// Key is an operation's name. Value is a vector of pointers to `AclNNOpCache` object.
    /// Cache is not shared between different types of operations.
    std::map<std::string, std::vector<std::shared_ptr<AclNNOpCache>>> aclnnGlobalCache_;
};

} // namespace common
} // namespace atb_speed
#endif