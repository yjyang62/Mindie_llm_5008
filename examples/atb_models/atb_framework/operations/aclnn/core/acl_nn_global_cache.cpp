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
#include <sstream>
#include "atb_speed/log.h"
#include "atb_speed/utils/operation_util.h"
#include "operations/aclnn/utils/utils.h"
#include "acl_nn_global_cache.h"

namespace atb_speed {
namespace common {

AclNNGlobalCache::AclNNGlobalCache()
{
    uint64_t globalCacheCountMax = DEFAULT_ACLNN_GLOBAL_CACHE_SIZE;
    const char *envStr = std::getenv("MINDIE_ACLNN_CACHE_GLOBAL_COUNT");
    if (envStr != nullptr) {
        char* endPtr = nullptr;
        errno = 0;
        uint64_t val = std::strtoull(envStr, &endPtr, 10);
        if (errno == 0 && endPtr != envStr && *endPtr == '\0' && val < 100) {   // 100: threshold
            globalCacheCountMax = val;
        }
    }
    this->globalCacheCountMax_ = globalCacheCountMax;
}

std::shared_ptr<AclNNOpCache> AclNNGlobalCache::GetGlobalCache(std::string opName, atb::VariantPack variantPack)
{
    // 获取Op对应的Global Cache列表
    std::map<std::string, std::vector<std::shared_ptr<AclNNOpCache>>>::iterator it = \
        this->aclnnGlobalCache_.find(opName);
    if (it == this->aclnnGlobalCache_.end()) {
        ATB_SPEED_LOG_DEBUG("Plugin Op Cache: Op name[" << opName << "] not found in AclNNGlobalCache");
        return nullptr;
    }
    std::vector<std::shared_ptr<AclNNOpCache>> &opGlobalCacheList = it->second;

    // 在Global Cache列表中基于variantPack找到匹配的Cache
    for (size_t i = 0; i < opGlobalCacheList.size(); i++) {
        if (opGlobalCacheList[i] == nullptr) {
            ATB_SPEED_LOG_DEBUG("Plugin Op Cache: Global Cache index " << i << " is nullptr");
            continue;
        }
        ATB_SPEED_LOG_DEBUG("Plugin Op Cache: Global Cache index " << i << " call IsVariankPackEqual");
        if (opGlobalCacheList[i]->executorRepeatable && \
            IsVariankPackEqual(opGlobalCacheList[i]->aclnnVariantPack, variantPack)) {
            // Global Cache命中
            return opGlobalCacheList[i];
        }
    }

    return nullptr;
}

atb::Status AclNNGlobalCache::UpdateGlobalCache(std::string opName, std::shared_ptr<AclNNOpCache> cache)
{
    // 若Local Cache中Executor不可复用，不更新Global Cache
    if (!cache->executorRepeatable) {
        ATB_SPEED_LOG_DEBUG("Plugin Op Cache: Op name[" << opName << "] not repeatable, do not update global cache");
        return atb::NO_ERROR;
    }

    // Check Global Cache Size
    if (this->globalCacheCountMax_ == 0) {
        return atb::NO_ERROR;
    }
    
    // 获取Op对应的Global Cache列表
    std::map<std::string, std::vector<std::shared_ptr<AclNNOpCache>>>::iterator it = \
        this->aclnnGlobalCache_.find(opName);
    if (it == this->aclnnGlobalCache_.end()) {
        // 不存在opName对应的Cache列表
        ATB_SPEED_LOG_DEBUG("Plugin Op Cache: Op name[" << opName << "] not found in AclNNGlobalCache, add one");
        this->aclnnGlobalCache_[opName] = {cache};
        return atb::NO_ERROR;
    }
    std::vector<std::shared_ptr<AclNNOpCache>> &opGlobalCacheList = it->second;

    // Cache未已满
    if (opGlobalCacheList.size() < this->globalCacheCountMax_) {
        ATB_SPEED_LOG_DEBUG("Plugin Op Cache: Op name[" << opName << "] global cache is not full, add one");
        opGlobalCacheList.push_back(cache);
        return atb::NO_ERROR;
    }

    // Cache已满
    ATB_SPEED_LOG_DEBUG("Plugin Op Cache: Op name["
                  << opName << "] global cache is full, update index " << nextUpdateIndex_);
    opGlobalCacheList[nextUpdateIndex_] = cache;
    CHECK_PARAM_NE(globalCacheCountMax_, 0);
    nextUpdateIndex_ = (nextUpdateIndex_ + 1) % globalCacheCountMax_;
    return atb::NO_ERROR;
}

std::string AclNNGlobalCache::PrintGlobalCache()
{
    std::stringstream ss;
    ss << "Plugin Op Cache: Global Cache Summary ";
    std::map<std::string, std::vector<std::shared_ptr<AclNNOpCache>>>::iterator it;
    for (it = this->aclnnGlobalCache_.begin(); it != this->aclnnGlobalCache_.end(); it++) {
        ss << "Op name[" << it->first << "] ";
        std::vector<std::shared_ptr<AclNNOpCache>> &opGlobalCacheList = it->second;
        for (size_t i = 0; i < opGlobalCacheList.size(); i++) {
            ss << "Cache Addr[" << opGlobalCacheList[i].get() << "] ";
        }
    }
    return ss.str();
}

} // namespace common
} // namespace atb_speed
