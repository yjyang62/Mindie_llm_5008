/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * MindIE is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

#include "atb_speed/utils/singleton.h"
#include "atb_speed/base/external_comm_manager.h"
#include "parallel_info.h"

namespace atb_speed {
namespace common {

std::string InitCommBackend(uint32_t localWorldSize, const std::vector<uint32_t> rankIds, std::string commBackend)
{
    if (localWorldSize <= 0) {
        throw std::runtime_error("Number of devices in the current node is less than or equal to 0.");
    }
    // Get backend
    std::string backend = commBackend;
    // change to hccl if the communication channel across nodes
    int32_t currentDevice = -1;
    for (uint32_t item : rankIds) {
        if (currentDevice != -1 && static_cast<int32_t>(ceil(item / localWorldSize)) != currentDevice) {
            backend = "hccl";
            break;
        }
        currentDevice = static_cast<int32_t>(ceil(item / localWorldSize));
    }
    // The hccl backend is utilized in the single node scenario
    // when a rankTableFile is supplied and the communication channel spans the entire world size.
    uint32_t worldSize = GetSingleton<ExternalCommManager>().worldSize_;
    if (worldSize <= localWorldSize && GetSingleton<ExternalCommManager>().rankTableFile_ != "" && \
        rankIds.size() == worldSize) {
        backend = "hccl";
    }
    return backend;
}

void ParallelInfo::InitCommDomain(HcclComm& hcclComm, std::string& commDomain, std::string backend) const
{
    if (backend == "") {
        backend = this->defaultBackend;
    }
    // Get current stream id
    uint32_t streamId = GetSingleton<DapManager>().GetStreamId();

    // Assign commDomain by rankIds and rank
    commDomain = GetSingleton<ExternalCommManager>().GetCommDomain(
        this->groupId, this->rankIds, this->rank, backend, this->bufferSize, streamId);
    // Get hcclComm (only created when hccl backend is used and inference across multi nodes)
    hcclComm = GetSingleton<ExternalCommManager>().GetCommPtr(commDomain);

    ATB_SPEED_LOG_DEBUG(this->ToString());
}

bool ParallelInfo::IsEnabled() const
{
    return this->rankIds.size() > 1;
}

std::string ParallelInfo::ToString() const
{
    std::stringstream ss;
    ss << "ParallelInfo: rank: " << this->rank
        << ", rankIds: " << this->rankIds
        << ", groupId: " << this->groupId
        << ", defaultBackend: " << this->defaultBackend
        << ", bufferSize: " << this->bufferSize;
    return ss.str();
}

} // namespace common
} // namesapce atb_speed