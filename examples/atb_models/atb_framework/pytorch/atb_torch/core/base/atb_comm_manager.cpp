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

#include "atb_comm_manager.h"

#include "atb_speed/log.h"
#include "atb_speed/base/external_comm_manager.h"
#include "atb_speed/utils/singleton.h"

namespace atb_torch {
void initProcessGroup(std::string backend, uint32_t worldSize, uint32_t rank,
    std::string rankTableFile)
{
    atb_speed::GetSingleton<atb_speed::ExternalCommManager>().SetLcclCommDomainRange(0, 65535);
    atb_speed::GetSingleton<atb_speed::ExternalCommManager>().Init(worldSize, rank,
        backend, rankTableFile, 0);
    ATB_SPEED_LOG_DEBUG("Init default process group done.");
}

std::string newGroup(const std::vector<uint32_t> &rankIds,
    uint32_t subCommRankId, std::string backend, uint32_t bufferSize)
{
    std::string commDomain = atb_speed::GetSingleton<atb_speed::ExternalCommManager>().GetCommDomain(
        rankIds[0], rankIds, subCommRankId, backend, bufferSize, 0, false);
    ATB_SPEED_LOG_DEBUG(
        "create a new process group: " << commDomain <<
        " rankIds " << rankIds << " subCommRankId " << subCommRankId <<
        " backend " << backend << " bufferSize " << bufferSize
    );
    return commDomain;
}

std::string getBackend(std::string processGroup)
{
    std::shared_ptr<atb_speed::CommInfo> commInfo = \
        atb_speed::GetSingleton<atb_speed::ExternalCommManager>().GetCommInfo(processGroup);
    return commInfo->backend_;
}

bool isPGInitialized()
{
    return atb_speed::GetSingleton<atb_speed::ExternalCommManager>().IsInitialized();
}
}