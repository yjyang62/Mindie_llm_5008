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

#ifndef ATB_TORCH_ATB_COMM_MANAGER_H
#define ATB_TORCH_ATB_COMM_MANAGER_H
#include "atb_speed/base/external_comm_manager.h"

namespace atb_torch {

void initProcessGroup(std::string backend, uint32_t worldSize, uint32_t rank,
    std::string rankTableFile);
std::string newGroup(const std::vector<uint32_t> &rankIds,
    uint32_t subCommRankId, std::string backend, uint32_t bufferSize);
std::string getBackend(std::string processGroup);
bool isPGInitialized();

} // namespace atb_torch
#endif
