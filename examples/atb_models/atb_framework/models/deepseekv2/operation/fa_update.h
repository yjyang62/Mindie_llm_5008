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

#ifndef ATB_SPEED_MODELS_COMMON_FA_UPDATE_H
#define ATB_SPEED_MODELS_COMMON_FA_UPDATE_H

#include <map>
#include <atb/atb_infer.h>
#include "models/deepseekv2/operation/latent_attention.h"

namespace atb_speed {
namespace common {

template <typename NormParamType>
atb::Status AddDecodeUpdate(const LatentAttentionParam<NormParamType> &param, atb::GraphParam& opGraph,
    std::map<std::string, uint32_t>& tensorMap);
}
}

#endif