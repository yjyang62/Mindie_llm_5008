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

#ifndef ATB_SPEED_LAYERS_MLP_310B_H
#define ATB_SPEED_LAYERS_MLP_310B_H

#include "atb/atb_infer.h"
#include "atb_speed/utils/operation_util.h"

namespace atb_speed {
namespace common {
struct MlpLiteParam {
    atb::infer::ActivationType activationType;
    bool transposeB = true;
    bool isBias = false;
    bool isPack = false;
    bool isQuant = false;
    bool isSparse = false;
    bool noGate = false;
    bool isBF16 = false;
};

atb::Status MlpLiteLayer(const MlpLiteParam &param, atb::Operation **operation);

} // namespace common
} // namespace atb_speed
#endif