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

#ifndef ATB_SPEED_MODELS_MINICPM_DECODER_LAYER_310B_H
#define ATB_SPEED_MODELS_MINICPM_DECODER_LAYER_310B_H

#include <vector>

#include "atb/atb_infer.h"
#include "atb_speed/log.h"

namespace atb_speed {
namespace minicpm {

struct DecoderLayerParam {
    float normEps = 0;
    float scaleDepth = 0;
    int layerId = 0;
    int numHiddenLayers = 0;
    int numAttentionHeads = 7;
    int numKeyValueHeads = 4;
    int hiddenSize = 0;
    int seqLength = 1;
    bool isGQA = false;
    bool isPrefill = false;
    bool isQuant = false;
    std::vector<int> packQuantType = {}; // 两个元素，第一个元素代表QKV pack的量化类型，第二个元素代表MLP pack的量化类型
    std::vector<int> linearQuantType = {}; // 七个元素，分别代表q，k，v，self attention out，gate，up，down linear的类型
    std::vector<int> linearTransposeType;
};

atb::Status DecoderLayer(const DecoderLayerParam &param, atb::Operation **operation);

}  // namespace minicpm
}  // namespace atb_speed
#endif