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
#include "models/deepseek/model/decoder_model.h"

namespace atb_speed {
namespace deepseek {

constexpr int DEEPSEEK_WEIGHT_COUNT_PER_LAYER = 68;

DecoderModel::DecoderModel(const std::string &param) : atb_speed::moe::MoeDecoderModel(param)
{
    this->weightCountPerLayer = DEEPSEEK_WEIGHT_COUNT_PER_LAYER;
}

} // namespace deepseek
} // namespace atb_speed