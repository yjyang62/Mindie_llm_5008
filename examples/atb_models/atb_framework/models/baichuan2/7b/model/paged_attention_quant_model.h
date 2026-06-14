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

#ifndef ATB_SPEED_MODELS_BAICHUAN2_7B_PA_QUANT_MODEL_H
#define ATB_SPEED_MODELS_BAICHUAN2_7B_PA_QUANT_MODEL_H

#include "atb_speed/base/model.h"
#include "atb_speed/utils/model_factory.h"
#include "models/baichuan2/7b/layer/paged_attention_quant_layer.h"
#include "models/base/model/decoder_model.h"

namespace atb_speed {
namespace baichuan2_7b {

class PagedAttentionQuantModel : public atb_speed::base::DecoderModel {
public:
    explicit PagedAttentionQuantModel(const std::string &param);

protected:
    atb::Status CreateLayerOperation(atb::Operation **op, uint32_t layerId) override;
};

} // namespace baichuan2_7b
} // namespace atb_speed
#endif
