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
#include "paged_attention_quant_model.h"

#include <vector>

#include "atb/atb_infer.h"
#include "nlohmann/json.hpp"

#include "models/baichuan2/7b/layer/paged_attention_quant_layer.h"

namespace atb_speed {
namespace baichuan2_7b {
REGISTER_MODEL(baichuan2_7b, PagedAttentionQuantModel);

PagedAttentionQuantModel::PagedAttentionQuantModel(const std::string &param) : atb_speed::base::DecoderModel(param) {}

atb::Status PagedAttentionQuantModel::CreateLayerOperation(atb::Operation **op, uint32_t layerId)
{
    atb_speed::base::LayerParam layerParam;
    atb_speed::base::DecoderModel::SetLayerParam(layerParam, layerId);
    PagedAttentionQuantLayer decoderLayer(layerParam);
    CHECK_OPERATION_STATUS_RETURN(decoderLayer.BuildGraph(op));
    return atb::NO_ERROR;
}

} // namespace baichuan2_7b
} // namespace atb_speed
