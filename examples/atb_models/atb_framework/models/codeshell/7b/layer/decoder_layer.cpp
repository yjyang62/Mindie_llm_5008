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
#include "codeshell/7b/layer/decoder_layer.h"

namespace atb_speed {
namespace codeshell_7b {

DecoderLayer::DecoderLayer(
    const atb_speed::base::LayerParam &param) : atb_speed::base::DecoderLayer<atb::infer::LayerNormParam>(
        static_cast<atb_speed::base::LayerParam>(param))
{
    this->param = param;
    this->param.CheckParam();
}

void DecoderLayer::SetMlpParam(atb_speed::common::MlpParam<atb::infer::LayerNormParam> &mlpParam)
{
    atb_speed::base::DecoderLayer<atb::infer::LayerNormParam>::SetMlpParam(mlpParam);
    mlpParam.mlpPackType = atb_speed::common::GetMlpPackType(param.packQuantType.at(1), true);
    if (!this->param.enableSwiGLU) {
        mlpParam.activationParam.activationType = atb::infer::ActivationType::ACTIVATION_GELU;
    }
}
} // namespace codeshell_7b
} // namespace atb_speed