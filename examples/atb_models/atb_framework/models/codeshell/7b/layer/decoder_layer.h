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

#ifndef ATB_SPEED_MODELS_CODESHELL_7B_DECODER_LAYER_H
#define ATB_SPEED_MODELS_CODESHELL_7B_DECODER_LAYER_H

#include "atb/atb_infer.h"
#include "models/base/param/layer_param.h"
#include "models/base/layer/decoder_layer.h"

namespace atb_speed {
namespace codeshell_7b {
class DecoderLayer : public atb_speed::base::DecoderLayer<atb::infer::LayerNormParam> {
public:
    explicit DecoderLayer(const atb_speed::base::LayerParam &param);
    ~DecoderLayer() override {};
    
protected:
    void SetMlpParam(atb_speed::common::MlpParam<atb::infer::LayerNormParam> &mlpParam) override;
};

}  // namespace codeshell_7b
}  // namespace atb_speed
#endif