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

#ifndef ATB_SPEED_MODELS_CODESHELL_7B_DECODER_MODEL_H
#define ATB_SPEED_MODELS_CODESHELL_7B_DECODER_MODEL_H

#include "atb_speed/utils/model_factory.h"
#include "models/codeshell/7b/layer/decoder_layer.h"
#include "models/base/model/decoder_model.h"

namespace atb_speed {
namespace codeshell_7b {

class DecoderModel : public atb_speed::base::DecoderModel {
public:
    explicit DecoderModel(const std::string &param);
    atb::Status InferShape(const std::vector<atb::TensorDesc> &inTensorDescs,
                           std::vector<atb::TensorDesc> &outTensorDescs) override;

protected:
    atb::Status CreateLayerOperation(atb::Operation **op, uint32_t layerId) override;
    void SetLmHeadParam(atb_speed::common::LmHeadParam &lmHeadParam) override;
};

REGISTER_MODEL(codeshell_7b, DecoderModel);
}  // namespace codeshell_7b
}  // namespace atb_speed
#endif
