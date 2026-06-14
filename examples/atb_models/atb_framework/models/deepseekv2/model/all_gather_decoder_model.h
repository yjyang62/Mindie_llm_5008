/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * MindIE is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

#ifndef ATB_SPEED_MODELS_DEEPSEEK_V2_ALL_GATHER_DECODER_MODEL_H
#define ATB_SPEED_MODELS_DEEPSEEK_V2_ALL_GATHER_DECODER_MODEL_H

#include "models/deepseekv2/model/decoder_model.h"

namespace atb_speed {
namespace deepseekV2 {

class AllGatherDecoderModel : public DecoderModel {
public:
    explicit AllGatherDecoderModel(const std::string &param);
    atb::Status InferShape(const std::vector<atb::TensorDesc> &inTensorDescs,
                           std::vector<atb::TensorDesc> &outTensorDescs) override;

private:
    int64_t BuildGraph() override;
    atb::Status AddExpertRouterAllGather();
    std::map<std::string, uint32_t> inTensorMap_;
};

REGISTER_MODEL(deepseekV2, AllGatherDecoderModel);

}  // namespace deepseekV2
}  // namespace atb_speed
#endif