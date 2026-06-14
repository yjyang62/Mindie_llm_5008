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

#ifndef ATB_MODELS_BERT_ENCODER_MODEL_H
#define ATB_MODELS_BERT_ENCODER_MODEL_H


#include <atb/atb_infer.h>
#include <nlohmann/json.hpp>
#include "atb_speed/base/model.h"


namespace atb_speed::bert {

    class EncoderModel : public Model {
    public:
        struct Param {
            int dk = 0;
            int64_t geluApproximate = -1;
            int headNum = 0;
            float layerNormEps = 0;
            int64_t layerNormImplMode = 0;
            int layerNum = 0;
            float qkScale = 1.0;
            int rank = 0;
            int rankSize = 1;
            bool enableFasterGelu = false;
            bool enableAclNNMatmul = false;
            bool enableAclNNAttn = false;
            void FromString(const std::string &param);
        };

        explicit EncoderModel(const std::string &param);
        ~EncoderModel() override;

        [[nodiscard]] uint32_t GetInputNum() const override;
        [[nodiscard]] uint32_t GetOutputNum() const override;

        atb::Status InferShape(
            const std::vector<atb::TensorDesc> &inTensorDescs,
            std::vector<atb::TensorDesc> &outTensorDescs
        ) override;

    private:
        int64_t BuildGraph() override;
        int64_t Embedding();
        int64_t Layer();
        Param param_;
        atb::Status ParseParam(const std::string &param) override;
        atb::Status BindParamHostTensor(uint32_t nodeId) override;
        std::vector<int32_t> tokenOffset_;
        std::vector<int32_t> seqLen_;
    };

}  // namespace atb_speed::bert

#endif  // ATB_MODELS_BERT_ENCODER_MODEL_H
