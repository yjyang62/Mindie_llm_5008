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

#ifndef ATB_MODELS_BERT_ENCODER_LAYER_H
#define ATB_MODELS_BERT_ENCODER_LAYER_H


#include <atb/atb_infer.h>
#include <atb/svector.h>
#include "atb_speed/log.h"
#include "atb_speed/utils/str_split.h"


namespace atb_speed::bert {

    struct EncoderLayerParam {
        int dk = 0;
        int64_t geluApproximate = -1;
        int headNum = 0;
        float layerNormEps = 0;
        int64_t layerNormImplMode = 0;
        std::string model = "bert";
        int rank = 0;
        int rankSize = 1;
        bool enableFasterGelu = false;
        bool enableAclNNMatmul = false;
        bool enableAclNNAttn = false;
    };

    atb::Status EncoderLayer(const EncoderLayerParam &param, atb::Operation **operation);

}  // namespace atb_speed::bert

#endif  // ATB_MODELS_BERT_ENCODER_LAYER_H
