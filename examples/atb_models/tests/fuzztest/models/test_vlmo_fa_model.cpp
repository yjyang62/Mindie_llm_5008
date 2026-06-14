/**
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

#include <gtest/gtest.h>
#include <atb/atb_infer.h>
#include "atb_speed/log.h"
#include "models/vlmo/2b/layer/encoder_layer.h"
#include "models/vlmo/2b/layer/encoder_vl_layer.h"
#include "models/vlmo/2b/model/flash_attention_model.h"
#include "nlohmann/json.hpp"
#include "../utils/fuzz_utils.h"
#include "secodeFuzz.h"

namespace atb_speed {
TEST(VlmoModelDTFuzz, FlashAttentionModel)
{
    std::srand(time(NULL));
    ATB_SPEED_LOG_DEBUG("begin====================");
    std::string fuzzName = "VlmoModelDTFuzzFlashAttentionModel";

    DT_FUZZ_START(0, 1000, const_cast<char*>(fuzzName.c_str()), 0) {
        uint32_t fuzzIndex = 0;

        nlohmann::json parameter, aclParameter;
        parameter["rmsNormEps"] = 0.01;
        parameter["headNum"] = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 1, 1, 1);
        int dk = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 1, 1, 8);
        parameter["dk"] = dk;
        int layerNum = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 1, 1, 200);
        parameter["layerNum"] = layerNum;
        int rankSize = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 1, 1, 8);
        parameter["rank"] = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 0, 0, rankSize);
        parameter["rankSize"] = rankSize;
        int maxTextLen = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 1, 1, 200);
        parameter["maxTextLen"] = maxTextLen;
        int vlLayerIndex = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 1, 1, 1);
        parameter["vlLayerIndex"] = vlLayerIndex;
        parameter["backend"] = "vlmo";
        std::string parameter_string = parameter.dump();

        try {
            auto model = new atb_speed::vlmo::FlashAttentionModel(parameter_string);
            model->Init(nullptr, nullptr, nullptr);
        } catch (const std::exception& e) {
            std::cerr << e.what() << std::endl;
        }
    }
    DT_FUZZ_END()
    SUCCEED();
}
}