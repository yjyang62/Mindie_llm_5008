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

#include <atb/atb_infer.h>
#include <gtest/gtest.h>
#include <pybind11/embed.h>
#include <pybind11/stl.h>
#include <vector>

#include "atb_speed/log.h"
#include "atb_framework/operations/fusion/moe/moe_mlp.h"
#include "secodeFuzz.h"
#include "../utils/fuzz_utils.h"


namespace atb_speed {
TEST(DeepseekOperationMlpDTFuzz, DecoderOperation)
{
    std::srand(time(NULL));
    ATB_SPEED_LOG_DEBUG("begin====================");
    std::string fuzzName = "DeepseekOperationDTFuzzDecoderOperationMlp";
    pybind11::module sys = pybind11::module_::import("sys");
    sys.attr("path").attr("append")("tests/fuzztest/core/python");
    pybind11::module modelModule = pybind11::module_::import("base_fuzz");
    DT_FUZZ_START(0, 10000, const_cast<char*>(fuzzName.c_str()), 0) {
        uint32_t fuzzIndex = 0;
        int NumOfExperts[] = {8};
        int GmmQuantType[] = {0};
        int PackQuantType[] = {0};
        int DenseQuantType[] = {0};
        uint32_t Topk[] = {2};
        atb_speed::common::MoeMlpParam param;
        param.transpose = FuzzUtil::GetRandomBool(fuzzIndex);
        param.supportSwiGLU = FuzzUtil::GetRandomBool(fuzzIndex);
        param.hasBias = FuzzUtil::GetRandomBool(fuzzIndex);
        param.isBF16 = FuzzUtil::GetRandomBool(fuzzIndex);
        param.gateUpTransposeB = FuzzUtil::GetRandomBool(fuzzIndex);
        param.downTransposeB = FuzzUtil::GetRandomBool(fuzzIndex);
        param.topk = *Topk;
        param.numOfExperts = *NumOfExperts;
        param.gmmQuantType = *GmmQuantType;
        param.packQuantType = *PackQuantType;
        param.denseQuantType = *DenseQuantType;
        int moeLinearQuantTypeSize = *(int *) DT_SetGetNumberRange(&g_Element[fuzzIndex++], 1, 1, 20);
        std::vector<int> modelMoeLinearQuantType;
        for (int i = 0; i < moeLinearQuantTypeSize; ++i) {
            modelMoeLinearQuantType.push_back(std::rand() % 100); // 100 is max len of seqlen
        }
        param.moeLinearQuantType = modelMoeLinearQuantType;
        atb::Node linearNode;
        try {
            pybind11::object llamaFuzz = modelModule.attr("BaseFuzz");
            pybind11::object llamaFuzzIns = llamaFuzz("common_MoeMlp");
            
        } catch (const std::exception& e) {
            std::cerr << e.what() << std::endl;
        }
    }
    DT_FUZZ_END()
    SUCCEED();
}
}