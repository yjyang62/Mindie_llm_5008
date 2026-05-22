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
#include <gtest/gtest.h>
#include <mockcpp/mockcpp.hpp>
#include <acl/acl.h>
#include <aclnn/acl_meta.h>
#include <atb/atb_infer.h>
#include "models/qwen/model/decoder_model.h"

namespace atb_speed {


class TestableQwenDecoderModel : public qwen::QwenDecoderModel {
public:
    using QwenDecoderModel::QwenDecoderModel;

    const std::map<std::string, uint32_t>& GetIntensorMap() const { return inTensorMap; }
    const std::map<std::string, uint32_t>& GetInternalTensorMap() const { return internalTensorMap; }
    const std::map<std::string, uint32_t>& GetOutTensorMap() const { return outTensorMap; }

    void Construct()
    {
        ConstructInTensorMap();
        ConstructInternalTensorMap();
        ConstructOutTensorMap();
    }

    bool CanUseEmbedTable() { return ReuseEmbedTable(); }

    bool NeedOutputEmbedTable() { return OutputEmbedTable(); }

};

TEST(QwenDecoderModelTest, DecoderModel)
{
    GlobalMockObject::verify();

    const std::string param = R"({
        "isFA": false,
        "isBF16": true,
        "skipWordEmbedding": false,
        "isEmbeddingParallel": true,
        "isLmHeadParallel": true,
        "linearTransposeType": [
            [
                0,
                -1,
                -1,
                0,
                0,
                -1,
                0
            ]
        ],
        "lmHeadTransposeType": 0,
        "enableSwiGLU": true,
        "enableSwigluQuant": true,
        "preFetchWeightSize": 0,
        "normEps": 0.000001,
        "normType": 0,
        "numAttentionHeadsPerRank": 8,
        "hiddenSizePerAttentionHead": 128,
        "numHiddenLayers": 1,
        "numKeyValueHeadsPerRank": 1,
        "rank": 0,
        "isUnpadInputs": true,
        "hiddenSize": 5120,
        "enableFA3": false,
        "worldSize": 1,
        "backend": "lccl",
        "attnBackend": 0,
        "linearQuantType": [
            [
                0,
                -1,
                -1,
                0,
                0,
                -1,
                0
            ]
        ],
        "packQuantType": [
            [
                1,
                1
            ]
        ],
        "quantGroupSize": 0,
        "enableKvQuant": false,
        "enableLora": false,
        "isLongSeq": true,
        "enableAddNorm": false,
        "isYarn": true,
        "mscale": 1.138629436111989,
        "rankTableFile": "",
        "useQKNorm": true,
        "linearHasBias": [
            [
                false,
                false,
                false,
                false
            ]
        ],
        "matmulBackend": 0,
        "enableIntraLayerAddNorm": false,
        "enableInterLayerAddNorm": false,
        "enableGreedySearchOpt": false,
        "enableOmniAttention": false,
        "enableQScale": false,
        "enableModelConfuscation": false,
        "modelConfuscationFd": 0,
        "mapping": {
            "worldSize": 1,
            "rank": 0,
            "rankTableFile": "",
            "localWorldSize": 1,
            "lcclCommDomainLowerBound": 0,
            "lcclCommDomainUpperBound": 65536,
            "wordEmbedTp": {
                "groupId": 0,
                "rankIds": [
                    0
                ],
                "rank": 0,
                "bufferSize": 128
            },
            "wordEmbedDp": {
                "groupId": 0,
                "rankIds": [
                    0
                ],
                "rank": 0,
                "bufferSize": 128
            },
            "attnTp": {
                "groupId": 0,
                "rankIds": [
                    0
                ],
                "rank": 0,
                "bufferSize": 128
            },
            "attnDp": {
                "groupId": 0,
                "rankIds": [
                    0
                ],
                "rank": 0,
                "bufferSize": 128
            },
            "attnInnerSp": {
                "groupId": 0,
                "rankIds": [
                    0
                ],
                "rank": 0,
                "bufferSize": 128
            },
            "attnCp": {
                "groupId": 0,
                "rankIds": [
                    0
                ],
                "rank": 0,
                "bufferSize": 128
            },
            "attnPrefixcacheCp": {
                "groupId": 0,
                "rankIds": [
                    0
                ],
                "rank": 0,
                "bufferSize": 128
            },
            "attnOProjTp": {
                "groupId": 0,
                "rankIds": [
                    0
                ],
                "rank": 0,
                "bufferSize": 128
            },
            "attnOProjDp": {
                "groupId": 0,
                "rankIds": [
                    0
                ],
                "rank": 0,
                "bufferSize": 128
            },
            "mlpTp": {
                "groupId": 0,
                "rankIds": [
                    0
                ],
                "rank": 0,
                "bufferSize": 128
            },
            "mlpDp": {
                "groupId": 0,
                "rankIds": [
                    0
                ],
                "rank": 0,
                "bufferSize": 128
            },
            "moeTp": {
                "groupId": 0,
                "rankIds": [
                    0
                ],
                "rank": 0,
                "bufferSize": 64
            },
            "moeEp": {
                "groupId": 0,
                "rankIds": [
                    0
                ],
                "rank": 0,
                "bufferSize": 512
            },
            "moeEpIntraNode": {
                "groupId": 0,
                "rankIds": [
                    0
                ],
                "rank": 0,
                "bufferSize": 128
            },
            "moeEpInterNode": {
                "groupId": 0,
                "rankIds": [
                    0
                ],
                "rank": 0,
                "bufferSize": 128
            },
            "lmHeadTp": {
                "groupId": 0,
                "rankIds": [
                    0
                ],
                "rank": 0,
                "bufferSize": 128
            },
            "lmHeadDp": {
                "groupId": 0,
                "rankIds": [
                    0
                ],
                "rank": 0,
                "bufferSize": 128
            },
            "denseTp": {
                "groupId": 0,
                "rankIds": [
                    0
                ],
                "rank": 0,
                "bufferSize": 128
            },
            "dynamicEplb": {
                "groupId": 0,
                "rankIds": [
                    0
                ],
                "rank": 0,
                "bufferSize": 128
            }
        },
        "isPrefill": true,
        "enableLcoc": true,
        "enableSplitFuse": false,
        "enableMC2": false,
        "linearDescs": [
            [
                1,
                -1,
                -1,
                1,
                1,
                -1,
                1
            ]
        ],
        "enableRopeQuantKvcache": false,
        "layerwiseMode": 1,
        "startLayerId": 62,
        "endLayerId": 63,
        "layerwiseDisaggregated": true,
        "reuseEmbedTable": true,
        "outputEmbedTable": false
    })";
    atb_speed::TestableQwenDecoderModel decoderModel(param);
    EXPECT_EQ(decoderModel.CanUseEmbedTable(), true);
    EXPECT_EQ(decoderModel.NeedOutputEmbedTable(), false);

    decoderModel.Construct();
    const std::map<std::string, uint32_t>& inTensorMap = decoderModel.GetIntensorMap();
    const std::map<std::string, uint32_t>& internalTensorMap = decoderModel.GetInternalTensorMap();
    const std::map<std::string, uint32_t>& outTensorMap = decoderModel.GetOutTensorMap();

    EXPECT_EQ(inTensorMap.at("cosine_embed_table"), inTensorMap.size() - 2);
    EXPECT_EQ(inTensorMap.at("sine_embed_table"), inTensorMap.size() - 1);

    auto it = internalTensorMap.find("cosine_embed_table");
    EXPECT_EQ(it, internalTensorMap.end());

    it = outTensorMap.find("cosine_embed_table");
    EXPECT_EQ(it, outTensorMap.end());
}
} // namespace atb_speed