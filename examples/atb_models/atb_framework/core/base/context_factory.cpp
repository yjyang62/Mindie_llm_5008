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
#include "atb_speed/base/context_factory.h"
#include <thread>
#include "atb_speed/log.h"
#include "atb_speed/utils/singleton.h"
#include "atb_speed/utils/config.h"

namespace atb_speed {

const int MAX_STREAM_NUM = 2;

thread_local std::shared_ptr<atb::Context> g_localContext;

bool ContextFactory::cacheWorkspace_ = false;

std::vector<aclrtStream> ContextFactory::GetSubStreams()
{
    static std::vector<aclrtStream> streams;
    static bool initialized = false;
    if (!initialized) {
        aclError ret = aclInit(nullptr);
        if (ret != ACL_SUCCESS && ret != ACL_ERROR_REPEAT_INITIALIZE) {
            ATB_SPEED_LOG_WARN("Failed to aclInit: " << ret);
        }

        for (int i = 0; i < MAX_STREAM_NUM; ++i) {
            aclrtStream subStream;
            
            ret = aclrtCreateStream(&subStream);
            if (ret != ACL_SUCCESS) {
                ATB_SPEED_LOG_ERROR("Failed to create aclrtStream: " << ret);
            }
            ret = aclrtSetStreamFailureMode(subStream, ACL_STOP_ON_FAILURE);
            if (ret != 0) {
                ATB_SPEED_LOG_ERROR("Failed to aclrtSetStreamFailureMode: " << ret);
            }
            streams.push_back(subStream);
        }
        initialized = true;
    }

    return streams;
}

std::shared_ptr<atb::Context> ContextFactory::GetAtbContext(void *stream)
{
    if (g_localContext) {
        ATB_SPEED_LOG_DEBUG("ContextFactory return localContext");
        return g_localContext;
    }
    ATB_SPEED_LOG_DEBUG("ContextFactory create atb::Context start");
    atb::Context *context = nullptr;
    atb::Status st = atb::CreateContext(&context);
    if (st != 0) {
        ATB_SPEED_LOG_ERROR("ContextFactory create atb::Context fail");
    }

    if (context) {
        context->SetExecuteStream(stream);
        if (atb_speed::GetSingleton<atb_speed::Config>().IsUseTilingCopyStream()) {
            ATB_SPEED_LOG_DEBUG("ContextFactory use tiling copy stream");
            context->SetAsyncTilingCopyStatus(true);
        } else {
            ATB_SPEED_LOG_DEBUG("ContextFactory not use tiling copy stream");
        }
    }

    std::shared_ptr<atb::Context> tmpLocalContext(context, [](atb::Context* context) {atb::DestroyContext(context);});
    g_localContext = tmpLocalContext;

    return g_localContext;
}

void ContextFactory::FreeAtbContext()
{
    ATB_SPEED_LOG_DEBUG("ContextFactory FreeAtbContext start.");
    if (!g_localContext) {
        return;
    }
    
    ATB_SPEED_LOG_DEBUG("ContextFactory localContext use_count: " << g_localContext.use_count());
    if (g_localContext.use_count() != 1) {
        return;
    }
    ATB_SPEED_LOG_DEBUG("ContextFactory localContext reset.");
    g_localContext.reset();
}
}