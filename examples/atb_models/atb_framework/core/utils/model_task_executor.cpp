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
#include <acl/acl.h>

#include "atb_speed/utils/model_task_executor.h"

namespace atb_speed {
ModelTaskExecutor::~ModelTaskExecutor()
{
    for (auto &worker : workers_) {
        auto task = [&worker]() -> int {
            worker.stop = true;
            return 0;
        };
        worker.queue.Enqueue(task);
        worker.thread.join();
    }
}

void ModelTaskExecutor::PushTask(int idx, const Task &task)
{
    auto it = idx2worker_.find(idx);
    if (it == idx2worker_.end()) {
        std::lock_guard<std::mutex> guard(mutex_);
        it = idx2worker_.find(idx);
        if (it == idx2worker_.end()) {
            uint32_t workerId = workers_.size();
            workers_.emplace_back();
            auto &worker = workers_[workerId];
            worker.deviceIdx = idx;
            worker.thread = std::thread(&ModelTaskExecutor::WorkerThread, this, workerId);
            it = idx2worker_.insert({idx, workerId}).first;
        }
    }
    auto &worker = workers_[it->second];
    worker.queue.Enqueue(task);
    return;
}

void ModelTaskExecutor::WorkerThread(int workerId)
{
    ATB_SPEED_LOG_DEBUG("WorkerThread " << workerId << " start.");
    auto &worker = workers_[workerId];
    int ret = aclrtSetDevice(worker.deviceIdx);
    if (ret != 0) {
        ATB_SPEED_LOG_ERROR("AsdRtDeviceSetCurrent fail, error:" << ret);
    }
    while (!worker.stop) {
        auto task = worker.queue.Dequeue();
        task();
    }
    ATB_SPEED_LOG_DEBUG("WorkerThread " << workerId << " end.");
    return;
}
} // namespace atb_speed