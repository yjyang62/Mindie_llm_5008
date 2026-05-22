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

#ifndef ATB_SPEED_MODEL_TASK_EXECUTOR_H
#define ATB_SPEED_MODEL_TASK_EXECUTOR_H

#include <acl/acl.h>
#include <thread>
#include <mutex>
#include <deque>
#include <map>
#include <functional>

#include "atb_speed/utils/task_queue.h"
#include "atb_speed/log.h"

namespace atb_speed {

class ModelTaskExecutor {
public:
    struct Worker {
        bool stop = false;
        std::thread thread;
        TaskQueue queue;
        int deviceIdx = -1;
    };

public:
    static ModelTaskExecutor& Instance()
    {
        static ModelTaskExecutor instance;
        return instance;
    }

public:
    ~ModelTaskExecutor();

    void PushTask(int idx, const Task &task);

private:
    ModelTaskExecutor() {}

    void WorkerThread(int workerId);

private:
    std::mutex mutex_;
    std::deque<Worker> workers_;
    std::map<int, uint32_t> idx2worker_;
};
}  // namespace atb_speed

#endif  // ATB_SPEED_MODEL_TASK_EXECUTOR_H
