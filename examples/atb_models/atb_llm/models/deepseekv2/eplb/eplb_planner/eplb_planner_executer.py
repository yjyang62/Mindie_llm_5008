# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import queue


class EplbPlannerExecuter(object):
    def __init__(self):
        self.prepare_queue = queue.Queue()
        self.load_queue = queue.Queue()

    #for loader thread
    def set_load_prepare_done_and_wait(self):
        self.prepare_queue.put(1)
        self.load_queue.get()

    #for forward thread
    def is_load_prepare_done(self):
        return not self.prepare_queue.empty()

    #for forward thread
    def set_load_deploy_done_and_notify(self):
        self.load_queue.put_nowait(1)


