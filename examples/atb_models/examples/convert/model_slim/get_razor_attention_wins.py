# Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
def get_global_wins(model_type, num_layers):
    prefix_matching = "prefix_matching"
    copying = "copying"
    first_sink = None
    head_dict = None
    if model_type == 'llama' and num_layers == 80:
        first_sink = 40
        head_dict = {
            prefix_matching: {
                0: [0, 1, 2, 3, 4, 5, 6, 7], 1: [0, 1, 2, 3, 4, 5, 6, 7],
                18: [5, 0, 2, 6, 1, 7], 76: [7, 1, 5, 3], 35: [5, 3, 2, 0, 6, 4],
                74: [0, 3, 2, 1, 5], 52: [2, 7, 4, 6, 3], 56: [0, 5, 2, 3, 4],
                77: [6, 0, 5, 1, 7, 4], 64: [3, 6], 33: [2, 1, 5, 4], 53: [0, 2],
                37: [3, 4, 1, 2, 0, 5], 54: [1, 3], 21: [4, 6, 7, 3, 2], 47: [2],
                72: [4], 31: [2, 4, 5, 3, 0, 6], 44: [5, 0], 67: [7, 6],
                22: [1, 6, 4, 3], 68: [6, 2, 1, 7], 23: [5, 1, 4, 2, 6],
                71: [0, 6, 4, 1], 39: [5, 1, 4], 36: [6, 5, 1, 2, 0],
                27: [4, 2, 6, 5, 1], 73: [0, 4, 7, 5, 2, 1], 30: [6, 4, 5, 3],
                14: [4, 2], 38: [6, 2, 3, 7], 60: [2, 3, 7, 6], 34: [0, 3, 4, 5],
                41: [7, 2, 4], 19: [4, 0], 69: [4, 5], 29: [1, 6, 3, 7],
                75: [2, 3, 0, 4, 5, 1], 61: [0, 4], 49: [0],
                25: [1, 7, 4], 57: [0, 1, 3, 6], 17: [6, 3, 7, 1, 2], 58: [0],
                24: [0, 5, 3], 32: [2, 5, 6], 42: [3], 55: [0], 70: [1, 6], 28: [5, 1, 0],
                48: [7, 6], 50: [4], 16: [4, 5], 7: [3, 0], 63: [7, 4], 51: [3], 78: [0],
                5: [5], 59: [4], 26: [3, 4, 5], 66: [0], 15: [2], 40: [4], 43: [2],
                45: [0], 6: [4]},
            copying: {
                14: [2], 17: [7], 74: [0], 33: [2], 52: [2], 15: [2],
                31: [4], 38: [6], 71: [0], 23: [5], 27: [4, 2], 30: [4, 6],
                19: [0], 77: [5], 75: [3], 47: [2], 21: [6]}
        }
    elif model_type == 'chatglm' and num_layers == 40:
        first_sink = 4
        head_dict = {
            prefix_matching: {
                19: [0, 1],
                20: [2],
                21: [0, 1, 3],
                22: [3],
                23: [0, 1, 3],
                24: [1, 2, 3],
                25: [2, 3],
                26: [2, 3],
                28: [0, 1, 2, 3],
                29: [0, 1, 2, 3],
                31: [2, 3],
                32: [0, 1, 2, 3],
                33: [0, 1, 2, 3],
                34: [0, 1, 2, 3],
            },
            copying: {5: [2], 6: [3], 9: [0], 10: [0], 12: [2], 13: [1], 14: [2], 15: [2], 16: [2]}
        }
    return head_dict, first_sink