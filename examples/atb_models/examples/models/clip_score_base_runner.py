#!/usr/bin/env python
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import argparse
import json
import os
import time
import open_clip
import torch
import torch_npu
import torch.nn.functional as F

from atb_llm.utils.file_utils import safe_open, check_file_safety
from atb_llm.utils.log import logger
from PIL import Image


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device_ids",
        type=str,
        required=True,
        help="device for torch.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="open clip model name",
    )
    parser.add_argument(
        "--model_weights_path",
        type=str,
        required=True,
        help="open clip model weights",
    )
    parser.add_argument(
        "--image_info",
        type=str,
        required=True,
        help="Image_info.json file.",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="dataset path for precision test.",
    )
    return parser.parse_args()


class ClipScoreBaseRunner:
    def __init__(self, args):
        self.args = args

    @staticmethod
    def clip_score(model_clip, tokenizer, preprocess, model_answer, image_file):

        img = preprocess(Image.open(image_file)).unsqueeze(0).npu()
        text = tokenizer([model_answer]).npu()

        with torch.no_grad():
            text_ft = model_clip.encode_text(text).float()
            img_ft = model_clip.encode_image(img).float()
            score = F.cosine_similarity(img_ft, text_ft).squeeze()

        return score.cpu()

    def set_torch_env(self):
        torch_npu.npu.set_device(int(self.args.device_ids))
        torch.npu.set_compile_mode(jit_compile=False)

    def run(self):
        args = self.args
        self.set_torch_env()

        t_b = time.time()
        logger.info("Load clip model...")
        check_file_safety(args.model_weights_path, is_check_file_size=False)
        model_clip, _, preprocess = open_clip.create_model_and_transforms(
            args.model_name,
            pretrained=args.model_weights_path,
            device=f"npu:{args.device_ids}",
        )
        model_clip.eval()
        logger.info(f">done. elapsed time: {(time.time() - t_b):.3f} s")

        tokenizer = open_clip.get_tokenizer("ViT-H-14")
        with safe_open(args.image_info, "r", override_flags=os.O_RDONLY) as f:
            image_info = json.load(f)

        t_b = time.time()

        logger.info("Calc clip score...")
        all_scores = []
        for image_file, model_answer in image_info.items():
            # 单个图片  单个answer
            check_file_safety(args.dataset_path, is_check_file_size=False)
            image_file_path = os.path.join(args.dataset_path, image_file)
            logger.info(f"cur image file: {image_file_path}")
            image_score = self.clip_score(
                model_clip, tokenizer, preprocess, model_answer, image_file_path
            )
            logger.info(f"{image_score=}")
            all_scores.append(image_score)
        all_scores_mean = torch.mean(torch.tensor(all_scores))
        logger.info(f"平均分：{all_scores_mean=}")
        logger.info(f">done. elapsed time: {(time.time() - t_b):.3f} s")


if __name__ == "__main__":
    arguments = parse_arguments()
    runner = ClipScoreBaseRunner(arguments)
    runner.run()
