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
import torch
from tqdm import tqdm
from atb_llm.utils.file_utils import safe_open, check_file_safety
from atb_llm.utils.log import logger

torch.manual_seed(1234)
OUTPUT_JSON_PATH = "./coco_predict.json"


class CocoBaseRunner:
    def __init__(self, model_path, image_path, **kwargs):
        parser = argparse.ArgumentParser(description="Demo")

        parser.add_argument(
            "--model_path", default=model_path, help="Model and tokenizer path."
        )
        parser.add_argument(
            "--image_path", default=image_path, help="Image path for inference."
        )

        logger.info(f"===== model_path: {model_path}")
        logger.info(f"===== image_path: {image_path}")

        self.args = parser.parse_args()
        self.image_answer = {}
        # set save_output parameters
        self.override_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        self.encoding = "utf-8"
        self.indent = None
        self.ensure_ascii = True

    @staticmethod
    def save_output(
        sorted_dict,
        override_flags=os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        encoding="utf-8",
        indent=None,
        ensure_ascii=True,
    ):
        if not os.path.exists(OUTPUT_JSON_PATH):
            with safe_open(
                OUTPUT_JSON_PATH, "w", override_flags=override_flags, encoding=encoding
            ) as fw:
                json.dump(sorted_dict, fw, ensure_ascii=ensure_ascii, indent=indent)
        else:
            with safe_open(OUTPUT_JSON_PATH, "r") as f:
                old_data = json.load(f)
            old_data.update(sorted_dict)
            sorted_dict = dict(sorted(old_data.items()))
            with safe_open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as fw:
                json.dump(sorted_dict, fw, ensure_ascii=ensure_ascii, indent=indent)
        logger.info(f"run coco precision finish! output file: {OUTPUT_JSON_PATH}")

    # prepare model and data
    def prepare(self):
        raise NotImplementedError()

    def process(self, img_path, img_name):
        raise NotImplementedError()

    def run(self):
        model_path = self.args.model_path
        image_path = self.args.image_path
        check_file_safety(image_path, is_check_file_size=False)
        if os.path.exists(model_path) and os.path.exists(image_path):
            images_list = os.listdir(image_path)

            self.prepare()  # need customized implementation

            for _, img_name in enumerate(tqdm(images_list)):
                img_path = os.path.join(image_path, img_name)

                self.process(img_path, img_name)  # need customized implementation

            sorted_dict = dict(sorted(self.image_answer.items()))
            torch.cuda.empty_cache()
            self.save_output(
                sorted_dict,
                override_flags=self.override_flags,
                encoding=self.encoding,
                indent=self.indent,
                ensure_ascii=self.ensure_ascii,
            )

        else:
            logger.info("model_path or image_path not exist")
