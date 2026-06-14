# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.buque
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
"""PyTorch qwen2_audio model."""
from io import BytesIO
import numpy as np
import torchaudio
import librosa

from atb_llm.utils.log.error_code import ErrorCode
from atb_llm.utils.log.logging import logger
from ...utils.file_utils import safe_open
from ...utils.multimodal_utils import safe_open_audio

SAMPLE_RATE = 16000


def load_audio(audio_path):
    try:
        with safe_open(audio_path, 'rb') as file:
            content = file.read()
    except Exception as e:
        logger.error("`audio_path` cannot open normally or is not exist.",
        ErrorCode.ATB_MODELS_PARAM_INVALID)
        raise FileNotFoundError("`audio_path` cannot open normally or is not exist.") from e
    return content


def prepare_conversation(conversation, audio, processor, is_server=False):
    audios, audios_path = [], []
    if is_server:
        audios_path = audio
        text = conversation
    else:
        text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        for message in conversation:
            if isinstance(message["content"], list):
                for ele in message["content"]:
                    if ele["type"] == "audio":
                        audios_path.append(ele['audio_url'])
    for audio_path in audios_path:
        audios.append(
            librosa.load(
                BytesIO(load_audio(audio_path)),
                sr=processor.feature_extractor.sampling_rate)[0]
        )

    inputs = processor(text=text, audios=audios, return_tensors="pt")
    return inputs


def get_prefill_data(text, audio, processor):
    if isinstance(text, str) and isinstance(audio, str):
        audio, sr = safe_open_audio(torchaudio, audio)
        audio = torchaudio.functional.resample(audio, orig_freq=sr, new_freq=SAMPLE_RATE)
        audio = np.array(audio)[0]
        inputs = processor(text=text, audios=audio, return_tensors="pt")
    elif isinstance(text, list) and isinstance(text[0], dict):
        inputs = prepare_conversation(text, audio, processor)
    elif isinstance(text, str) and isinstance(audio, list):
        inputs = prepare_conversation(text, audio, processor, is_server=True)
    else:
        logger.error("The `text` should be str or list[dict].",
        ErrorCode.ATB_MODELS_PARAM_INVALID)
        raise ValueError("The `text` should be str or list[dict].")
    return inputs


def load_feature_by_torchaudio(audio_path):
    try:
        audio, sr = safe_open_audio(torchaudio, audio_path)
    except FileNotFoundError as e:
        raise FileNotFoundError("audio_path is not correct, please check.") from e
    audio = torchaudio.functional.resample(audio, orig_freq=sr, new_freq=SAMPLE_RATE)
    audio = np.array(audio)[0]
    return audio
