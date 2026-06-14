# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
import os

from atb_llm.runner.model_runner import ModelRunner
from atb_llm.utils.log import logger, print_log
from atb_llm.utils.env import ENV
from atb_llm.utils import file_utils
from atb_llm.utils.file_utils import MAX_PATH_LENGTH
from atb_llm.utils.argument_utils import ArgumentParser, StringArgumentValidator
from examples.convert.convert_utils import copy_tokenizer_files, modify_config


MAX_KW_ARGS_LENGTH = 4096


class WeightSharder:
    def __init__(self, **kwargs):
        self.rank = kwargs.get('rank', '0')
        self.world_size = kwargs.get('world_size', '1')
        self.local_rank = kwargs.get('local_rank', self.rank)
        self.model_path = kwargs.get('model_path', None)
        self.save_directory = kwargs.get('save_directory', None)
        if self.save_directory is not None:
            self.save_directory = file_utils.standardize_path(self.save_directory, check_link=False)
            file_utils.check_path_permission(self.save_directory)
        self.trust_remote_code = kwargs.get('trust_remote_code', False)
        kw_args = {}
        if 'num_speculative_tokens' not in kw_args:
            kw_args['num_speculative_tokens'] = int(ENV.deepseek_mtp)

        ENV.auto_transpose_enable = False

        self.model = ModelRunner(
            self.model_path, 
            rank=self.rank, 
            world_size=self.world_size,
            local_rank=self.local_rank,
            dp=kwargs.get("dp", -1),
            tp=kwargs.get("tp", -1),
            moe_tp=kwargs.get("moe_tp", -1),
            moe_ep=kwargs.get("moe_ep", -1),
            sp=kwargs.get("sp", -1),
            cp=kwargs.get("cp", -1),
            trust_remote_code=self.trust_remote_code,
            **kw_args)
        self.model.load_weights()
        print_log(self.rank, logger.info, "Weight load success.")

    def save_sharded(self):
        self.model.save_sharded(save_directory=self.save_directory)
        if self.local_rank == 0:
            modify_config(self.model_path, self.save_directory, self.model.dtype, self.model.quantize, is_exist_ok=True)
            copy_tokenizer_files(self.model_path, self.save_directory)


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('--model_path', type=str, help="model and tokenizer path",
                        validator=StringArgumentValidator(min_length=1, max_length=MAX_PATH_LENGTH))
    parser.add_argument('--save_directory', type=str, required=True,
                        validator=StringArgumentValidator(min_length=1, max_length=MAX_PATH_LENGTH))
    parser.add_argument('--trust_remote_code', action='store_true')
    parser.add_argument('--dp', type=int, default=-1)
    parser.add_argument('--tp', type=int, default=-1)
    parser.add_argument('--moe_tp', type=int, default=-1)
    parser.add_argument('--moe_ep', type=int, default=-1)
    parser.add_argument('--sp', type=int, default=-1)
    parser.add_argument('--cp', type=int, default=-1)

    return parser.parse_args()


def main():
    os.environ["ATB_LLM_ENABLE_AUTO_TRANSPOSE"] = "0"
    args = parse_arguments()

    rank = ENV.rank
    world_size = ENV.world_size
    local_rank = ENV.local_rank
    
    input_dict = {
        'rank': rank,
        'world_size': world_size,
        'local_rank': local_rank,
        **vars(args)
    }

    model_path = args.model_path
    save_directory = args.save_directory
    if not os.path.exists(save_directory):
        os.makedirs(save_directory, exist_ok=True)

    weight_sharder = WeightSharder(**input_dict)
    weight_sharder.save_sharded()


if __name__ == '__main__':
    main()