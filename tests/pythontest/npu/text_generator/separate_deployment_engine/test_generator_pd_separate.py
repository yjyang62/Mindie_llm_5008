# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
import time
import multiprocessing
from multiprocessing import Queue
from dataclasses import dataclass
from pathlib import Path
import configparser
import numpy as np
from mindie_llm.utils.status import MindieLlmStatusCode

from mindie_llm.utils.log.logging import logger
from mindie_llm.text_generator.utils.generation_metadata import GenerationParams
from mindie_llm.text_generator.utils.request import Request
from mindie_llm.text_generator.utils.input_metadata import InputMetadata
from mindie_llm.examples.scheduler import Scheduler
from mindie_llm.examples.scheduler import decode_token
from mindie_llm.examples.run_generator import GeneratorRunner, parse_arguments


@dataclass
class PdInput:
    inputs: list[str]
    batch_size: int
    sampling_params: np.ndarray
    is_chat_model: bool


class PdScheduler:
    @staticmethod
    def generate(generator_runner, pd_input, queue):
        input_ids, adapter_ids = generator_runner.build_model_inputs(pd_input.inputs, pd_input.is_chat_model)

        max_new_tokens = generator_runner.max_new_tokens if generator_runner.max_new_tokens \
            else generator_runner.max_output_length
        generation_params = GenerationParams(
            best_of=generator_runner.best_of,
            ignore_eos=generator_runner.request_ignore_eos,
            include_stop_str_in_output=generator_runner.include_stop_str_in_output,
            max_new_tokens=max_new_tokens,
            skip_special_tokens=generator_runner.skip_special_tokens,
            stop_strings=generator_runner.stop_strings,
            stop_token_ids=generator_runner.stop_token_ids
        )
        req_list = [Request.request_from_token(input_ids[0], pd_input.sampling_params, generation_params,
                                               req_id=i) for i in range(pd_input.batch_size)]
        generation_params.adapter_id = adapter_ids[0]
        scheduler = Scheduler(generator_runner.max_batch_size, generator_runner.max_prefill_tokens,
            generator_runner.generator, generator_runner.load_tokenizer, False, 0)
        
        if scheduler.generator.pd_config.model_role == 'prefill':
            req_begin_idx = 0
            while req_begin_idx < len(req_list):
                prefill_requests, end_idx = scheduler.get_prefilling_requests(req_list, req_begin_idx)
                req_begin_idx = end_idx + 1

                if prefill_requests:
                    logger.info('=============prefill')
                    generation_output = scheduler.generator.prefill(prefill_requests)
                    logger.info('=============prefill success')
                    scheduler.parse_outputs(prefill_requests, generation_output, is_prefill=True)
                first_token = prefill_requests[0].sequences[0].out_token_list[0]
                logger.info(f"Prefill第一条请求生成的首token:{first_token}")

                queue.put(prefill_requests)
                logger.info(f"prefill Send message: {prefill_requests}")

                
        elif scheduler.generator.pd_config.model_role == 'decoder':
            logger.info('wait prefill success')
            prefill_requests = queue.get()
            
            logger.info(f"decoder Received message: {prefill_requests}")
            logger.info(f"prefill_requests num: {len(prefill_requests)}")

            block_tables = np.asarray([request.block_tables for request in prefill_requests])
            input_metadata = InputMetadata.from_requests(prefill_requests, block_tables, True)  
                    
            logger.info('=============pull kv start')
            scheduler.generator.pull_kv(input_metadata, [(2, block_tables.tolist()[0], block_tables.tolist()[0])])
            logger.info('=============pull kv success')
            decoder_requests = scheduler.get_decoding_requests(prefill_requests)
            logger.info(f"decoder_requests num: {len(decoder_requests)}")
            
            while decoder_requests:
                generation_output = scheduler.generator.decode(decoder_requests)
                scheduler.parse_outputs(decoder_requests, generation_output, False)
                decoder_requests = scheduler.get_decoding_requests(prefill_requests)
        decode_text_list, token_num_list, cumulative_logprobs, logprobs_list, top_logprobs_list = (
            decode_token(prefill_requests, scheduler.tokenizer))
        logger.info(f"============decode_text_list：{decode_text_list}")
        return decode_text_list, token_num_list


def test_separate_deployment_engine(role: str, queue):

    #定义配置文件的路径
    script_path = Path(__file__)
    config_path = script_path.parent / 'pd_config.ini'

    #创建configparser对象
    conf = configparser.ConfigParser()

    #检查配置文件是否存在
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    #读取配置文件
    conf.read(str(config_path))

    #获取配置项
    deployment = 'Deployment'
    decoder_ip = conf[deployment]['decoder_ip']
    prefill_ip = conf[deployment]['prefill_ip']
    model_path = conf[deployment]['model_path']

    decoder_logic_device_id = int(conf[deployment]['decoder_logic_device_id'])
    prefill_logic_device_id = int(conf[deployment]['prefill_logic_device_id'])
    decoder_physical_device_id = int(conf[deployment]['decoder_physical_device_id'])
    prefill_physical_device_id = int(conf[deployment]['prefill_physical_device_id'])
    decoder_model_instance_id = int(conf[deployment]['decoder_model_instance_id'])
    prefill_model_instance_id = int(conf[deployment]['prefill_model_instance_id'])
    prefill_host_ip = conf[deployment]['prefill_host_ip']
    decoder_host_ip = conf[deployment]['decoder_host_ip']
    prefill = 'prefill'
    decoder = 'decoder'
    args = parse_arguments()

    if role == decoder:
        model_role = decoder
        npu_device_id = decoder_logic_device_id
        local_physical_device_id = decoder_physical_device_id
        local_model_instance_id = decoder_model_instance_id
        local_device_ip = decoder_ip
        local_host_ip = decoder_host_ip
        remote_model_instance_ids = [prefill_model_instance_id]
        remote_device_ip = prefill_ip
    elif role == prefill:
        model_role = prefill
        npu_device_id = prefill_logic_device_id
        local_physical_device_id = prefill_physical_device_id
        local_model_instance_id = prefill_model_instance_id
        local_device_ip = prefill_ip
        local_host_ip = prefill_host_ip
        remote_model_instance_ids = [decoder_model_instance_id]
        remote_device_ip = decoder_ip
    
    input_dict = {
        'rank': 0,
        'world_size': 1,
        'local_rank': 0,
        'model_role': model_role,
        'npu_id': npu_device_id,
        'local_model_instance_id': local_model_instance_id,
        'local_device_ip': local_device_ip,
        'remote_model_instance_ids': remote_model_instance_ids,
        'local_physical_device_id': local_physical_device_id,
        'local_host_ip': local_host_ip,
        'max_position_embeddings': 4096,
        'kv_trans_timeout': 5,
        **vars(args)
    }

    input_dict['model_path'] = model_path
    input_dict['npu_mem'] = 4

    generator_runner = GeneratorRunner(**input_dict)
    pd_scheduler = PdScheduler()
    if role == prefill:
        sampling_params_ins = None
        pd_input = PdInput(['what is deep learning?'], 1, sampling_params_ins, False)
        generate_res, token_nums = pd_scheduler.generate(generator_runner, pd_input, queue)

    if generator_runner.generator.pd_config.model_role == prefill:
        target_role = decoder
        link_result = generator_runner.generator.link(
            remote_cluster_ids={0: [decoder_model_instance_id]},
            remote_physical_device_ids={0: [decoder_physical_device_id]},
            remote_device_ips={0: [remote_device_ip]},
            host_ips={0: [decoder_host_ip]}
        )
        if link_result != MindieLlmStatusCode.SUCCESS:
            logger.error(f"{role}:link operation faild with the error code {link_result}")
            raise Exception(f"{role}:link operation faild with the error code {link_result}")
        logger.info(f"{role} link success to remote cluster {decoder_model_instance_id}")    
    elif generator_runner.generator.pd_config.model_role == decoder:
        target_role = prefill
        link_result = generator_runner.generator.link(
            remote_cluster_ids={0: [prefill_model_instance_id]},
            remote_physical_device_ids={0: [prefill_physical_device_id]},
            remote_device_ips={0: [remote_device_ip]},
            host_ips={0: [prefill_host_ip]}
        )
        if link_result != MindieLlmStatusCode.SUCCESS:
            logger.error(f"{role}:link operation faild with the error code {link_result}")
            raise Exception(f"{role}:link operation faild with the error code {link_result}")
        logger.info(f"{role} link success to remote cluster {prefill_model_instance_id}")   
        
    if role == decoder:
        sampling_params_ins = None
        pd_input = PdInput(['what is deep learning?'], 1, sampling_params_ins, False)
        generate_res, token_nums = pd_scheduler.generate(generator_runner, pd_input, queue)
        

    if target_role == decoder:
        logger.info("==>decoder finish")
        queue.put('quit')
    if target_role == prefill:
        time.sleep(10)
        while True:
            msg = queue.get()
            logger.info(f"prefill Received message: {msg}")
            if msg == 'quit':
                break

if __name__ == '__main__':
    q = Queue()
    p1 = multiprocessing.Process(target=test_separate_deployment_engine, args=('prefill', q,))
    p2 = multiprocessing.Process(target=test_separate_deployment_engine, args=('decoder', q,))
    p1.start()
    p2.start()
    p1.join()
    p2.join()