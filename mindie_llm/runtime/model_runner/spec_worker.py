# Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import functools
from abc import ABC, abstractmethod
from typing import Callable, Any, Dict, Type, Optional
import torch
from mindie_llm.utils.log.logging import logger


class BaseWorkerProxy(ABC):

    @abstractmethod
    def __getattr__(self, name):
        pass

    @abstractmethod
    def forward(self, *args, **kwargs):
        pass


class MtpWorker(BaseWorkerProxy):
    """
    MtpWorker is a proxy class that decorate the ModelRunner class to run both the main model and the draft model.

    This class handles the forward process during both the prefill and decode stages by coordinating interactions
    between the main model runner and the draft model runner.
    """
    def __init__(self, target_cls: Type, *args, **kwargs):
        self.num_speculative_tokens = kwargs.get('num_speculative_tokens', None)
        logger.info(f"The number of sepculative tokens is {self.num_speculative_tokens}. Init MtpWorker...")
        self.main_model_runner = self.create_main_model_runner(target_cls, *args, **kwargs)
        self.draft_model_runner = self.create_draft_model_runner(target_cls, *args, **kwargs)

        # This flag will be removed after support for deploying DP-in-and-DP-out is implemented.
        # When is_dp_and_server_centralized = True, it indicates that mtp logits and mtp hidden states
        # will be reprocessed for the next MTP stage."""
        self.is_dp_and_server_centralized = self.mapping.has_dp() and not self.distributed_enable
  
    def __getattr__(self, name):
        return getattr(self.main_model_runner, name)

    @staticmethod
    def create_main_model_runner(target_cls, *args, **kwargs):
        return target_cls(*args, **kwargs)
    
    @staticmethod
    def create_draft_model_runner(target_cls, *args, **kwargs):
        kwargs["is_draft_model"] = True
        return target_cls(*args, **kwargs)

    def load_weights(self):
        """
        Load the weights of main model and draft model.
        """
        self.main_model_runner.load_weights()
        self.draft_model_runner.load_weights()
    
    def update_roll_mtp_model_args(self,
                                   input_ids,
                                   positions,
                                   **kwargs):
        logits_mtp = kwargs["logits_mtp"]
        hidden_states_mtp = kwargs["hidden_states_mtp"]
        seq_lens = kwargs["seq_lens"]
        lm_head_indices = kwargs.get("lm_head_indices", None)
        next_token = logits_mtp.argmax(dim=-1)
        if lm_head_indices is not None:
            lm_head_indices = lm_head_indices
        else:
            cumsum_indices = torch.cumsum(seq_lens, dim=0)
            lm_head_indices = cumsum_indices - 1
        input_ids_mtp = torch.roll(input_ids, shifts=-1, dims=0)
        input_ids_mtp[lm_head_indices] = next_token
        positions_mtp = positions + 1
        return input_ids_mtp, positions_mtp, hidden_states_mtp

    def update_mtp_args(self, input_ids, positions, logits_mtp, hidden_states_mtp, seq_lens):
        input_ids_mtp, positions_mtp, last_hidden_states = self.update_roll_mtp_model_args(
            input_ids,
            positions,
            logits_mtp=logits_mtp,
            hidden_states_mtp=hidden_states_mtp,
            seq_lens=seq_lens)

        return input_ids_mtp, positions_mtp, last_hidden_states
    
    def get_dp_logits_and_hiddenstates_and_lmhead_indices(self, logits, hidden_states, kwargs):
        """
        Extract the logits and hidden_states by lmhead when dp > 1 and not distributed, 
        """
        dummy_data = False
        dp_logits = logits
        dp_hidden_states = hidden_states
        if self.is_dp_and_server_centralized:
            mtp_logits_gather_indices = kwargs.get('mtp_logits_gather_indices')
            if mtp_logits_gather_indices.numel() > 0:
                dp_logits = logits[mtp_logits_gather_indices]
            else:
                dummy_data = True

            shard_effective_token_indices = kwargs.get('shard_effective_token_indices')
            dp_hidden_states = hidden_states[shard_effective_token_indices]

        return dummy_data, dp_logits, dp_hidden_states

    def forward_mtp_prefill(self, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the prefill stage of MtpWork.

        Process prefill of main model first, then prefill draft model based on the final hidden_states of main model

        Only return the logits and hidden_states of main model.
        """
        # main model forward
        logits, hidden_states = self.main_model_runner.forward(**kwargs)
        hidden_states_return = hidden_states[kwargs.get('lm_head_indices')]

        torch.npu.synchronize()
        seq_lens = kwargs.get("input_lengths")
        input_ids = kwargs.get("input_ids")
        positions = kwargs.get("position_ids")
        # update mtp args
        is_dummy_data, dp_logits, dp_hidden_states = \
            self.get_dp_logits_and_hiddenstates_and_lmhead_indices(logits, hidden_states, kwargs)
        if is_dummy_data:
            input_ids_mtp, positions_mtp, hidden_states = input_ids, positions, dp_hidden_states
        else:
            input_ids_mtp, positions_mtp, hidden_states = \
                self.update_mtp_args(input_ids, positions, dp_logits, dp_hidden_states, seq_lens)
        # draft forward for updating cache.
        kwargs.update({"input_ids": input_ids_mtp})
        kwargs.update({"position_ids": positions_mtp})
        kwargs.update({"last_hidden_states": hidden_states})
        _, _ = self.draft_model_runner.forward(**kwargs)
        return (logits, hidden_states_return)

    def mtp_iter_slot_calc(self, slot_input):
        """
        Process the slot when MTP > 1.

        Split the origin slot_input for different MTP (i.e., [3,4,5,6]->[3,4,5] [4,5,6])
        """
        slot_list = []
        if self.num_speculative_tokens == 1:
            slot_list.append(slot_input)
            return slot_list
        slot_num_per_batch = self.num_speculative_tokens * 2
        used_slot_num_per_iter = self.num_speculative_tokens + 1
        offsets = torch.arange(used_slot_num_per_iter).npu()
        shift_num = (slot_input.size(0) - used_slot_num_per_iter) // slot_num_per_batch + 1
        shift_values = torch.arange(shift_num).npu()
        for mtp_idx in range(self.num_speculative_tokens):
            if slot_input.size(0) > 1:
                starts = mtp_idx + shift_values * slot_num_per_batch
                valid_starts = starts[starts + used_slot_num_per_iter <= slot_input.size(0)]
                indices = valid_starts.view(-1, 1) + offsets
                slot_new = slot_input[indices].flatten()
                slot_list.append(slot_new)
            else:
                slot_list.append(slot_input)
        return slot_list

    def forward_draft_decode(self, **kwargs) -> torch.Tensor:
        """
        Forward pass of the decode stage of draft model.

        Return the final logits computed by draft model.
        """
        all_logits_mtp = []
        draft_kwargs = kwargs.copy()
        sub_model_inputs = draft_kwargs.get("sub_model_inputs")
        input_ids_mtp = sub_model_inputs.input_ids
        positions_mtp = sub_model_inputs.position_ids
        block_tables = sub_model_inputs.block_tables
        seq_lens = sub_model_inputs.context_length
        lm_head_indices = sub_model_inputs.prefill_head_indices
        slots = sub_model_inputs.slots

        hidden_states_mtp_input = draft_kwargs.get("hidden_states")
        q_lens = draft_kwargs.get("q_lens")
        slot_list = self.mtp_iter_slot_calc(slots)
        draft_kwargs["last_hidden_states"] = hidden_states_mtp_input
        # draft forward
        for mtp_idx in range(self.num_speculative_tokens):
            slot_mapping = slot_list[mtp_idx]
            draft_kwargs.update({"input_ids": input_ids_mtp})
            draft_kwargs.update({"position_ids": positions_mtp})
            draft_kwargs.update({"input_lengths": seq_lens})
            draft_kwargs.update({"block_tables": block_tables})
            draft_kwargs.update({"slots": slot_mapping})
            draft_kwargs.update({"last_hidden_states": hidden_states_mtp_input})
            draft_kwargs.update({"lm_head_indices": lm_head_indices})

            logits_mtp, hidden_states_mtp = self.draft_model_runner.forward(**draft_kwargs)
            all_logits_mtp.append(logits_mtp)

            if mtp_idx < self.num_speculative_tokens - 1:
                is_dummy_data = False
                if self.is_dp_and_server_centralized:
                    shard_effective_token_indices = draft_kwargs.get('shard_effective_token_indices')
                    mtp_logits_gather_indices = draft_kwargs.get('mtp_logits_gather_indices')
                    if mtp_logits_gather_indices.numel() > 0:
                        logits_mtp = logits_mtp[mtp_logits_gather_indices]
                        hidden_states_mtp = hidden_states_mtp[shard_effective_token_indices]                       
                    else:
                        is_dummy_data = True

                if not is_dummy_data:
                    lm_head_indices_for_dp_update = draft_kwargs.get('lm_head_local_dp', None) \
                        if self.is_dp_and_server_centralized else lm_head_indices

                    input_ids_mtp, positions_mtp, hidden_states_mtp = self.update_roll_mtp_model_args(
                        input_ids_mtp,
                        positions_mtp,
                        logits_mtp=logits_mtp,
                        hidden_states_mtp=hidden_states_mtp,
                        seq_lens=q_lens,
                        lm_head_indices=lm_head_indices_for_dp_update)
                    
                    hidden_states_mtp_input = hidden_states_mtp

        if self.num_speculative_tokens > 1:
            all_logits_mtp = torch.stack(all_logits_mtp, dim=1)
            all_logits_mtp = all_logits_mtp.view(-1, all_logits_mtp.shape[-1])
        else:
            
            all_logits_mtp = all_logits_mtp[0][lm_head_indices]

        return all_logits_mtp

    def update_model_input_ids_by_draft(self, 
                                        input_ids, 
                                        input_ids_draft_out, 
                                        kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Update the input token id of main model by draft model output.

        Args:
            input_ids (torch.Tensor): Origin input token IDs tensor 
            input_ids_draft_out (torch.Tensor): Input token IDs generated by draft model
 
        Returns:
            tuple[torch.Tensor, torch.Tensor]: 
                - torch.Tensor: Origin input token IDs with draft out token IDs
                - torch.Tensor: Input token IDs generated by draft model
        """
        # 'draft_token_indices' indicates that the index of token for current dp group.
        # This flag will be removed after support for deploying DP-in-and-DP-out is implemented.
        if self.is_dp_and_server_centralized:
            dp_rank_ids = kwargs.get('dp_rank_ids')
            draft_token_indices = torch.where(dp_rank_ids == self.mapping.attn_dp.rank)[0]
            if draft_token_indices.numel() == 0:
                # It indicates the current device is running Dummy Data.
                return input_ids, input_ids_draft_out
        else:
            draft_token_indices = None
        input_ids_reshaped = input_ids.view(-1, self.num_speculative_tokens + 1)
        input_ids_draft_out_reshaped = input_ids_draft_out.view(-1, self.num_speculative_tokens)
        if draft_token_indices is not None:
            input_ids_draft_out_reshaped = input_ids_draft_out_reshaped[draft_token_indices]
        input_ids_reshaped[:, 1:] = input_ids_draft_out_reshaped
        input_ids = input_ids_reshaped.flatten()
        return input_ids, input_ids_draft_out 

    def forward_mtp_decode(self, **kwargs) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of the decode stage of MtpWork.

        Process decode of draft model first, then decode main model based on the token generated by draft model
        """
        sub_model_inputs = kwargs.get("sub_model_inputs", None)
        input_ids = kwargs.get("input_ids")

        if sub_model_inputs is None:
            # Note: This code is hard for warm up currently.
            logits, hidden_states = self.main_model_runner.forward(**kwargs)
            return (logits, hidden_states)

        all_logits_mtp = self.forward_draft_decode(**kwargs)
        input_ids_draft_out = torch.argmax(all_logits_mtp, dim=-1)
            
        # update args and main model forward
        input_ids, input_ids_draft_out = \
            self.update_model_input_ids_by_draft(input_ids, input_ids_draft_out, kwargs)

        kwargs.update({"input_ids": input_ids})

        logits, hidden_states = self.main_model_runner.forward(**kwargs)

        lm_head_indices = kwargs.get("lm_head_indices", None)
        if lm_head_indices is None:
            lm_head_indices = torch.tensor(range(input_ids.shape[0]),
                                            dtype=torch.int64, device=input_ids.device)
        hidden_states = hidden_states[lm_head_indices]
        return (logits, hidden_states, input_ids_draft_out)

    def forward(self, **kwargs):
        """
        Forward pass of the MtpWork, including prefill and decode
        """
        is_prefill = kwargs.get("is_prefill", True)
        if is_prefill:
            return self.forward_mtp_prefill(**kwargs)
        else:
            return self.forward_mtp_decode(**kwargs)


SelectorType = Callable[[Dict[str, Any]], Optional[Type[BaseWorkerProxy]]]


def auto_speculative_method_router(selector_fn: SelectorType):
    """
    Decorator:

    Instead of directly instantiating the original class, it first runs the selector.

    If the selector returns a proxy class, it passes the original class plus parameters 
    
    to the proxy class for instantiation.
    """
    def decorator(original_class):
        
        @functools.wraps(original_class, updated=())
        def factory(*args, **kwargs):
            proxy_class = selector_fn(kwargs)
            
            if proxy_class:
                return proxy_class(original_class, *args, **kwargs)
            else:
                return original_class(*args, **kwargs)
        for attr in dir(original_class):
            if not attr.startswith("__") and not callable(getattr(original_class, attr)):
                setattr(factory, attr, getattr(original_class, attr))
        return factory
    return decorator


def speculative_worker_selector(kwargs):
    """
    Selector function that determines whether speculative decoding should be used.

    If speculative tokens are enabled (num_speculative_tokens > 0), it returns `MtpWorker`,

    Otherwise, it returns `None`, indicating that the default worker should be used.
    """
    num_speculative_tokens = kwargs.get('num_speculative_tokens')
    
    if num_speculative_tokens > 0:
        return MtpWorker
    return None
