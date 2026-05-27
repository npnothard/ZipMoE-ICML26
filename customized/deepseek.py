# Copyright (c) 2026 <ZipMoE / MINT, Nanjing University>.
# All rights reserved.
#
# Modifications and additions to this file are licensed under the
# Academic Non-Commercial License. See the LICENSE file in the
# project root for details.
#
# -------------------------------------------------------------------
# DERIVED FROM:
# EfficientMoE (Apache License 2.0)
# Copyright (c) EfficientMoE.
#
# The original code is licensed under the Apache License, Version 2.0.
# This file contains substantial modifications.
# -------------------------------------------------------------------


from typing import Dict, Optional, Tuple
from experts.expert_executor import ExpertExecutor
import torch
import torch.nn as nn
import torch.nn.functional as F
import threading
import nvtx
import time
import safetensors
from safetensors import safe_open
from pathlib import Path
from typing import Dict, List, Tuple, Any
import json



class DeepseekMoEBlock(nn.Module):
    def __init__ (self, config):
        super().__init__()
        self.config = config
        self.num_experts_per_tok = config.num_experts_per_tok
        if self.config.model_type == "deepseek_v2":
            from models.deepseek import DeepseekV2MLP, MoEGate
            self.mlp_cls = DeepseekV2MLP
            self.gate_cls = MoEGate
        self.experts = nn.ModuleList(
            [
                self.mlp_cls(
                    config,
                    intermediate_size = config.moe_intermediate_size
                )
                for i in range(config.n_routed_experts)
            ]
        )
        self.gate = self.gate_cls(config)
        if config.n_shared_experts is not None:
            intermediate_size = (
                config.moe_intermediate_size * config.n_shared_experts
            )
            self.shared_experts = self.mlp_cls(
                config,
                intermediate_size = intermediate_size
            )
        # Inject customized properties
        self.zipmoe_engine = None
        self.expert_tensor_ids: Dict[int, int] = None
        self.expert_executor: ExpertExecutor = None

    @nvtx.annotate("DeepseekMOEBlock", color="blue")
    def forward( self, hidden_states ):
        identity = hidden_states
        orig_shape = hidden_states.shape
        gate_output = self.gate(hidden_states)
        if len(gate_output) == 3:
            topk_idx, topk_weight, aux_loss = gate_output
        else:
            topk_idx, topk_weight = gate_output
            aux_loss = None
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        batch_size, sequence_length, hidden_dim = orig_shape
        router_mask = F.one_hot(
            topk_idx,
            num_classes = self.config.n_routed_experts
        )
        routing_weights_mask = (topk_weight[:,:,None]*router_mask).permute(
            0 , 2 , 1
        )
        routing_weights_mask = torch.sum(routing_weights_mask, dim = -1)
        router_mask = router_mask.permute(0, 2, 1)
        for i in range(self.config.num_experts_per_tok):
            router_mask[:,:,0] = torch.logical_or(
                router_mask[:,:,0],
                router_mask[:,:,i]
            )
        router_mask = router_mask[:,:,0]
        output_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device
        )
        results = self.expert_executor.dispatch_local(
            hidden_states,
            router_mask,
            self.layer_id
        )
        for output, layer_idx , expert_idx, _ in results:
            token_indices = router_mask[:, expert_idx].bool()
            output_hidden_states[token_indices,:] += (
                output.to(routing_weights_mask.device) * routing_weights_mask[token_indices, expert_idx][:,None]
            )
        output_hidden_states = output_hidden_states.view(
            batch_size, sequence_length, hidden_dim
        )
        if self.config.n_shared_experts is not None:
            shared_out = self.shared_experts(
                identity
            )
            output_hidden_states = output_hidden_states + shared_out
        return output_hidden_states
