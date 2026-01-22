# Copyright (c) 2026 <ZipMoE / Anonymous Team>.
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


from typing import Dict
from experts.expert_executor import ExpertExecutor

import torch
import torch.nn as nn
from transformers import SwitchTransformersConfig
from transformers.activations import ACT2FN
from transformers.models.switch_transformers.modeling_switch_transformers import (
    SwitchTransformersDenseActDense,
    SwitchTransformersTop1Router,
)

from utils import ZipMoEConfig

GPU_IDX_COUNTER = 0

class SwitchTransformersDenseGatedActDense(nn.Module):
    def __init__(self, config: SwitchTransformersConfig):
        super().__init__()
        self.wi_0 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.wi_1 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.wo = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.act = ACT2FN[config.dense_act_fn]

    def forward(self, hidden_states):
        hidden_gelu = self.act(self.wi_0(hidden_states))
        hidden_linear = self.wi_1(hidden_states)
        hidden_states = hidden_gelu * hidden_linear
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.wo(hidden_states)
        return hidden_states


class SyncSwitchTransformersSparseMLP(nn.Module):
    zipmoe_config: ZipMoEConfig = None
    layer_id: int = None

    def __init__(
        self,
        config: SwitchTransformersConfig,
        expert_class: nn.Module = SwitchTransformersDenseActDense,
    ):
        super().__init__()
        self.router = SwitchTransformersTop1Router(config)
        if config.model_type == "switch_transformers" and config.d_ff == 10240:
            expert_class = SwitchTransformersDenseGatedActDense
        self.experts = nn.ModuleDict()
        for idx in range(config.num_experts):
            self.experts[f"expert_{idx}"] = expert_class(config)
        self.expert_tensor_ids: Dict[int, int] = None
        self.expert_executor: ExpertExecutor = None
        self.expert_prefetcher = None
        
    def forward(self, hidden_states):
        router_mask, router_probs, router_logits = self.router(hidden_states)
        expert_index = torch.argmax(router_mask, dim=-1)
        next_states = hidden_states.clone()
        batch_size = hidden_states.shape[0]
        expert_index = expert_index.reshape(batch_size, -1)
        results = self.expert_executor.dispatch_local(
            hidden_states, 
            router_mask, 
            self.layer_id
        )
        for output, _, idx, _ in results:
            token_indices = router_mask[:, :, idx].bool()
            next_states[token_indices] = output.to(next_states.device)
        hidden_states = router_probs * next_states
        return hidden_states, (
            router_logits.to("cuda:0", non_blocking=True),
            expert_index.to("cuda:0", non_blocking=True),
        )