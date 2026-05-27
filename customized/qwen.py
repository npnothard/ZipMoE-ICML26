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


import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.qwen2_moe.modeling_qwen2_moe import Qwen2MoeMLP
class Qwen2MoEBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob 
        self.gate = nn.Linear(
            config.hidden_size, config.num_experts, bias=False
        )
        self.experts = nn.ModuleList(
            [
                Qwen2MoeMLP(
                    config, intermediate_size=config.moe_intermediate_size
                )
                for _ in range(self.num_experts)
            ]
        )
        self.shared_expert = Qwen2MoeMLP(
                config, intermediate_size=config.shared_expert_intermediate_size
        )
        self.shared_expert_gate = torch.nn.Linear(config.hidden_size, 1, bias=False)
    def __prepare_expert_route(self, hidden_states):
        router_logits = self.gate(hidden_states)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(
            routing_weights, self.top_k, dim=-1
        )
        if self.norm_topk_prob:
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(hidden_states.dtype)
        return selected_experts, routing_weights, router_logits

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        identity = hidden_states
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        shared_expert_output = self.shared_expert(hidden_states)
        topk_idx, topk_weight, logits = (
            self.__prepare_expert_route(hidden_states)
        )
        router_mask = F.one_hot(
            topk_idx, num_classes=self.num_experts
        )
        routing_weights_mask = (topk_weight[:, :, None] * router_mask).permute(
            0, 2, 1
        )
        routing_weights_mask = torch.sum(routing_weights_mask, dim=-1)
        router_mask = router_mask.permute(0, 2, 1)
        for i in range(self.top_k):
            router_mask[:, :, 0] = torch.logical_or(
                router_mask[:, :, 0], router_mask[:, :, i]
            )
        router_mask = router_mask[:, :, 0]
    
        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        results = self.expert_executor.dispatch_local(
            hidden_states, 
            router_mask, 
            self.layer_id
        )

        for output, _, idx, _ in results:
            token_indices = router_mask[:, idx].bool()
            final_hidden_states[token_indices, :] += (
                output.to(routing_weights_mask.device)
                * routing_weights_mask[token_indices, idx][:, None]
            )

        final_hidden_states = final_hidden_states.view(
            batch_size, sequence_length, hidden_dim
        )
        shared_expert_output = F.sigmoid(self.shared_expert_gate(hidden_states))*shared_expert_output
        final_hidden_states = final_hidden_states + shared_expert_output.view(batch_size, sequence_length, hidden_dim)
    
        return final_hidden_states