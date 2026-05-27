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


import numpy as np
import torch
from utils import ZipMoEConfig
import time
import safetensors
from safetensors import safe_open
from pathlib import Path
from typing import Dict, List, Tuple, Any
import json



class ExpertExecutor:
    def __init__(self, zipmoe_config: ZipMoEConfig):
        self.zipmoe_config = zipmoe_config
        
    def set_expert_dispatcher( self, expert_dispatcher ):
        self.expert_dispatcher = expert_dispatcher
        
    def set_expert_predictor( self, expert_predictor ):
        self.expert_predictor = expert_predictor
    
    def dispatch_local(
        self, 
        hidden_states, router_mask, layer_id
    ):
        num_expert = router_mask.shape[-1]
        expert_count = (
            torch.sum(router_mask.view((-1, num_expert)), dim=0)
            .cpu()
            .numpy()
            .flatten()
        )
        expert_list = (
            np.arange(num_expert).astype(int)[expert_count > 0].tolist()
        )
        num_activated_experts = len(expert_list)
        
        self.expert_dispatcher.set_inputs(hidden_states, router_mask)
        self.expert_dispatcher.set_expected_queue(num_activated_experts)
        expert_token_count_list = expert_count[expert_count > 0].tolist()

        for i, expert_id in enumerate(expert_list):
            num_token_reqs = expert_token_count_list[i]
            self.expert_dispatcher.enqueue_expert(
                layer_id,
                expert_id,
                num_token_reqs
            )

        if self.expert_predictor is not None:   
            self.expert_predictor.update(layer_id, expert_list)

        self.expert_dispatcher.ops_schedule()
        result = self.expert_dispatcher.wait_expert()
        
        if self.expert_predictor is not None:
            priority = 1
            next_layer_id, predicted_expert_id_list = self.expert_predictor.predict_next_layer(layer_id)
            for predicted_expert_id in predicted_expert_id_list:
                self.expert_dispatcher.submit_prefetch(
                    next_layer_id,
                    predicted_expert_id,
                    priority
                )
                priority += 1   
        return result
        
        