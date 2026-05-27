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
from transformers import PretrainedConfig
from utils import parse_moe_param

class ExpertPrefetcher:
    first_k_dense_replace: int = 0
    
    def __init__(self, config: PretrainedConfig):
        print(config)
        self.num_layers, self.num_experts, self.num_encoder_layers = (
            parse_moe_param(config)
        )
        
    def set_zipmoe_engine(self, zipmoe_engine):
        self.zipmoe_engine = zipmoe_engine
        
    def prefetch_experts_list(
        self,
        layer_id,
        expert_list
    ):
        tensor_ids = []
        for j in expert_list:
            tensor_ids.append(
                self.expert_tensor_map[(layer_id, j)]
            )
        for tensor_id in tensor_ids:
            self.zipmoe_engine.enqueue_prefetch(
                tensor_id, 
                0
            )
    def fetch_experts_lock_cache(
        self,
        layer_id,
        expert_list
    ):
        tensor_ids = []
        for j in expert_list:
            tensor_ids.append(
                self.expert_tensor_map[(layer_id, j)]
            )
        self.zipmoe_engine.replace_cache_candidates(tensor_ids)

    def prefetch_experts(
        self,
        layer_id,
        expert_matrix
    ):
        expert_list = []
        for i in range( layer_id , self.num_layers ):
            for j in range( self.num_layers ):
                if expert_matrix[i,j] > 0:
                    expert_list.append(
                        ( self.expert_tensor_map[(i,j)] , expert_matrix )
                    )
                    
        ordered_expert_list = sorted(
            expert_list,
            key = lambda x: x[1],
            reverse = True
        )
        tensor_ids = [x[0] for x in ordered_expert_list]
        assert len(np.unique(tensor_ids)) == len(tensor_ids)
        
        for tensor_id in tensor_ids:
            self.zipmoe_engine.enqueue_prefetch(
                tensor_id, 
                0
            )