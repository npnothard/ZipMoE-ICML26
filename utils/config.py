# Copyright (c) 2026 <ZipMoE / MINT, Nanjing University>.
# All rights reserved.
#
# This source code is licensed under the Academic Non-Commercial License.
# See the LICENSE file in the project root for details.



import os
from dataclasses import dataclass, field

import torch
from transformers import HfArgumentParser

@dataclass
class ZipMoEConfig:
    
    offload_path: str = field(
        default = "ZIPMOE-PREFIX/ZipMoE/offload/model_name", 
        metadata = {"help": "Path to the compressed tensor binaries. [Warning] need to align with model."}
    )
    
    offload_file_name: str = field(
        default = "zipmoe_param", 
        metadata = {"help": "Name of the offloaded tensor file."}
    )
    
    code_type: str = field(
        default = "ZSTD", 
        metadata = {"help": "Name of the compression algorithm."}
    )
    
    compression_ratio: float = field(
        default = 0.36, 
        metadata = {"help": "The compression ratio of the selected code type."}
    )
    
    batch_size: int = field(
        default = 1, 
        metadata = {"help": "Batch size."}
    )
    
    trace_path: str = field(
        default = "ZIPMOE-PREFIX/ZipMoE/trace/model_name", 
        metadata = {"help": "Path to the model traces. [Warning] need to align with model."}
    )    
    
    
    caching_algorithm: str = field(
        default = "ZipMoE", 
        metadata = {"help": "Capacity of trace."}
    )
    
    device_memory_ratio: float = field(
        default = 0.95, 
        metadata = {"help": "Ratio of memory pool to total hardware capacity."}
    )
    
    gpu_pool_ratio: float = field(
        default = 0.6, 
        metadata = {"help": "Ratio of GPU pool to GPU + Pinned."}
    )
    
    decompression_delay: float = field(
        default = 1800, 
        metadata = {"help": "Decompression delay. [Warning] a func. of chunk, code, and model."}
    )    
    
    sm_io_delay: float = field(
        default = 900, 
        metadata = {"help": "SM I/O delay. [Warning] a func. of model."}
    )    
    
    
    num_compute_threads: int = field(
        default = 6, 
        metadata = {"help": "Number of compute threads."}
    )
    
    num_file_chunks: int = field(
        default = 5, 
        metadata = {"help": "Number of file chunks we split into."}
    )
    
    
    prefetcher_topk: int = field(
        default = 0, 
        metadata = {"help": "Number of nodes to prefetch in each layer."}
    )
    
    
    expert_topk: int = field(
        default = 6, 
        metadata = {"help": "Number of experts to active each layer. [Warning] need to align with model."}
    )
    
    num_elements_per_expert: int = field(
        default = 1408*2048, 
        metadata = {"help": "Number of elements in each expert tensor. [Warning] need to align with model."}
    )
    
    num_tensors_per_expert: int = field(
        default = 3, 
        metadata = {"help": "Number of tensors in each experts. [Warning] need to align with model."}
    )
    
    num_expert_layers: int = field(
        default = 26, 
        metadata = {"help": "Number of expert layers in the model. [Warning] need to align with model."}
    )
    
    num_experts: int = field(
        default = 64, 
        metadata = {"help": "Number of experts in each sparse layer. [Warning] need to align with model."}
    )
    
    
    first_k_dense_replace: int = field(
        default = 1,
        metadata = {"help": "Number of dense layers before sparse layers. [Warning] need to align with model."}
    )
    
    LZ4_accelerationLevel: int = field(
        default = 5,
        metadata = {"help": "LZ4 acceleration level."}
    )
    
    LZ4HC_compressionLevel: int = field(
        default = 9,
        metadata = {"help": "LZ4HC compression level."}
    )
    
    ZSTD_compressionLevel: int = field(
        default = 1,
        metadata = {"help": "ZSTD compression level."}
    )
    
    hyperparam_state_margin: float = field(
        default = 0.1,
        metadata = {"help": "Cache state transition margin."}
    )
    
    bind_core: bool = field(
        default = False,
        metadata = {"help": "Restrict each thread into one CPU core."}
    )
    
    @classmethod
    def load_from_file(self, config_path):
        parser = HfArgumentParser(self)
        self = parser.parse_json_file(json_file=config_path)[0]
        return self
    
    @classmethod
    def load_from_json(self, config_json):
        parser = HfArgumentParser(self)
        self = parser.parse_dict(config_json)[0]
        return self
