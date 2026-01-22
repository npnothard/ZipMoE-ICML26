# Copyright (c) 2026 <ZipMoE / Anonymous Team>.
# All rights reserved.
#
# This source code is licensed under the Academic Non-Commercial License.
# See the LICENSE file in the project root for details.



from transformers import (
    PretrainedConfig
)

from models.deepseek import (
    DeepseekV2ForCausalLM
)

from transformers import (
    SwitchTransformersForConditionalGeneration,
    Qwen2MoeForCausalLM
)

MODEL_MAPPING_NAMES = {
    "deepseek": DeepseekV2ForCausalLM,
    "switch": SwitchTransformersForConditionalGeneration,
    "qwen": Qwen2MoeForCausalLM
}

MODEL_MAPPING_TYPES = {
    "deepseek": 0,
    "switch": 1,
    "qwen": 0
}

LZ4DELAY = 2500 
ZSTDDELAY = 3600
DELAY_PROFILE = {
    "LZ4": LZ4DELAY,
    "LZ4HC": LZ4DELAY,
    "ZSTD": ZSTDDELAY
}


LZ4CR = 0.4
ZSTDCR = 0.36
COMPRESSION_RATIO_PROFILE = {
    "LZ4": LZ4CR,
    "LZ4HC": LZ4CR,
    "ZSTD": ZSTDCR    
}

List_expert_topk = {
    "deepseek": 6,
    "switch": 1,
    "qwen": 4
}

List_num_elements_per_expert = {
    "deepseek": 1408*2048,
    "switch": 1024*4096,
    "qwen": 1408*2048
}

List_num_tensors_per_expert = {
    "deepseek": 3,
    "switch": 2,
    "qwen": 3
}

List_num_expert_layers = {
    "deepseek": 26,
    "switch": 24,
    "qwen": 24
}

List_num_experts = {
    "deepseek": 64,
    "switch": 128,
    "qwen": 60
}


List_first_k_dense_replace = {
    "deepseek": 1,
    "switch": 0,
    "qwen": 0
}

def parse_expert_type(config: PretrainedConfig) -> int:
    architecture = config.architectures[0].lower()
    arch = None
    for supp_arch in MODEL_MAPPING_NAMES:
        if supp_arch in architecture:
            arch = supp_arch
            break
    if arch is None:
        raise RuntimeError(
            f"The `load_checkpoint_and_dispatch` function does not support the architecture {architecture}."
            f"Please provide a model that is supported by the function."
            f"Supported architectures are {list(MODEL_MAPPING_NAMES.keys())}"
        )
    return MODEL_MAPPING_TYPES[arch]