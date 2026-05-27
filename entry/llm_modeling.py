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


import os
import warnings
from typing import Any, Dict, Union
import torch
import transformers
from accelerate.utils.versions import is_torch_version
from huggingface_hub import snapshot_download
from transformers import AutoConfig
import models
from utils.constants import MODEL_MAPPING_NAMES
from models import (
    apply_rotary_pos_emb,
    apply_rotary_pos_emb_deepseek
)
from utils import ZipMoEConfig, get_checkpoint_paths
from runtime import ZipMoEEngine

class MoE:
    def __init__(
        self,
        model_name_or_path: Union[str, os.PathLike],
        config: Union[str, os.PathLike, Dict] = None,
    ) -> None:
        if not is_torch_version(">=", "2.0"):
            raise RuntimeError(
                "The `load_checkpoint_and_dispatch` function requires PyTorch >= 2.0. "
                "Please update PyTorch."
            )
        if config is None:
            default_config_path = os.path.join(
                os.path.dirname(__file__), "config.json"
            )
            if not os.path.exists(default_config_path):
                raise RuntimeError(
                    "The `load_checkpoint_and_dispatch` function requires a configuration file. "
                    f"Please provide a configuration file or create a default one at {default_config_path}."
                )
            config = default_config_path
        else:

            model_config = AutoConfig.from_pretrained(
                model_name_or_path, trust_remote_code=True
            )
        architecture = model_config.architectures[0].lower()
        self.model_config = model_config
        print(f"Arch: {architecture}")
        arch = None
        for supp_arch in MODEL_MAPPING_NAMES:
            if supp_arch in architecture:
                arch = supp_arch
                break
        if arch is None:
            raise RuntimeError(
                f"The `load_checkpoint_and_dispatch` function does not support the architecture {architecture}. "
                f"Please provide a model that is supported by the function. "
                f"Supported architectures are {list(MODEL_MAPPING_NAMES.keys())}."
            )
        self.arch = arch
        model_cls = MODEL_MAPPING_NAMES[arch]
        if os.path.exists(model_name_or_path):
            checkpoint_paths = get_checkpoint_paths(model_name_or_path)
        else:
            checkpoint_paths = None
            model_path = snapshot_download(
                model_name_or_path,
                cache_dir=os.environ.get("TRANSFORMERS_CACHE", None),
                ignore_patterns=["flax*", "tf*"],
            )
            if model_path is None:
                raise RuntimeError(
                    f"The `snapshot_download` function could not find the checkpoint {model_name_or_path}. "
                    f"Please provide a valid checkpoint."
                )
            checkpoint_paths = get_checkpoint_paths(model_path)

        if isinstance(config, dict):
            engine_config = ZipMoEConfig.load_from_json(config)
        else:
            engine_config = ZipMoEConfig.load_from_file(config)

        self.engine = ZipMoEEngine(
            model_config
        )
        
        self.engine.ckpt_files = checkpoint_paths
        is_flash_attn_available = False
        try:
            import flash_attn
            is_flash_attn_available = True
            if (
                arch == "switch"
            ):
                is_flash_attn_available = False
        except ImportError:
            print(
                "[WARNING] FlashAttention is not available in the current environment. Using default attention."
            )
        with self.engine.init(model_class=model_cls, engine_config=config):
            self.model = model_cls.from_pretrained(
                model_name_or_path,
                attn_implementation=(
                    "flash_attention_2" if is_flash_attn_available else "eager"
                ),
                is_flash_attn_available=is_flash_attn_available,
                trust_remote_code=True,
            )

    def _configure_hook(self, input_ids: torch.LongTensor):

        if self.arch == "deepseek":
            models.deepseek.modeling_deepseek.apply_rotary_pos_emb = apply_rotary_pos_emb_deepseek

        batch_size = input_ids.shape[0]

    def generate(self, input_ids: torch.LongTensor, **kwargs) -> Any:
        self._configure_hook(input_ids)
        self.model.eval()
        with torch.no_grad():
            return self.model.generate(input_ids, **kwargs)
        self.engine.zipmoe_engine.reset_access_counts()

    def forward(self, input_ids: torch.LongTensor, *args, **kwargs) -> Any:
        self._configure_hook(input_ids)
        return self.model(input_ids, *args, **kwargs)

    def __call__(self, *args, **kwargs) -> Any:

        return self.forward(*args, **kwargs)
