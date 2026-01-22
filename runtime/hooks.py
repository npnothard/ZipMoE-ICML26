# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0
# EfficientMoE Team
# 
# This file is derived from the EfficientMoE project and is
# redistributed in compliance with the Apache License, Version 2.0.
#
# See the NOTICE file in the project root for attribution details.



import functools
from typing import Callable
import torch

from models import (
    apply_rotary_pos_emb,
    apply_rotary_pos_emb_deepseek
)

def do_nothing_decorator( original_func: Callable ):
    @functools.wraps( original_func )
    def do_nothing( *args, **kwargs ):
        pass
    return do_nothing


def empty_param_init_decorator(orig_param_init: Callable) -> Callable:
    @functools.wraps(orig_param_init)
    def empty_param_init(cls, *args, **kwargs):
        orig_param_init(cls, *args, **kwargs)
        for name, param in cls.named_parameters(recurse=False):
            param.data = torch.zeros(1, dtype=param.dtype, device=param.device)
        for name, buf in cls.named_buffers(recurse=False):
            buf.data = torch.zeros(1, dtype=buf.dtype, device=buf.device)
    return empty_param_init


def activate_empty_init():
    for name, module in torch.nn.modules.__dict__.items():
        if not isinstance(module, type):
            continue
        if not issubclass(module, torch.nn.modules.module.Module):
            continue
        if name in [
            "Module",
            "Sequential",
            "ModuleDict",
            "ModuleList",
            "ParameterList",
            "ParameterDict",
        ]:
            continue
        module._old_init = module.__init__
        module.__init__ = empty_param_init_decorator(module.__init__)

        if hasattr(module, "reset_parameters"):
            module._old_reset_parameters = module.reset_parameters
            module.reset_parameters = do_nothing_decorator(
                module.reset_parameters
            )

def deactivate_empty_init():
    for name, module in torch.nn.modules.__dict__.items():
        if not isinstance(module, type):
            continue
        if not issubclass(module, torch.nn.modules.module.Module):
            continue
        if name in [
            "Module",
            "Sequential",
            "ModuleDict",
            "ModuleList",
            "ParameterList",
            "ParameterDict",
        ]:
            continue
        if hasattr(module, "_old_init"):
            module.__init__ = module._old_init
            del module._old_init

        if hasattr(module, "_old_reset_parameters"):
            module.reset_parameters = module._old_reset_parameters
            del module._old_reset_parameters
