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



import torch

def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    device = position_ids.device
    position_ids = position_ids.to(cos.device)
    cos = cos[position_ids].unsqueeze(unsqueeze_dim).to(q.device)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim).to(q.device)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    position_ids = position_ids.to(device)
    return q_embed, k_embed

def apply_rotary_pos_emb_deepseek(
    q, k, cos, sin, position_ids, unsqueeze_dim=1
):
    device = position_ids.device
    position_ids = position_ids.to(cos.device)
    cos = cos[position_ids].unsqueeze(unsqueeze_dim).to(q.device)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim).to(q.device)

    b, h, s, d = q.shape
    q = q.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)

    b, h, s, d = k.shape
    k = k.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    position_ids = position_ids.to(device)
    return q_embed, k_embed
