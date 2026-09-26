"""
ViT visual trunk through ``ln_post`` (CLS), **without** the projection head.

Kept in a tiny module so ``clip_vit_multiview_finetune`` / ``student_clip`` do not
depend on the full LongCLIP embedding pipeline being present on the import path.
"""

from __future__ import annotations

import torch


def vit_visual_ln_post_forward(visual: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """
    ViT trunk only: patch embed → CLS + patches → ``ln_pre`` → transformer → CLS → ``ln_post``.
    Returns ``(N, width)`` with **no** ``@ proj`` (768-D on ViT-B/16).
    """
    x = x.type(visual.conv1.weight.dtype)
    x = visual.conv1(x)
    x = x.reshape(x.shape[0], x.shape[1], -1)
    x = x.permute(0, 2, 1)
    cls_broadcast = visual.class_embedding.to(x.dtype) + torch.zeros(
        x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
    )
    x = torch.cat([cls_broadcast, x], dim=1)
    x = x + visual.positional_embedding.to(x.dtype)
    x = visual.ln_pre(x)
    x = x.permute(1, 0, 2)
    x = visual.transformer(x)
    x = x.permute(1, 0, 2)
    return visual.ln_post(x[:, 0, :])
