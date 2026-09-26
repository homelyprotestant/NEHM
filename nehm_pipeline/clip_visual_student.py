"""
Deployable **single** ``nn.Module``: OpenAI CLIP ``visual`` + student head.

The student is usually ``ImageEmbeddingStudent`` (768→2304 + classifier) or
``ImageToLogitsStudent`` (768→logits only).

Weights are one ``state_dict`` with keys ``visual.*`` and ``student.*``.
Preprocessing (resize / normalize) stays **outside** the module — same as CLIP ``preprocess``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn


class ImageToLogitsStudent(nn.Module):
    """
    Pooled CLIP **image_dim** vector (e.g. 768) → optional dropout → linear **classifier**.

    Same ``forward`` contract as ``ImageEmbeddingStudent``: returns ``(z, logits)`` where
    ``z`` is the pre-classifier embedding (here the same as the input ``image_emb``, not L2
    re-normalized).
    """

    def __init__(
        self,
        image_dim: int,
        num_classes: int,
        *,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        d = float(dropout)
        self.drop = nn.Dropout(d) if d > 0 else nn.Identity()
        self.classifier = nn.Linear(int(image_dim), int(num_classes))

    def set_head_trainable(self, _trainable: bool) -> None:
        """No separate head; API compatibility with ``ImageEmbeddingStudent`` notebooks."""

    def set_classifier_trainable(self, trainable: bool) -> None:
        for p in self.classifier.parameters():
            p.requires_grad = bool(trainable)

    def forward(self, image_emb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        f = image_emb.float()
        logits = self.classifier(self.drop(f))
        return f, logits


class ClipVisualStudent(nn.Module):
    """
    Batched multiview contract (matches training pooling in ``batched_pooled_visual_embeddings``):

    - ``x`` shape ``(B * num_views, C, H, W)`` in CLIP input space (after ``preprocess``).
    - ``forward`` runs ``visual``, L2 per view, mean-pool, L2, then ``student`` → ``(z, logits)``.
    - ``num_views == 1`` → ``x`` is simply ``(B, C, H, W)``.
    """

    def __init__(self, visual: nn.Module, student: nn.Module) -> None:
        super().__init__()
        self.visual = visual
        self.student = student

    @staticmethod
    def pool_multiview_visual_outputs(e: torch.Tensor, num_views: int) -> torch.Tensor:
        """``e``: ``(B * V, D)`` float — output of ``visual`` on stacked crops."""
        bv, d = e.shape
        if num_views < 1 or bv % num_views != 0:
            raise ValueError(f"Expected batch divisible by num_views, got {bv=} {num_views=}")
        b = bv // num_views
        e = e.float().view(b, num_views, d)
        e = e / e.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        pooled = e.mean(dim=1)
        pooled = pooled / pooled.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        return pooled

    def forward_visual_pooled(self, x: torch.Tensor, num_views: int) -> torch.Tensor:
        e = self.visual(x).float()
        return self.pool_multiview_visual_outputs(e, num_views)

    def forward(
        self, x: torch.Tensor, num_views: int = 1
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        emb = self.forward_visual_pooled(x, num_views)
        return self.student(emb)


def load_clip_visual_student_checkpoint_skip_classifier(
    path: Union[str, Path],
    model: ClipVisualStudent,
    *,
    map_location: Any = None,
) -> ClipVisualStudent:
    """
    Load ``visual`` + ``student`` weights when ``student.classifier`` output dim changed.

    Drops ``student.classifier.{weight,bias}`` from the checkpoint if present so the
    in-model classifier (e.g. 170-way ``logitb``) stays freshly initialized.
    """
    ckpt = torch.load(path, map_location=map_location)
    if isinstance(ckpt, nn.Module):
        raise TypeError("Expected a checkpoint dict, not an nn.Module")
    if "state_dict" in ckpt:
        sd = dict(ckpt["state_dict"])
    elif "visual_state_dict" in ckpt and "student_state_dict" in ckpt:
        sd = {}
        for k, v in ckpt["visual_state_dict"].items():
            sd[f"visual.{k}"] = v
        for k, v in ckpt["student_state_dict"].items():
            sd[f"student.{k}"] = v
    else:
        raise ValueError(
            f"Checkpoint must contain 'state_dict' or both visual/student dicts: keys={list(ckpt.keys())}"
        )
    for key in list(sd.keys()):
        if key.startswith("student.classifier."):
            del sd[key]
    missing, unexpected = model.load_state_dict(sd, strict=False)
    miss = [k for k in missing if not k.startswith("student.classifier.")]
    if miss or unexpected:
        raise RuntimeError(f"load_state_dict mismatch: missing={miss!r} unexpected={unexpected!r}")
    return model


def load_clip_visual_student_checkpoint(
    path: Union[str, Path],
    model: ClipVisualStudent,
    *,
    map_location: Any = None,
    strict: bool = True,
) -> ClipVisualStudent:
    """
    Load **unified** ``{"state_dict": ...}`` or **legacy** ``visual_state_dict`` +
    ``student_state_dict`` checkpoints into ``model``.
    """
    ckpt = torch.load(path, map_location=map_location)
    if isinstance(ckpt, nn.Module):
        raise TypeError("Expected a checkpoint dict, not an nn.Module")
    if "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"], strict=strict)
    elif "visual_state_dict" in ckpt and "student_state_dict" in ckpt:
        model.visual.load_state_dict(ckpt["visual_state_dict"], strict=strict)
        model.student.load_state_dict(ckpt["student_state_dict"], strict=strict)
    else:
        raise ValueError(
            f"Checkpoint must contain 'state_dict' or both visual/student dicts: keys={list(ckpt.keys())}"
        )
    return model


def load_visual_branch_from_unified_checkpoint(
    path: Union[str, Path],
    model: ClipVisualStudent,
    *,
    map_location: Any = None,
) -> ClipVisualStudent:
    """
    Load only ``visual.*`` weights from a unified ViT fine-tune checkpoint.

    Use when the student was replaced (e.g. ``ImageToLogitsStudent``) and old ``student.*``
    keys must not be loaded.
    """
    ckpt = torch.load(path, map_location=map_location)
    if isinstance(ckpt, nn.Module):
        raise TypeError("Expected a checkpoint dict, not an nn.Module")
    if "state_dict" in ckpt:
        sd = ckpt["state_dict"]
    elif "visual_state_dict" in ckpt:
        sd = {f"visual.{k}": v for k, v in ckpt["visual_state_dict"].items()}
    else:
        raise ValueError(
            f"Checkpoint must contain 'state_dict' or 'visual_state_dict'; keys={list(ckpt.keys())}"
        )
    if not isinstance(sd, dict):
        raise TypeError("state_dict must be a dict")
    vis = {k[len("visual.") :]: v for k, v in sd.items() if isinstance(k, str) and k.startswith("visual.")}
    if not vis:
        raise ValueError(f"No visual.* tensors in checkpoint: {path!r}")
    model.visual.load_state_dict(vis, strict=True)
    return model


def unified_finetune_checkpoint_dict(
    model: ClipVisualStudent,
    *,
    meta: Dict[str, Any],
    stage: Optional[str] = None,
    epoch_in_stage: Optional[int] = None,
    val_loss: Optional[float] = None,
    val_acc: Optional[float] = None,
) -> Dict[str, Any]:
    """Payload for ``torch.save`` (deployment + provenance)."""
    out: Dict[str, Any] = {
        "state_dict": model.state_dict(),
        "meta": dict(meta),
    }
    if stage is not None:
        out["stage"] = stage
    if epoch_in_stage is not None:
        out["epoch_in_stage"] = epoch_in_stage
    if val_loss is not None:
        out["val_loss"] = val_loss
    if val_acc is not None:
        out["val_acc"] = val_acc
    return out
