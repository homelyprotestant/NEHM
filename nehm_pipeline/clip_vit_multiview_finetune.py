"""
End-to-end fine-tuning helpers: **ViT-L/14@336px** ``visual`` + ``ImageEmbeddingStudent`` head.

Multiview policy (training / eval contract you asked for):

1. **Height-336 pipeline:** ``notebook_single_crop_336`` → one **center** 336×336 crop.
2. **Native resolution:** ``random_square_crops_native_resolution`` → ``n_native`` 336×336 crops; RNG matches
   ``embed.encode_dataset`` when ``LabeledMultiviewFinetuneDataset(window_sample_seed=…)`` is set (default ``42``).
3. Encode each crop with CLIP ``visual`` (after ``preprocess``: same geometry as CLIP, **linear RGB
   [0, 1]** by default — no per-channel mean/std; see ``device.load_openai_clip``), **L2-normalize**
   per crop, **mean-pool**, **L2-normalize** the pooled vector — same convention as
   ``student_clip.pooled_clip_visual_embedding``.

Use **only rows with a non-negative material column** (e.g. ``logit`` or ``logitb``); loss is typically CE on material classes.
"""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from PIL import ImageEnhance
from torch.utils.data import DataLoader, Dataset, get_worker_info

from nehm_pipeline.vit_ln_post_utils import vit_visual_ln_post_forward
from nehm_pipeline.preprocess_image import (
    notebook_single_crop_336,
    pil_train_geom_augment,
    random_square_crops_adaptive_native_resolution,
    random_square_crops_native_resolution,
    short_side_resize_center_crop,
)


def _cuda_amp_autocast(
    device: torch.device,
    enabled: bool,
    dtype: Optional[torch.dtype] = None,
):
    """CUDA autocast; no-op on non-CUDA or when ``enabled`` is False (PyTorch 1.x + 2.x).

    ``dtype`` selects the autocast dtype (``torch.float16`` or ``torch.bfloat16``).
    Defaults to ``torch.float16`` to preserve historical behavior.
    """
    if not enabled or device.type != "cuda":
        return contextlib.nullcontext()
    amp_dtype = dtype or torch.float16
    try:
        return torch.amp.autocast("cuda", enabled=True, dtype=amp_dtype)
    except (AttributeError, TypeError):
        try:
            return torch.cuda.amp.autocast(enabled=True, dtype=amp_dtype)
        except TypeError:
            return torch.cuda.amp.autocast(enabled=True)


def make_cuda_grad_scaler():
    """``torch.amp.GradScaler('cuda')`` on PT 2+, else ``torch.cuda.amp.GradScaler``."""
    try:
        return torch.amp.GradScaler("cuda")
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler()


def multiview_crops_finetune(
    pil: Image.Image,
    n_native_random: int,
    rng: np.random.Generator,
    target: int = 336,
    *,
    tile_px: Optional[int] = None,
    global_kind: str = "height",
    pre_crop_margin_px: int = 0,
    tile_min_px: int = 16,
    tile_max_px: int = 224,
    tile_scale_ref_px: int = 3000,
) -> List[Image.Image]:
    """
    Returns ``1 + n_native_random`` crops:

    - **One global view** at ``target``×``target``. ``global_kind`` controls how the global
      view is built before the center-crop:

      - ``"height"`` *(default, scheme A)*: ``notebook_single_crop_336`` — resize so **height**
        equals ``target`` (preserve aspect), then center-crop ``target``×``target`` (with a
        narrow-width fallback that upsamples width to ``target``).
      - ``"short_side"`` *(scheme B)*: ``short_side_resize_center_crop`` — resize so the
        **short** side equals ``target`` (preserve aspect), then center-crop ``target``×``target``.
        Symmetric in width vs height; matches the global branch of the LongCLIP teacher
        (``LongCLIP_Embeddings_v1.run_longclip_embeddings``).

    - **N native-resolution random crops** sized ``crop_px``×``crop_px``:

      - ``tile_px=None`` *(default, scheme A)*: ``crop_px = target`` — each random crop is the
        full ``target``×``target`` (e.g. 336×336 native-resolution patches).
      - ``tile_px=<int>`` *(scheme B v1)*: small native-resolution **tiles** at ``tile_px``×``tile_px``.
      - ``tile_px=0`` *(scheme B v3.1, scale-adaptive)*: per-image tile side computed by
        :func:`compute_adaptive_tile_px` from ``tile_min_px``, ``tile_max_px``,
        ``tile_scale_ref_px`` (defaults match the teacher: 16, 224, 3000). Larger
        images get larger tiles (saturating at ``tile_max_px``); smaller images
        get smaller tiles (saturating at ``tile_min_px``).
      - ``n_native_random=0`` *(scheme B v2)*: no random tiles; only the single global view is returned.

    ``pre_crop_margin_px`` (default 0): if > 0, drop that many pixels from each side of ``pil``
    **before** any further processing (so both the global view and any random tiles see the
    trimmed image). Use 1 for **B v2 / B v3** to ignore the typical 1-pixel scan-border artifact
    on Pigment-Compendium-style book scans. The default of 0 preserves Pipeline A and B v1 behavior.

    A and B share the same dataset/loader/runner; only this function differs.
    """
    pil = pil.convert("RGB")
    m = max(int(pre_crop_margin_px), 0)
    if m > 0:
        w, h = pil.size
        if w > 2 * m and h > 2 * m:
            pil = pil.crop((m, m, w - m, h - m))
    if global_kind == "short_side":
        center = short_side_resize_center_crop(pil, target)
    elif global_kind == "height":
        center = notebook_single_crop_336(pil, target)
    else:
        raise ValueError(f"global_kind must be 'height' or 'short_side', got {global_kind!r}")
    if tile_px is None:
        crop_px = int(target)
        native = random_square_crops_native_resolution(pil, crop_px, n_native_random, rng)
    elif int(tile_px) <= 0:
        native, _tile_used = random_square_crops_adaptive_native_resolution(
            pil,
            n_native_random,
            rng,
            tile_min_px=tile_min_px,
            tile_max_px=tile_max_px,
            tile_scale_ref_px=tile_scale_ref_px,
        )
    else:
        crop_px = int(tile_px)
        native = random_square_crops_native_resolution(pil, crop_px, n_native_random, rng)
    return [center] + native


def batched_pooled_visual_embeddings(
    visual: nn.Module,
    preprocess: Callable[[Image.Image], torch.Tensor],
    batch_crops: Sequence[Sequence[Image.Image]],
    device: torch.device,
    *,
    encode_chunk_size: Optional[int] = None,
) -> torch.Tensor:
    """
    ``batch_crops[i]`` is a list of PIL crops for sample ``i`` (same length ``V`` for all ``i``).

    Returns ``(B, D)`` float32 pooled image embeddings (L2-normalized mean of L2-normalized crops).

    ``encode_chunk_size``: max images per ``visual`` forward. **MPS:** if ``None`` or ``<= 0``,
    defaults to ``4`` whenever ``B×V > 4`` (training ViT-L/14@336 often OOMs without this).
    **CUDA/CPU:** ``None`` means one forward over all crops. Set a positive int to override.
    """
    if not batch_crops:
        raise ValueError("empty batch")
    v0 = len(batch_crops[0])
    if v0 == 0:
        raise ValueError("zero crops per sample")
    for crops in batch_crops:
        if len(crops) != v0:
            raise ValueError("variable multiview count in batch")
    b = len(batch_crops)
    clip_dtype = next(visual.parameters()).dtype
    flat: List[Image.Image] = []
    for crops in batch_crops:
        flat.extend(crops)
    n_flat = len(flat)
    chunk = int(encode_chunk_size) if encode_chunk_size is not None else 0
    if chunk <= 0 and device.type == "mps":
        chunk = 4
    # One forward only when everything fits in a single chunk (or non-MPS with chunk==0).
    if chunk <= 0 or n_flat <= chunk:
        tensor = torch.stack([preprocess(p) for p in flat]).to(device=device, dtype=clip_dtype)
        e = visual(tensor).float()
    else:
        parts: List[torch.Tensor] = []
        for start in range(0, n_flat, chunk):
            sub = flat[start : start + chunk]
            t = torch.stack([preprocess(p) for p in sub]).to(device=device, dtype=clip_dtype)
            parts.append(visual(t).float())
            if device.type == "mps":
                try:
                    torch.mps.empty_cache()
                except Exception:
                    pass
        e = torch.cat(parts, dim=0)
    d = e.shape[-1]
    e = e.view(b, v0, d)
    e = e / e.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    pooled = e.mean(dim=1)
    pooled = pooled / pooled.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    return pooled


def batched_pooled_visual_ln_post_embeddings(
    visual: nn.Module,
    preprocess: Callable[[Image.Image], torch.Tensor],
    batch_crops: Sequence[Sequence[Image.Image]],
    device: torch.device,
    *,
    encode_chunk_size: Optional[int] = None,
) -> torch.Tensor:
    """
    Same pooling policy as :func:`batched_pooled_visual_embeddings`, but each crop is encoded with
    **ViT ``ln_post``(CLS) only** (768-D on ViT-B/16) — no ``visual.proj``.

    ``visual`` must be a LongCLIP/OpenAI-style ``VisionTransformer`` with ``conv1``, ``ln_pre``,
    ``transformer``, ``ln_post``.
    """
    if not batch_crops:
        raise ValueError("empty batch")
    v0 = len(batch_crops[0])
    if v0 == 0:
        raise ValueError("zero crops per sample")
    for crops in batch_crops:
        if len(crops) != v0:
            raise ValueError("variable multiview count in batch")
    b = len(batch_crops)
    clip_dtype = next(visual.parameters()).dtype
    flat: List[Image.Image] = []
    for crops in batch_crops:
        flat.extend(crops)
    n_flat = len(flat)
    chunk = int(encode_chunk_size) if encode_chunk_size is not None else 0
    if chunk <= 0 and device.type == "mps":
        chunk = 4
    if chunk <= 0 or n_flat <= chunk:
        tensor = torch.stack([preprocess(p) for p in flat]).to(device=device, dtype=clip_dtype)
        e = vit_visual_ln_post_forward(visual, tensor).float()
    else:
        parts: List[torch.Tensor] = []
        for start in range(0, n_flat, chunk):
            sub = flat[start : start + chunk]
            t = torch.stack([preprocess(p) for p in sub]).to(device=device, dtype=clip_dtype)
            parts.append(vit_visual_ln_post_forward(visual, t).float())
            if device.type == "mps":
                try:
                    torch.mps.empty_cache()
                except Exception:
                    pass
        e = torch.cat(parts, dim=0)
    d = e.shape[-1]
    e = e.view(b, v0, d)
    e = e / e.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    pooled = e.mean(dim=1)
    pooled = pooled / pooled.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    return pooled


def set_module_requires_grad(module: nn.Module, requires_grad: bool) -> None:
    for p in module.parameters():
        p.requires_grad = requires_grad


def set_module_or_param_requires_grad(
    x: Union[nn.Module, nn.Parameter], requires_grad: bool
) -> None:
    """OpenAI CLIP ViT uses ``visual.proj`` as ``nn.Parameter``; some forks use ``nn.Linear``."""
    if isinstance(x, nn.Parameter):
        x.requires_grad = requires_grad
    else:
        set_module_requires_grad(x, requires_grad)


def clip_visual_num_resblocks(visual: nn.Module) -> int:
    return len(visual.transformer.resblocks)


def freeze_clip_visual(visual: nn.Module) -> None:
    set_module_requires_grad(visual, False)


def unfreeze_clip_visual_ln_post_proj_only(visual: nn.Module) -> None:
    """Train only ``ln_post`` + ``proj`` (all transformer blocks and stem frozen)."""
    freeze_clip_visual(visual)
    set_module_requires_grad(visual.ln_post, True)
    if hasattr(visual, "proj") and visual.proj is not None:
        set_module_or_param_requires_grad(visual.proj, True)


def unfreeze_clip_visual_last_n_blocks(
    visual: nn.Module,
    n_blocks: int,
    *,
    include_stem: bool = False,
    include_ln_post_proj: bool = True,
) -> None:
    """
    Freeze everything, then enable gradients on the **last** ``n_blocks`` ``ResidualAttentionBlock``s.

    - ``include_ln_post_proj``: also train ``ln_post`` and ``proj`` (recommended whenever any block trains).
    - ``include_stem``: if True, also train ``conv1``, ``ln_pre``, and positional/class embeddings (late stage).
    """
    freeze_clip_visual(visual)
    blocks = visual.transformer.resblocks
    total = len(blocks)
    n = max(0, min(int(n_blocks), total))
    for i in range(total - n, total):
        set_module_requires_grad(blocks[i], True)
    if include_ln_post_proj:
        set_module_requires_grad(visual.ln_post, True)
        if hasattr(visual, "proj") and visual.proj is not None:
            set_module_or_param_requires_grad(visual.proj, True)
    if include_stem:
        set_module_requires_grad(visual.conv1, True)
        set_module_requires_grad(visual.ln_pre, True)
        if hasattr(visual, "positional_embedding") and visual.positional_embedding is not None:
            visual.positional_embedding.requires_grad = True
        if hasattr(visual, "class_embedding") and visual.class_embedding is not None:
            visual.class_embedding.requires_grad = True


def unfreeze_clip_visual_full(visual: nn.Module) -> None:
    set_module_requires_grad(visual, True)


class LabeledMultiviewFinetuneDataset(Dataset):
    """
    Rows = global manifest indices with non-negative material targets (e.g. ``logit`` / ``logitb``).

    **Native random crops** match ``nehm_pipeline.embed.encode_dataset`` for
    ``pooled_random_hr_crops_336``:

    - If ``window_sample_seed`` is an ``int``, uses
      ``np.random.default_rng(window_sample_seed + i * 1_000_003)`` where ``i`` is the
      **global** manifest row index (same as ``idx`` in the embed loop). The same image
      always gets the **same** 16 windows as in ``image_embeddings.npy`` when the pipeline
      was run with the same ``PipelineConfig.window_sample_seed``.
    - If ``window_sample_seed`` is ``None``, matches embed's unseeded mode: **new** random
      native crops on every ``__getitem__`` (stochastic training).

    If ``fused_teacher`` is set (``N, D`` aligned with manifest), returns
    ``(crops, teacher_row, y)`` with ``teacher_row = fused_teacher[i]`` for joint distillation.
    Otherwise returns ``(crops, y)``.
    """

    def __init__(
        self,
        row_indices: np.ndarray,
        material_logits: np.ndarray,
        filenames: Sequence[str],
        images_dir: Path,
        n_native_random: int = 16,
        crop_size: int = 336,
        fused_teacher: np.ndarray | None = None,
        window_sample_seed: Optional[int] = 42,
        geom_augment: bool = False,
        augment_max_rotation_deg: float = 5.0,
        augment_hflip_p: float = 0.5,
        augment_vflip_p: float = 0.5,
        augment_jitter_brightness: float = 0.0,
        augment_jitter_contrast: float = 0.0,
        augment_jitter_saturation: float = 0.0,
        *,
        tile_px: Optional[int] = None,
        global_kind: str = "height",
        pre_crop_margin_px: int = 0,
        tile_min_px: int = 16,
        tile_max_px: int = 224,
        tile_scale_ref_px: int = 3000,
    ) -> None:
        """
        ``tile_px`` and ``global_kind`` switch between scheme **A** (defaults: ``tile_px=None``,
        ``global_kind='height'``; 16 random ``crop_size``×``crop_size`` native crops + 1 height-resized
        global) and scheme **B v1** (``tile_px=16``, ``global_kind='short_side'``, ``n_native_random=256``;
        256 native-resolution 16×16 tiles + 1 short-side-resized global). For **B v2** pass
        ``n_native_random=0`` and the dataset returns just the single global view per row.

        For **B v3.1 (scale-adaptive multiview)** pass ``tile_px=0``,
        ``global_kind='short_side'``, ``n_native_random=16``, ``crop_size=224``,
        ``pre_crop_margin_px=1``, plus ``tile_min_px=16``, ``tile_max_px=224``,
        ``tile_scale_ref_px=3000``. The per-image tile side is then chosen as
        ``clip(round(tile_max_px * sqrt(W*H) / tile_scale_ref_px), tile_min_px, tile_max_px)``
        so that the tile fraction-of-FOV is roughly constant across sources.

        ``pre_crop_margin_px`` (default 0): if > 0, drop that many pixels from each side of the raw
        image before any other processing. Use 1 for B v2 / B v3 to ignore the typical 1-pixel
        scan-border artifact on Pigment Compendium-style book scans. See
        :func:`multiview_crops_finetune`.
        """
        self.row_indices = np.asarray(row_indices, dtype=np.int64)
        self.material_logits = np.asarray(material_logits, dtype=np.int64).ravel()
        self.filenames = list(filenames)
        self.images_dir = Path(images_dir)
        self.n_native_random = int(n_native_random)
        self.crop_size = int(crop_size)
        self.tile_px = None if tile_px is None else int(tile_px)
        self.global_kind = str(global_kind)
        self.pre_crop_margin_px = int(pre_crop_margin_px)
        self.tile_min_px = int(tile_min_px)
        self.tile_max_px = int(tile_max_px)
        self.tile_scale_ref_px = int(tile_scale_ref_px)
        self.window_sample_seed = window_sample_seed
        self.geom_augment = bool(geom_augment)
        self.augment_max_rotation_deg = float(augment_max_rotation_deg)
        self.augment_hflip_p = float(augment_hflip_p)
        self.augment_vflip_p = float(augment_vflip_p)
        self.augment_jitter_brightness = float(augment_jitter_brightness)
        self.augment_jitter_contrast = float(augment_jitter_contrast)
        self.augment_jitter_saturation = float(augment_jitter_saturation)
        self.fused_teacher = (
            None if fused_teacher is None else np.asarray(fused_teacher, dtype=np.float32)
        )

    def __len__(self) -> int:
        return len(self.row_indices)

    def __getitem__(self, idx: int):
        i = int(self.row_indices[idx])
        path = self.images_dir / self.filenames[i]
        pil = Image.open(path).convert("RGB")
        if self.window_sample_seed is not None:
            rng = np.random.default_rng(
                int(self.window_sample_seed) + i * 1_000_003
            )
        else:
            wi = get_worker_info()
            if wi is not None:
                rng = np.random.default_rng(
                    int(wi.seed)
                    ^ (idx * 0x9E3779B9)
                    ^ int(torch.randint(0, 2**31, (1,)).item())
                )
            else:
                rng = np.random.default_rng()
        if self.geom_augment:
            aug_rng = np.random.default_rng()
            pil = pil_train_geom_augment(
                pil,
                aug_rng,
                max_rotation_deg=self.augment_max_rotation_deg,
                hflip_p=self.augment_hflip_p,
                vflip_p=self.augment_vflip_p,
            )
            # Small photometric jitter for robustness; values are symmetric around 1.0.
            if self.augment_jitter_brightness > 0:
                f = float(
                    aug_rng.uniform(
                        max(0.0, 1.0 - self.augment_jitter_brightness),
                        1.0 + self.augment_jitter_brightness,
                    )
                )
                pil = ImageEnhance.Brightness(pil).enhance(f)
            if self.augment_jitter_contrast > 0:
                f = float(
                    aug_rng.uniform(
                        max(0.0, 1.0 - self.augment_jitter_contrast),
                        1.0 + self.augment_jitter_contrast,
                    )
                )
                pil = ImageEnhance.Contrast(pil).enhance(f)
            if self.augment_jitter_saturation > 0:
                f = float(
                    aug_rng.uniform(
                        max(0.0, 1.0 - self.augment_jitter_saturation),
                        1.0 + self.augment_jitter_saturation,
                    )
                )
                pil = ImageEnhance.Color(pil).enhance(f)
        crops = multiview_crops_finetune(
            pil,
            self.n_native_random,
            rng,
            target=self.crop_size,
            tile_px=self.tile_px,
            global_kind=self.global_kind,
            pre_crop_margin_px=self.pre_crop_margin_px,
            tile_min_px=self.tile_min_px,
            tile_max_px=self.tile_max_px,
            tile_scale_ref_px=self.tile_scale_ref_px,
        )
        y = int(self.material_logits[i])
        if self.fused_teacher is not None:
            t = np.asarray(self.fused_teacher[i], dtype=np.float32).copy()
            return crops, t, y
        return crops, y


def vit_finetune_unfreeze_schedule_doc() -> str:
    """Human-readable default schedule for ViT-L/14 (24 blocks)."""
    return """
Recommended **unfreezing sequence** (ViT-L/14 = **24** ``resblocks``; tune epochs to your val curve).

**Stage A — readout + student**
Call ``unfreeze_clip_visual_ln_post_proj_only(visual)``.
Train ``ln_post``, ``proj``, and the full ``ImageEmbeddingStudent``. Freeze stem + all 24 blocks.
LR: head ~1e-4–3e-4; ``ln_post``/``proj`` ~3e-5–1e-4. Epochs: **2–5**.

**Stage B — top blocks**
``unfreeze_clip_visual_last_n_blocks(visual, 2)`` (last two blocks + ln_post + proj).
Block LR ~5e-6–2e-5. Epochs: **5–15**.

**Stage C — deepen**
``n`` = 4 → 8 → 12 in steps; multiply backbone LR by ~0.5–0.7 each expansion. Epochs: **10–30** total.

**Stage D — full ViT**
``unfreeze_clip_visual_full(visual)``; use **param groups** with layer decay (early blocks 0.1×–0.3× of last-block LR).
Backbone ~1e-6–5e-6, head ~1e-5–1e-4 (AdamW). Optional final short sub-stage with ``include_stem=True`` only if you must adapt the patch stem.

**Practical**
Weight decay on ViT; grad clip ~1.0 when unfreezing blocks. **17 views** → small **batch size** (1–2 on MPS) or gradient accumulation.
"""


def collect_ln_post_proj_params(
    visual: nn.Module, *, trainable_only: bool = True
) -> List[nn.Parameter]:
    """Parameters for ``ln_post`` and ``proj`` (``nn.Parameter`` or submodule)."""
    out: List[nn.Parameter] = []
    for p in visual.ln_post.parameters():
        if not trainable_only or p.requires_grad:
            out.append(p)
    proj = getattr(visual, "proj", None)
    if proj is None:
        return out
    if isinstance(proj, nn.Parameter):
        if not trainable_only or proj.requires_grad:
            out.append(proj)
    else:
        for p in proj.parameters():
            if not trainable_only or p.requires_grad:
                out.append(p)
    return out


def collect_stem_params(visual: nn.Module, *, trainable_only: bool = True) -> List[nn.Parameter]:
    """Patch stem + positional / class embeddings (when present)."""
    out: List[nn.Parameter] = []
    for name in ("conv1", "ln_pre"):
        m = getattr(visual, name, None)
        if m is not None:
            for p in m.parameters():
                if not trainable_only or p.requires_grad:
                    out.append(p)
    for aname in ("positional_embedding", "class_embedding"):
        t = getattr(visual, aname, None)
        if isinstance(t, nn.Parameter) and (not trainable_only or t.requires_grad):
            out.append(t)
    return out


def unfreeze_clip_visual_stem_only(visual: nn.Module) -> None:
    """
    Train only the ViT **stem** (``conv1``, ``ln_pre``, positional + class embeddings).
    All ``transformer.resblocks``, ``ln_post``, and ``proj`` stay frozen.
    """
    freeze_clip_visual(visual)
    for p in collect_stem_params(visual, trainable_only=False):
        p.requires_grad = True


def build_adamw_vit_finetune(
    visual: nn.Module,
    student: nn.Module,
    *,
    lr_student: float,
    lr_readout: float,
    lr_backbone: Optional[float],
    weight_decay: float,
    full_visual_layer_decay: bool = False,
    layer_decay: float = 0.75,
    device: Optional[torch.device] = None,
) -> torch.optim.AdamW:
    """
    Build param groups after you have set ``requires_grad`` on ``visual`` (via unfreeze helpers).

    - **Partial backbone:** one LR for all trainable resblocks; ``lr_backbone`` required if any block trains.
    - **Full visual:** set ``full_visual_layer_decay=True``; last resblock uses ``lr_backbone``, earlier
      blocks use ``lr_backbone * layer_decay ** (depth from last)``; stem gets one more decay step.
    """
    groups: List[Dict] = []
    blocks = visual.transformer.resblocks
    n = len(blocks)

    if full_visual_layer_decay:
        if lr_backbone is None:
            raise ValueError("lr_backbone required when full_visual_layer_decay=True")
        ld = float(layer_decay)
        for i in range(n):
            pms = [p for p in blocks[i].parameters() if p.requires_grad]
            if pms:
                lr_i = lr_backbone * (ld ** (n - 1 - i))
                groups.append({"params": pms, "lr": lr_i, "weight_decay": weight_decay})
        stem = collect_stem_params(visual, trainable_only=True)
        if stem:
            lr_stem = lr_backbone * (ld**n)
            groups.append({"params": stem, "lr": lr_stem, "weight_decay": weight_decay})
    else:
        backbone: List[nn.Parameter] = []
        for blk in blocks:
            for p in blk.parameters():
                if p.requires_grad:
                    backbone.append(p)
        if backbone:
            if lr_backbone is None:
                raise ValueError("lr_backbone required when transformer blocks are trainable")
            groups.append(
                {"params": backbone, "lr": lr_backbone, "weight_decay": weight_decay}
            )

    readout = collect_ln_post_proj_params(visual, trainable_only=True)
    if readout:
        groups.append({"params": readout, "lr": lr_readout, "weight_decay": weight_decay})

    st = [p for p in student.parameters() if p.requires_grad]
    groups.append({"params": st, "lr": lr_student, "weight_decay": weight_decay})

    if device is not None and device.type == "cuda":
        try:
            return torch.optim.AdamW(groups, fused=True)
        except TypeError:
            pass
    return torch.optim.AdamW(groups)


def vit_finetune_run_epoch(
    visual: nn.Module,
    student: nn.Module,
    preprocess: Callable[[Image.Image], torch.Tensor],
    loader: DataLoader,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
    *,
    train: bool,
    lambda_emb: float,
    lambda_ce: float,
    emb_kind: str,
    label_smoothing: float,
    grad_clip: float = 1.0,
    use_tqdm: bool = False,
    tqdm_desc: Optional[str] = None,
    log_every_batches: int = 0,
    log_prefix: str = "",
    student_frozen_eval: bool = False,
    ce_class_weights: Optional[torch.Tensor] = None,
    visual_encode_chunk_size: Optional[int] = None,
    visual_ln_post_only: bool = False,
    use_amp: bool = False,
    amp_dtype: Optional[torch.dtype] = None,
    grad_scaler: Optional[Any] = None,
) -> Tuple[float, float, float, float]:
    """One pass over ``loader``; joint loss matches ``student_clip.joint_material_loss``.

    If ``student_frozen_eval`` is True, ``student.eval()`` is forced even on training batches
    (no dropout on a frozen student head).

    ``ce_class_weights``: optional ``(num_classes,)`` tensor — applied to CE **only when**
    ``train=True`` (validation should stay unweighted for a natural distribution).

    ``visual_encode_chunk_size``: passed to pooled visual encoders (MPS OOM fix).

    ``visual_ln_post_only``: if True, pool **768-D** ``ln_post``(CLS) vectors (no ``visual.proj``);
    training must use a student head with matching ``image_dim``.

    ``use_amp``: CUDA mixed precision (forward + loss in autocast). Validation uses autocast only.

    ``amp_dtype``: ``torch.float16`` (default) or ``torch.bfloat16``. fp16 training requires a
    ``grad_scaler`` from :func:`make_cuda_grad_scaler`. bf16 training **must not** use a
    GradScaler (pass ``grad_scaler=None``); bf16's larger exponent range makes loss scaling
    unnecessary.

    The fourth return value is **classification accuracy on labeled rows only** (targets ``!= -1``,
    i.e. material logits **≥ 0** when ``-1`` marks missing labels).

    Metrics print only **after** the full epoch unless ``log_every_batches > 0`` (running averages).
    """
    from nehm_pipeline.student_clip import joint_material_loss

    ign = -1

    if train:
        student.train()
        if any(p.requires_grad for p in visual.parameters()):
            visual.train()
        else:
            visual.eval()
    else:
        student.eval()
        visual.eval()
    if student_frozen_eval:
        student.eval()
    tot = tot_emb = tot_ce = n = corr = n_labeled_acc = 0
    params = [p for p in list(visual.parameters()) + list(student.parameters()) if p.requires_grad]
    amp_here = bool(use_amp and device.type == "cuda")
    amp_dtype_eff = amp_dtype or torch.float16
    if (
        train
        and amp_here
        and optimizer is not None
        and grad_scaler is None
        and amp_dtype_eff is torch.float16
    ):
        raise ValueError(
            "CUDA fp16 AMP training requires grad_scaler (see make_cuda_grad_scaler()); "
            "pass None only for eval-only epochs or when amp_dtype=torch.bfloat16."
        )
    if amp_here and grad_scaler is not None and amp_dtype_eff is torch.bfloat16:
        raise ValueError(
            "bf16 AMP must not use a GradScaler; pass grad_scaler=None when amp_dtype=torch.bfloat16."
        )
    ctx = torch.enable_grad() if train else torch.inference_mode()
    n_batches = len(loader)
    iterator: Union[DataLoader, object] = loader
    if use_tqdm:
        from tqdm.auto import tqdm

        iterator = tqdm(
            loader,
            leave=False,
            total=n_batches,
            desc=tqdm_desc or (log_prefix.strip() or None),
        )
    with ctx:
        for i_batch, (crops_b, t, y) in enumerate(iterator, start=1):
            y = y.to(device)
            t = t.to(device, dtype=torch.float32)
            if train and optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            with _cuda_amp_autocast(device, amp_here, amp_dtype_eff):
                if visual_ln_post_only:
                    emb = batched_pooled_visual_ln_post_embeddings(
                        visual,
                        preprocess,
                        crops_b,
                        device,
                        encode_chunk_size=visual_encode_chunk_size,
                    )
                else:
                    emb = batched_pooled_visual_embeddings(
                        visual,
                        preprocess,
                        crops_b,
                        device,
                        encode_chunk_size=visual_encode_chunk_size,
                    )
                z, logits = student(emb)
                loss, l_emb, l_ce = joint_material_loss(
                    z,
                    t,
                    logits,
                    y,
                    lambda_emb=lambda_emb,
                    lambda_ce=lambda_ce,
                    emb_kind=emb_kind,
                    label_smoothing=label_smoothing,
                    ce_class_weights=ce_class_weights if train else None,
                )
            if train and optimizer is not None:
                if amp_here and grad_scaler is not None:
                    grad_scaler.scale(loss).backward()
                    if grad_clip > 0 and params:
                        grad_scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(params, grad_clip)
                    grad_scaler.step(optimizer)
                    grad_scaler.update()
                else:
                    loss.backward()
                    if grad_clip > 0 and params:
                        torch.nn.utils.clip_grad_norm_(params, grad_clip)
                    optimizer.step()
            bs = y.size(0)
            tot += float(loss.detach()) * bs
            tot_emb += float(l_emb) * bs
            tot_ce += float(l_ce) * bs
            n += bs
            pred = logits.argmax(-1)
            labeled = y != ign
            if labeled.any():
                corr += int((pred[labeled] == y[labeled]).sum().item())
                n_labeled_acc += int(labeled.sum().item())
            if log_every_batches > 0 and i_batch % int(log_every_batches) == 0:
                pref = f"[{log_prefix}] " if log_prefix else ""
                print(
                    f"  {pref}batch {i_batch}/{n_batches}  "
                    f"loss={tot / n:.5f} emb={tot_emb / n:.5f} ce={tot_ce / n:.5f} "
                    f"acc_labeled={corr / max(n_labeled_acc, 1):.4f}",
                    flush=True,
                )
    denom = max(n, 1)
    acc_labeled = float(corr / max(n_labeled_acc, 1)) if n_labeled_acc else float("nan")
    return tot / denom, tot_emb / denom, tot_ce / denom, acc_labeled


def collate_multiview_pil(batch):
    """``(crops, y)`` batches: stack labels; leave PIL crop lists per sample."""
    crops_b = [b[0] for b in batch]
    y = torch.tensor([b[1] for b in batch], dtype=torch.long)
    return crops_b, y


def collate_multiview_pil_teacher(batch):
    """``(crops, teacher_np, y)``: PIL lists + ``(B, D)`` float teacher + labels."""
    crops_b = [b[0] for b in batch]
    t = torch.from_numpy(np.stack([b[1] for b in batch], axis=0))
    y = torch.tensor([b[2] for b in batch], dtype=torch.long)
    return crops_b, t, y
