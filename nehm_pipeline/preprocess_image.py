"""Clip_clusterb-style PIL preprocessing before CLIP ``preprocess`` (ViT-L/14@336px)."""

from __future__ import annotations

from typing import List, Optional, Tuple, Union

import numpy as np
from PIL import Image


def compute_adaptive_tile_px(
    pil_or_size: Union[Image.Image, Tuple[int, int]],
    *,
    tile_min_px: int = 16,
    tile_max_px: int = 224,
    tile_scale_ref_px: int = 3000,
) -> int:
    """Field-of-view-aware tile side, in native pixels.

    Computes the geometric mean of the image dimensions ``sqrt(W * H)`` and maps
    it linearly to a tile side, such that an image whose geometric-mean side is
    ``tile_scale_ref_px`` returns exactly ``tile_max_px``. Smaller images get
    proportionally smaller tiles (clipped at ``tile_min_px``); larger images
    saturate at ``tile_max_px``.

    Rationale (NEHM B v3.1 multiview): the dataset spans roughly 150 - 3020-pixel
    short sides across sources, but every image is approximately the same
    physical field of view (a single microscopy / Pigment Compendium frame).
    Sampling a *fixed* pixel size from every image therefore samples wildly
    different *physical* fractions of the FOV. Scaling the tile side with the
    image preserves a roughly constant fraction of the FOV per tile, and gives
    the ViT something close to a constant amount of *content* per token-row no
    matter the source resolution.

    Returned value is in native pixels (no resize is applied here; this is the
    side length to pass to a subsequent crop call, e.g.
    :func:`random_square_crops_native_resolution`).
    """
    if isinstance(pil_or_size, Image.Image):
        w, h = pil_or_size.size
    else:
        w, h = int(pil_or_size[0]), int(pil_or_size[1])
    if w <= 0 or h <= 0:
        return int(tile_min_px)
    if tile_scale_ref_px <= 0:
        raise ValueError(f"tile_scale_ref_px must be positive, got {tile_scale_ref_px!r}")
    if tile_max_px < tile_min_px:
        raise ValueError(
            f"tile_max_px ({tile_max_px}) must be >= tile_min_px ({tile_min_px})"
        )
    geo = float(w * h) ** 0.5
    raw = float(tile_max_px) * geo / float(tile_scale_ref_px)
    clamped = min(float(tile_max_px), max(float(tile_min_px), raw))
    return int(round(clamped))


def random_square_crops_adaptive_native_resolution(
    pil: Image.Image,
    n: int,
    rng: np.random.Generator,
    *,
    tile_min_px: int = 16,
    tile_max_px: int = 224,
    tile_scale_ref_px: int = 3000,
) -> Tuple[List[Image.Image], int]:
    """Field-of-view-aware variant of :func:`random_square_crops_native_resolution`.

    Computes the per-image tile side via :func:`compute_adaptive_tile_px` and
    then draws ``n`` independent uniform random tiles of that size at native
    resolution, with the same narrow-image upsample fallback as the fixed-size
    helper. Returns ``(crops, tile_px_used)`` so the caller can log the actual
    tile size that was sampled.
    """
    tile_px = compute_adaptive_tile_px(
        pil,
        tile_min_px=tile_min_px,
        tile_max_px=tile_max_px,
        tile_scale_ref_px=tile_scale_ref_px,
    )
    crops = random_square_crops_native_resolution(pil, tile_px, n, rng)
    return crops, tile_px


def pil_train_geom_augment(
    pil: Image.Image,
    rng: np.random.Generator,
    *,
    max_rotation_deg: float = 5.0,
    hflip_p: float = 0.5,
    vflip_p: float = 0.5,
) -> Image.Image:
    """
    **Before** height-resize / square cropping: optional flips and small in-plane rotation.

    Uses ``expand=True`` on rotate so corners are not clipped. Fill is black.
    """
    pil = pil.convert("RGB")
    if rng.random() < hflip_p:
        pil = pil.transpose(Image.FLIP_LEFT_RIGHT)
    if rng.random() < vflip_p:
        pil = pil.transpose(Image.FLIP_TOP_BOTTOM)
    if max_rotation_deg > 0:
        ang = float(rng.uniform(-max_rotation_deg, max_rotation_deg))
        if abs(ang) > 1e-6:
            pil = pil.rotate(
                ang,
                resample=Image.BILINEAR,
                expand=True,
                fillcolor=(0, 0, 0),
            )
    return pil


def resize_height_preserve_aspect(img: Image.Image, new_height: int) -> Image.Image:
    width, height = img.size
    if height == 0:
        return img
    aspect_ratio = width / height
    new_width = int(round(new_height * aspect_ratio))
    return img.resize((new_width, new_height), Image.BILINEAR)


def notebook_single_crop_336(pil: Image.Image, target: int = 336) -> Image.Image:
    """
    Match Notebooks/Clip_clusterb.ipynb: resize so height is ``target`` (preserve aspect),
    then center-crop ``target``×``target``. If width is smaller than ``target`` after the
    height resize, scale width up to ``target`` (height may exceed ``target``) so the
    crop is always square.
    """
    pil = pil.convert("RGB")
    pil = resize_height_preserve_aspect(pil, target)
    w, h = pil.size
    if w < target:
        pil = pil.resize((target, int(round(h * target / w))), Image.BILINEAR)
        w, h = pil.size
    left = (w - target) // 2
    top = (h - target) // 2
    return pil.crop((left, top, left + target, top + target))


def random_square_crops_after_height_336(
    pil: Image.Image,
    target: int = 336,
    n: int = 1,
    rng: Optional[np.random.Generator] = None,
) -> List[Image.Image]:
    """
    Resize so height is ``target`` (preserve aspect), then take ``n`` independent uniform random
    ``target``×``target`` crops (same narrow-width upsample rule as ``notebook_single_crop_336``).

    Use this for **inference-time** views: multiple crops are pooled (mean of L2-normalized CLIP
    image embeddings) before the student MLP, matching multiview pooling in ``embed.encode_dataset``.
    """
    if n <= 0:
        return []
    rng = rng or np.random.default_rng()
    pil = pil.convert("RGB")
    pil = resize_height_preserve_aspect(pil, target)
    w, h = pil.size
    if w < target:
        pil = pil.resize((target, int(round(h * target / w))), Image.BILINEAR)
        w, h = pil.size
    max_x, max_y = w - target, h - target
    out: List[Image.Image] = []
    for _ in range(n):
        x = int(rng.integers(0, max_x + 1))
        y = int(rng.integers(0, max_y + 1))
        out.append(pil.crop((x, y, x + target, y + target)))
    return out


def random_square_crops_native_resolution(
    pil: Image.Image,
    crop_px: int,
    n: int,
    rng: np.random.Generator,
) -> List[Image.Image]:
    """
    Take ``n`` independent **uniform** random ``crop_px``×``crop_px`` crops from the image in its
    **native pixel resolution** (no global downscale before cropping). If ``min(w,h) < crop_px``,
    the image is upsampled so a square crop of side ``crop_px`` exists, then crops are drawn.

    Each crop is passed through CLIP ``preprocess`` separately; mean-pooling those embeddings
    yields a high-resolution multi-view image representation.
    """
    if n <= 0:
        return []
    pil = pil.convert("RGB")
    w, h = pil.size
    if min(w, h) < crop_px:
        scale = crop_px / float(min(w, h))
        nw = max(int(round(w * scale)), crop_px)
        nh = max(int(round(h * scale)), crop_px)
        pil = pil.resize((nw, nh), Image.BILINEAR)
        w, h = pil.size
    max_x, max_y = w - crop_px, h - crop_px
    out: List[Image.Image] = []
    for _ in range(n):
        x = int(rng.integers(0, max_x + 1))
        y = int(rng.integers(0, max_y + 1))
        out.append(pil.crop((x, y, x + crop_px, y + crop_px)))
    return out


def short_side_resize_center_crop(
    pil: Image.Image,
    target: int,
    *,
    pre_crop_margin_px: int = 0,
) -> Image.Image:
    """
    Resize so the **short** side equals ``target`` (preserve aspect), then center-crop ``target``×``target``.

    Mirrors the global-view convention used by the **B** image-embedding scheme
    (``LongCLIP_Embeddings_v1.run_longclip_embeddings.make_global_height_center_crop`` with
    height = crop_size). With short-side resize the global view always preserves the
    smaller dimension verbatim and crops the longer one symmetrically.

    ``pre_crop_margin_px`` (default 0): if > 0, drop that many pixels from each side of the
    raw image **before** the short-side resize. Use 1 to ignore the typical 1-pixel scan-border
    artifact present on Pigment Compendium-style book scans (the entire **B v2** pipeline passes 1).
    The argument defaults to 0 so legacy callers (Pipeline A, Pipeline B v1) are unchanged.
    """
    pil = pil.convert("RGB")
    w, h = pil.size
    if h == 0 or w == 0:
        return pil
    m = max(int(pre_crop_margin_px), 0)
    if m > 0 and w > 2 * m and h > 2 * m:
        pil = pil.crop((m, m, w - m, h - m))
        w, h = pil.size
    short = min(w, h)
    scale = float(target) / float(short)
    new_w = max(int(round(w * scale)), target)
    new_h = max(int(round(h * scale)), target)
    pil = pil.resize((new_w, new_h), Image.BILINEAR)
    left = (new_w - target) // 2
    top = (new_h - target) // 2
    return pil.crop((left, top, left + target, top + target))
