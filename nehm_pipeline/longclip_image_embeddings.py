"""
Importable LongCLIP image **and text** embedding generator for the **B** preprocessing scheme.

Pipeline **B v3.1 (LongCLIP edition, scale-adaptive multiview)** — **pre-projection teacher**:

- **Image embeddings (768-D, LongCLIP-B ViT-B/16):** CLS after ``visual.ln_post``, **before** ``visual.proj``. Per row, ``n_random_crops`` random
  **scale-adaptive** native-resolution tiles **plus** one global view.
  Each tile's side length in native pixels is chosen per-image as

      tile_px = clip(round(tile_max_px * sqrt(W*H) / tile_scale_ref_px),
                     tile_min_px, tile_max_px)

  with defaults ``tile_min_px = 16``, ``tile_max_px = 224``,
  ``tile_scale_ref_px = 3000``. This yields **224×224 tiles for the largest
  images** (~3000-pixel geometric-mean side; e.g. McCrone) and **down to 16×16
  tiles for the smallest images** (~150-pixel side; e.g. Emrath), so each tile
  samples a roughly constant fraction of the image's *physical field of view*
  rather than a fixed pixel area. Set ``random_crop_px > 0`` to fall back to
  the legacy fixed-size mode (Pipeline A: 16, B v1: 16, B v2: n/a — global only).

  The global view drops ``pre_crop_margin_px`` border pixels first, then short-
  side resizes to ``global_target_px`` and center-crops ``global_target_px``²
  (default 224 for LongCLIP-B). Tiles are L2 → mean-pool → L2; the global view
  is L2-normalized.

- **Text embeddings (512-D, LongCLIP-B):** EOT hidden state after ``ln_final``, **before**
  ``text_projection`` (same width as the transformer). Per DB column, 248-token context.

Channel normalization is the **caller's choice**: pass in either LongCLIP's standard
``preprocess`` (with CLIP's per-channel ``Normalize(mean, std)``) or the result of
``nehm_pipeline.device.clip_visual_preprocess_without_channel_normalize(preprocess)`` to keep
RGB linear in ``[0, 1]`` (the NEHM framework default — preserves microscopy-relevant per-channel
intensity ratios).

This is the *teacher-side* mirror of the **B v3.1 student preprocessing** in
``LabeledMultiviewFinetuneDataset(tile_px=None, tile_min_px=16, tile_max_px=224,
tile_scale_ref_px=3000, global_kind='short_side', n_native_random=16, crop_size=224,
pre_crop_margin_px=1)``, so teacher and student see the same view layout, the same backbone,
and the same channel scaling.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image

from nehm_pipeline.preprocess_image import (
    compute_adaptive_tile_px,
    random_square_crops_adaptive_native_resolution,
    random_square_crops_native_resolution,
    short_side_resize_center_crop,
)
from nehm_pipeline.vit_ln_post_utils import vit_visual_ln_post_forward

# LongCLIP-B (ViT-B/16) teacher without joint-space projections:
# fused teacher = 768 (image ln_post CLS) + 3 × 512 (text pre text_projection) = 2304.
LONGCLIP_B_IMAGE_LN_POST_DIM = 768
LONGCLIP_B_TEXT_PRE_PROJECTION_DIM = 512
LONGCLIP_B_FUSED_TEACHER_DIM = LONGCLIP_B_IMAGE_LN_POST_DIM + 3 * LONGCLIP_B_TEXT_PRE_PROJECTION_DIM


def clip_longclip_encode_text_after_ln_final(
    clip_model: torch.nn.Module, text_tokens: torch.Tensor
) -> torch.Tensor:
    """EOT vector after ``ln_final``, before ``@ text_projection`` (512-D on LongCLIP-B)."""
    x = clip_model.token_embedding(text_tokens).type(clip_model.dtype)
    dev = x.device
    x = x + (
        clip_model.positional_embedding.to(dev) * clip_model.mask1.to(dev)
    ).type(clip_model.dtype) + (
        clip_model.positional_embedding_res.to(dev) * clip_model.mask2.to(dev)
    ).type(clip_model.dtype)
    x = x.permute(1, 0, 2)
    x = clip_model.transformer(x)
    x = x.permute(1, 0, 2)
    x = clip_model.ln_final(x).type(clip_model.dtype)
    return x[torch.arange(x.shape[0], device=dev), text_tokens.argmax(dim=-1)]


@dataclass
class LongClipBImageEmbedConfig:
    """Hyperparameters for the **B**-scheme LongCLIP-B image embedding pass.

    **Pipeline B v3.1 defaults** are scale-adaptive multiview:

    - ``n_random_crops      = 16``    — 16 native random tiles per image
    - ``random_crop_px      = 0``     — ``0`` ⇒ scale-adaptive mode (recommended)
    - ``tile_min_px         = 16``    — smallest adaptive tile side (matches ViT-B/16 patch)
    - ``tile_max_px         = 224``   — largest adaptive tile side (matches LongCLIP-B input)
    - ``tile_scale_ref_px   = 3000``  — ``sqrt(W*H)`` value that maps to ``tile_max_px``
    - ``global_target_px    = 224``   — LongCLIP-B native input
    - ``pre_crop_margin_px  = 1``     — strip 1-pixel scan border on Pigment Compendium scans

    Backwards compatibility: setting ``random_crop_px > 0`` reverts to fixed-size
    tiles (Pipeline A / B v1 behavior). Setting ``n_random_crops = 0`` reproduces
    the global-only B v2 layout. Setting ``random_crop_px = 24`` and
    ``global_target_px = 336`` reproduces B v2's ViT-L/14@336 geometry.
    """

    n_random_crops: int = 16          # B v3.1 default: 16 native random tiles per image
    random_crop_px: int = 0           # 0 ⇒ scale-adaptive (recommended); >0 ⇒ fixed-size legacy
    tile_min_px: int = 16             # B v3.1: smallest adaptive tile side
    tile_max_px: int = 224            # B v3.1: largest adaptive tile side (= LongCLIP-B input)
    tile_scale_ref_px: int = 3000     # B v3.1: sqrt(W*H) image side that maps to tile_max_px
    global_target_px: int = 224       # LongCLIP-B native input resolution (use 336 for ViT-L/14)
    image_batch_size: int = 32        # per-tile minibatch on the GPU (random-crop branch)
    random_seed: int = 1337
    pre_crop_margin_px: int = 1       # B v3: strip 1px scan border before short-side resize
    image_ln_post_only: bool = True   # 768-D CLS after ``ln_post`` (no ``visual.proj``)


def _l2_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(n, eps)


def _encode_pil_batch(
    visual_or_clip_model: torch.nn.Module,
    preprocess: Callable[[Image.Image], torch.Tensor],
    pils: Sequence[Image.Image],
    device: torch.device,
    *,
    use_encode_image: bool,
    ln_post_only: bool = False,
) -> np.ndarray:
    """Encode a batch of PIL crops; returns ``(B, D)`` float32 numpy, L2-normalized per row.

    ``ln_post_only``: if True, run the ViT through ``ln_post``(CLS) only (768-D on ViT-B);
    ``use_encode_image`` is ignored in that case and ``visual_or_clip_model`` must be ``.visual``.
    """
    non_blocking = device.type == "cuda"
    batch = torch.stack([preprocess(p) for p in pils]).to(device, non_blocking=non_blocking)
    with torch.inference_mode():
        if ln_post_only:
            emb = vit_visual_ln_post_forward(visual_or_clip_model, batch)
        elif use_encode_image and hasattr(visual_or_clip_model, "encode_image"):
            emb = visual_or_clip_model.encode_image(batch)
        else:
            emb = visual_or_clip_model(batch)
    emb = emb.float()
    emb = emb / emb.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    return emb.detach().cpu().numpy().astype(np.float32)


def encode_image_random_pooled_b(
    clip_model: torch.nn.Module,
    preprocess: Callable[[Image.Image], torch.Tensor],
    pil: Image.Image,
    device: torch.device,
    *,
    n_random_crops: int,
    random_crop_px: int,
    image_batch_size: int,
    rng: np.random.Generator,
    use_encode_image: bool = True,
    tile_min_px: int = 16,
    tile_max_px: int = 224,
    tile_scale_ref_px: int = 3000,
    image_ln_post_only: bool = True,
) -> Tuple[np.ndarray, int]:
    """L2 → mean → L2 over ``n_random_crops`` random tiles at native res.

    If ``random_crop_px > 0``: legacy **fixed-size** mode — every tile is exactly
    ``random_crop_px``×``random_crop_px``.

    If ``random_crop_px <= 0``: **scale-adaptive** mode — the tile side is
    computed per-image as
    ``clip(round(tile_max_px * sqrt(W*H) / tile_scale_ref_px), tile_min_px, tile_max_px)``,
    so larger images get larger tiles (saturating at ``tile_max_px``) and
    smaller images get smaller tiles (saturating at ``tile_min_px``). This
    keeps the tile a roughly constant fraction of the image's physical FOV.

    Returns ``(pooled_emb, tile_px_used)`` so callers can log the actual tile
    size that was sampled per row.
    """
    if random_crop_px > 0:
        crops = random_square_crops_native_resolution(pil, random_crop_px, n_random_crops, rng)
        tile_px_used = int(random_crop_px)
    else:
        crops, tile_px_used = random_square_crops_adaptive_native_resolution(
            pil,
            n_random_crops,
            rng,
            tile_min_px=tile_min_px,
            tile_max_px=tile_max_px,
            tile_scale_ref_px=tile_scale_ref_px,
        )
    if not crops:
        raise RuntimeError("Zero random crops requested")
    enc = clip_model.visual if image_ln_post_only else clip_model
    parts: List[np.ndarray] = []
    for start in range(0, len(crops), image_batch_size):
        chunk = crops[start : start + image_batch_size]
        parts.append(
            _encode_pil_batch(
                enc,
                preprocess,
                chunk,
                device,
                use_encode_image=use_encode_image,
                ln_post_only=image_ln_post_only,
            )
        )
        if device.type == "mps":
            try:
                torch.mps.empty_cache()
            except Exception:
                pass
    e = np.vstack(parts)
    pooled = e.mean(axis=0, dtype=np.float32)
    return _l2_rows(pooled.reshape(1, -1))[0].astype(np.float32), tile_px_used


def encode_image_global_b(
    clip_model: torch.nn.Module,
    preprocess: Callable[[Image.Image], torch.Tensor],
    pil: Image.Image,
    device: torch.device,
    *,
    global_target_px: int,
    pre_crop_margin_px: int = 0,
    use_encode_image: bool = True,
    image_ln_post_only: bool = True,
) -> np.ndarray:
    """L2-normalized embedding of one short-side-resized + center-cropped global view.

    ``pre_crop_margin_px``: drop that many pixels from each side of ``pil`` before the resize
    (use 1 for **B v3** to ignore the 1-pixel scan-border artifact on Pigment Compendium scans).
    """
    global_pil = short_side_resize_center_crop(
        pil, global_target_px, pre_crop_margin_px=pre_crop_margin_px
    )
    enc = clip_model.visual if image_ln_post_only else clip_model
    e = _encode_pil_batch(
        enc,
        preprocess,
        [global_pil],
        device,
        use_encode_image=use_encode_image,
        ln_post_only=image_ln_post_only,
    )
    return e[0].astype(np.float32)


def regenerate_longclip_b_image_embeddings(
    *,
    clip_model: torch.nn.Module,
    preprocess: Callable[[Image.Image], torch.Tensor],
    device: torch.device,
    image_dir: Path,
    image_filenames: Sequence[str],
    n_rows: int,
    embed_dim: int = LONGCLIP_B_IMAGE_LN_POST_DIM,
    out_dir: Path,
    cfg: LongClipBImageEmbedConfig = LongClipBImageEmbedConfig(),
    use_encode_image: bool = True,
    progress: bool = True,
    skip_existing_rows: bool = True,
) -> Tuple[Optional[Path], Path, Path]:
    """
    Encode every image in ``image_filenames`` (one per global-manifest row) under the **B**
    preprocessing scheme on **LongCLIP-B**, writing npys aligned to the manifest:

    - ``out_dir / "image_embeddings_B_random_pooled.npy"``  — ``(n_rows, embed_dim)``
      (only when ``cfg.n_random_crops > 0``; ``None`` otherwise).
    - ``out_dir / "image_embeddings_B_global.npy"``         — ``(n_rows, embed_dim)``
    - ``out_dir / "image_manifest_B.csv"``                  — per-row resolved path + status

    Rows whose filename is empty or whose image cannot be loaded stay ``np.nan`` (skipped during
    fusion). When ``skip_existing_rows`` and the npys already exist, rows that already have a
    finite embedding are not re-encoded — useful for resuming a long run on Colab.

    Returns ``(random_path_or_None, global_path, manifest_path)``.
    """
    if n_rows <= 0:
        raise ValueError("n_rows must be positive")
    if embed_dim <= 0:
        raise ValueError("embed_dim must be positive")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    do_random = int(cfg.n_random_crops) > 0
    random_path = (out_dir / "image_embeddings_B_random_pooled.npy") if do_random else None
    global_path = out_dir / "image_embeddings_B_global.npy"
    manifest_path = out_dir / "image_manifest_B.csv"

    if skip_existing_rows and global_path.is_file() and (not do_random or random_path.is_file()):
        glob_arr = np.load(global_path).astype(np.float32)
        if glob_arr.shape != (n_rows, embed_dim):
            raise ValueError(
                f"existing global npy has wrong shape: {glob_arr.shape}, expected "
                f"({n_rows}, {embed_dim}); delete it or use skip_existing_rows=False"
            )
        if do_random:
            rand_arr = np.load(random_path).astype(np.float32)
            if rand_arr.shape != (n_rows, embed_dim):
                raise ValueError(
                    f"existing random npy has wrong shape: {rand_arr.shape}, expected "
                    f"({n_rows}, {embed_dim}); delete it or use skip_existing_rows=False"
                )
        else:
            rand_arr = None
    else:
        glob_arr = np.full((n_rows, embed_dim), np.nan, dtype=np.float32)
        rand_arr = np.full((n_rows, embed_dim), np.nan, dtype=np.float32) if do_random else None

    is_adaptive = do_random and int(cfg.random_crop_px) <= 0
    iterator: Iterable[int] = range(n_rows)
    if progress:
        from tqdm.auto import tqdm
        if not do_random:
            desc = (
                f"B global-only LongCLIP-B encode "
                f"(global @ {cfg.global_target_px}, pre_crop={cfg.pre_crop_margin_px}px)"
            )
        elif is_adaptive:
            desc = (
                f"B v3.1 LongCLIP-B encode "
                f"({cfg.n_random_crops}× scale-adaptive tiles "
                f"[{cfg.tile_min_px}-{cfg.tile_max_px}px @ ref={cfg.tile_scale_ref_px}] "
                f"+ global @ {cfg.global_target_px}, pre_crop={cfg.pre_crop_margin_px}px)"
            )
        else:
            desc = (
                f"B v3 LongCLIP-B encode "
                f"({cfg.n_random_crops}× {cfg.random_crop_px}px tiles + global @ "
                f"{cfg.global_target_px}, pre_crop={cfg.pre_crop_margin_px}px)"
            )
        iterator = tqdm(iterator, total=n_rows, desc=desc)

    manifest_rows: List[dict] = []
    for i in iterator:
        name = str(image_filenames[i]).strip() if i < len(image_filenames) else ""
        path = (Path(image_dir) / name) if name else None
        exists = bool(path is not None and path.is_file())
        row_meta: dict = {
            "row_index": i,
            "image_name": name,
            "resolved_path": str(path) if path is not None else "",
            "exists": exists,
            "image_w_px": -1,
            "image_h_px": -1,
            "image_geo_px": -1.0,
            "tile_px_used": -1,
        }
        manifest_rows.append(row_meta)
        if not exists:
            continue
        already = (
            skip_existing_rows
            and not np.isnan(glob_arr[i]).all()
            and (not do_random or not np.isnan(rand_arr[i]).all())
        )
        if already:
            continue
        try:
            with Image.open(path) as raw:
                raw.load()
                pil = raw.convert("RGB")
        except Exception:
            continue
        # Record native size **after** the pre-crop margin trim, since that's the image
        # the random tiles + global view actually see.
        m = max(int(cfg.pre_crop_margin_px), 0)
        if m > 0:
            w0, h0 = pil.size
            if w0 > 2 * m and h0 > 2 * m:
                pil_for_tiles = pil.crop((m, m, w0 - m, h0 - m))
            else:
                pil_for_tiles = pil
        else:
            pil_for_tiles = pil
        w_post, h_post = pil_for_tiles.size
        row_meta["image_w_px"] = int(w_post)
        row_meta["image_h_px"] = int(h_post)
        row_meta["image_geo_px"] = float((w_post * h_post) ** 0.5)
        try:
            if do_random:
                rng = np.random.default_rng(int(cfg.random_seed) + i)
                rand_emb, tile_used = encode_image_random_pooled_b(
                    clip_model,
                    preprocess,
                    pil_for_tiles,
                    device,
                    n_random_crops=cfg.n_random_crops,
                    random_crop_px=cfg.random_crop_px,
                    image_batch_size=cfg.image_batch_size,
                    rng=rng,
                    use_encode_image=use_encode_image,
                    tile_min_px=cfg.tile_min_px,
                    tile_max_px=cfg.tile_max_px,
                    tile_scale_ref_px=cfg.tile_scale_ref_px,
                    image_ln_post_only=cfg.image_ln_post_only,
                )
                rand_arr[i] = rand_emb
                row_meta["tile_px_used"] = int(tile_used)
            # The global view trims internally, so pass the original `pil` (not the
            # already-trimmed `pil_for_tiles`) to avoid a double-crop margin.
            glob_arr[i] = encode_image_global_b(
                clip_model,
                preprocess,
                pil,
                device,
                global_target_px=cfg.global_target_px,
                pre_crop_margin_px=cfg.pre_crop_margin_px,
                use_encode_image=use_encode_image,
                image_ln_post_only=cfg.image_ln_post_only,
            )
        except Exception as err:  # don't kill a multi-hour run for one bad image
            print(f"  row {i} ({name}): encode failed ({type(err).__name__}: {err})")
            continue

    np.save(global_path, glob_arr)
    if do_random:
        np.save(random_path, rand_arr)
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)
    return random_path, global_path, manifest_path


def _sanitize_filename(name: str) -> str:
    import re
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", str(name).strip())
    value = re.sub(r"_+", "_", value).strip("_")
    return value or "column"


def encode_text_batch_longclip(
    clip_model: torch.nn.Module,
    longclip_module,
    texts: Sequence[str],
    device: torch.device,
    *,
    pre_projection: bool = True,
) -> np.ndarray:
    """Encode a batch of strings through **LongCLIP's** text tower (248-token context).

    ``pre_projection`` (default True): return EOT after ``ln_final`` (512-D on LongCLIP-B),
    **before** ``text_projection``. Set False for legacy 512-D **joint** ``encode_text``.

    Falls back through a few common LongCLIP tokenizer entry points so this works against
    minor fork variations (some forks expose ``longclip.tokenize``, others use
    ``simple_tokenizer.tokenize``).

    Returns ``(B, D)`` float32 L2-normalized.
    """
    tokenize = getattr(longclip_module, "tokenize", None)
    if tokenize is None:
        # Some LongCLIP forks expose tokenize via simple_tokenizer at the package root.
        st = getattr(longclip_module, "simple_tokenizer", None)
        if st is not None and hasattr(st, "tokenize"):
            tokenize = st.tokenize  # type: ignore[assignment]
    if tokenize is None:
        raise AttributeError(
            "Could not locate `tokenize` on the loaded LongCLIP module. "
            "Expected `longclip_module.tokenize` (mirror of `clip.tokenize`)."
        )

    # LongCLIP's tokenize defaults to its 248-token context window. We force `truncate=True`
    # so any pathologically long Description rows just get clipped instead of raising.
    try:
        tokens = tokenize(list(texts), truncate=True)
    except TypeError:
        # Older LongCLIP forks don't accept `truncate=True`; fall back without it.
        tokens = tokenize(list(texts))

    tokens = tokens.to(device)
    with torch.inference_mode():
        if pre_projection:
            emb = clip_longclip_encode_text_after_ln_final(clip_model, tokens)
        else:
            emb = clip_model.encode_text(tokens)
    emb = emb.float()
    emb = emb / emb.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    return emb.detach().cpu().numpy().astype(np.float32)


def regenerate_longclip_b_text_embeddings(
    *,
    clip_model: torch.nn.Module,
    longclip_module,
    device: torch.device,
    df: pd.DataFrame,
    out_dir: Path,
    columns: Optional[Sequence[str]] = None,
    text_batch_size: int = 64,
    embed_dim: int = LONGCLIP_B_TEXT_PRE_PROJECTION_DIM,
    skip_columns: Sequence[str] = ("Image", "Image Index"),
    progress: bool = True,
    pre_projection: bool = True,
) -> Path:
    """
    For each non-skipped column in ``df``, encode every row's stringified value through
    **LongCLIP's** text tower (248-token context) and write a per-column ``.npy`` to
    ``out_dir``. Also writes a ``columns_manifest.csv`` in the same format the existing fusion
    code expects (one row per column with ``column_name, embedding_file, rows, embedding_dim``).

    Returns the manifest path.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "columns_manifest.csv"

    if columns is None:
        columns = [c for c in df.columns if c not in set(skip_columns)]
    else:
        columns = [c for c in columns if c not in set(skip_columns)]

    n_rows = int(len(df))
    manifest_rows: List[dict] = []
    iterator: Iterable[str] = columns
    if progress:
        from tqdm.auto import tqdm
        iterator = tqdm(
            columns,
            desc=f"B v3 LongCLIP text-encode ({n_rows} rows × {len(columns)} cols, 248-tok ctx)",
        )

    for col in iterator:
        values = df[col].fillna("").astype(str).tolist()
        out = np.empty((n_rows, embed_dim), dtype=np.float32)
        for start in range(0, n_rows, text_batch_size):
            chunk = values[start : start + text_batch_size]
            emb = encode_text_batch_longclip(
                clip_model, longclip_module, chunk, device, pre_projection=pre_projection
            )
            if emb.shape[1] != embed_dim:
                raise ValueError(
                    f"column {col!r}: text encoder dim {emb.shape[1]} != expected {embed_dim} "
                    f"(check LongCLIP-B text width or set pre_projection / embed_dim consistently)."
                )
            out[start : start + emb.shape[0]] = emb
        filename = f"{_sanitize_filename(col)}.npy"
        np.save(out_dir / filename, out)
        manifest_rows.append(
            {
                "column_name": col,
                "embedding_file": filename,
                "rows": n_rows,
                "embedding_dim": embed_dim,
            }
        )

    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)
    return manifest_path


def fill_missing_with_zeros(arr: np.ndarray) -> np.ndarray:
    """Replace NaN rows with zeros (so fusion downstream doesn't propagate NaN)."""
    out = np.array(arr, dtype=np.float32, copy=True)
    bad = np.isnan(out).any(axis=1)
    out[bad] = 0.0
    return out
