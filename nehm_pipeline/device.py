from __future__ import annotations

import logging
from typing import Any, Callable, Tuple

import torch
from torchvision import transforms

logger = logging.getLogger(__name__)


def mps_is_available() -> bool:
    """True when PyTorch was built with MPS and the runtime exposes a Metal device."""
    mps = getattr(torch.backends, "mps", None)
    return bool(mps is not None and mps.is_available())


def resolve_device(preference: str = "mps") -> torch.device:
    """
    Prefer MPS (Apple Silicon), then CUDA, then CPU.
    preference: 'mps' | 'cuda' | 'cpu' — if unavailable, falls down the chain.
    """
    pref = preference.lower().strip()
    if pref == "cpu":
        dev = torch.device("cpu")
        logger.info("Using device: cpu (requested)")
        return dev
    if pref == "cuda":
        if torch.cuda.is_available():
            dev = torch.device("cuda:0")
            logger.info("Using device: cuda:0")
            return dev
        logger.warning("CUDA requested but not available; falling back.")
    elif pref == "mps":
        if mps_is_available():
            dev = torch.device("mps")
            logger.info("Using device: mps")
            return dev
        logger.warning("MPS requested but not available; falling back.")

    if mps_is_available():
        dev = torch.device("mps")
        logger.info("Using device: mps (fallback)")
        return dev
    if torch.cuda.is_available():
        dev = torch.device("cuda:0")
        logger.info("Using device: cuda:0 (fallback)")
        return dev
    dev = torch.device("cpu")
    logger.info("Using device: cpu (fallback)")
    return dev


def dataloader_pin_memory(device: torch.device) -> bool:
    """pin_memory only helps host→CUDA copies; disable on MPS/CPU."""
    return device.type == "cuda"


def apply_pytorch_training_perf_for_vit_student(device: torch.device) -> None:
    """
    Speed hints for **Linear / matmul–heavy** training (CLIP ViT + MLP/decoder student head).

    - **CUDA:** enables cuDNN autotune (good when input shapes are fixed, as here).
    - **All:** ``torch.set_float32_matmul_precision("high")`` where available (e.g. TF32-style
      behavior on NVIDIA Ampere+). **MPS** may ignore or partially apply; safe to call.
    """
    if device.type == "cuda" and torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision("high")
    except (AttributeError, RuntimeError):
        pass


def clip_visual_preprocess_without_channel_normalize(preprocess: Callable) -> Callable:
    """
    Keep OpenAI CLIP's **spatial** steps (resize / center-crop / ``ToTensor``) but drop
    ``Normalize(mean, std)`` so RGB stays **linear** in ``[0, 1]`` (only `/255`` inside
    ``ToTensor``, i.e. values in ``[0, 1]``). Per-channel mean/std would re-center and
    rescale channels and break physically meaningful spectra/RGB ratios; omitting it is
    the default for NEHM.

    Set ``channel_normalize=True`` in :func:`load_openai_clip` to recover classic CLIP
    inputs (needed only for strict compatibility with off-the-shelf zero-shot behavior
    or checkpoints trained with normalization).
    """
    if not isinstance(preprocess, transforms.Compose):
        raise TypeError(
            f"expected torchvision Compose from clip.load, got {type(preprocess).__name__}"
        )
    kept = [t for t in preprocess.transforms if not isinstance(t, transforms.Normalize)]
    return transforms.Compose(kept)


def load_openai_clip(
    model_name: str,
    device: torch.device,
    *,
    channel_normalize: bool = False,
) -> Tuple[Any, Callable]:
    """
    Load OpenAI CLIP (``import clip``). On MPS, weights are staged on CPU first, then
    ``float().to(mps)``, which avoids several dtype/device edge cases with direct MPS load.

    ``channel_normalize``: if False (default), image ``preprocess`` matches CLIP geometry
    but **does not** apply CLIP's per-channel mean/std (linear RGB ``[0, 1]``).
    """
    import clip

    if device.type == "mps":
        model, preprocess = clip.load(model_name, device="cpu")
        model = model.float().to(device)
    else:
        model, preprocess = clip.load(model_name, device=str(device))
        model = model.float()

    if not channel_normalize:
        preprocess = clip_visual_preprocess_without_channel_normalize(preprocess)
        logger.info(
            "CLIP preprocess: spatial + ToTensor only (no per-channel mean/std); RGB in [0, 1]."
        )
    else:
        logger.info("CLIP preprocess: full OpenAI pipeline including Normalize(mean, std).")

    return model, preprocess


def load_openai_clip_nehm(
    model_name: str,
    device: torch.device,
    *,
    channel_normalize: bool = False,
) -> Tuple[Any, Callable]:
    """
    Same behavior as :func:`load_openai_clip` with ``channel_normalize``, but safe when an **older**
    checked-out ``device.py`` only defines the legacy two-argument ``load_openai_clip(model, device)``.

    Notebooks and Colab trees that lag behind the repo should call this (or use a ``TypeError`` fallback)
    so ``channel_normalize`` always works after updating **either** this module **or** the notebook.
    """
    try:
        return load_openai_clip(model_name, device, channel_normalize=channel_normalize)
    except TypeError:
        model, preprocess = load_openai_clip(model_name, device)
        if not channel_normalize:
            preprocess = clip_visual_preprocess_without_channel_normalize(preprocess)
        return model, preprocess
