"""
Trainable MLP + classifier on **precomputed** CLIP / ViT image vectors (``image_embeddings.npy``).

Teacher targets: ``fused_embeddings.npy`` (N, **2304**) = 768-D image (`ln_post` CLS) + 3×512-D text
(pre ``text_projection``). Cluster labels: ``hdbscan_labels.csv``.

Training never runs the ViT backbone. **Inference** uses frozen LongCLIP ``visual`` on PIL images:
same multiview pooling as ``embed.py`` / ``batched_pooled_visual_*``, then this student head.
When using **Pipeline B v3.1 (pre-projection teacher)**, pass ``visual_ln_post_only=True`` / 768-D
pooled vectors into the student.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

logger = logging.getLogger(__name__)


def load_manifest_filenames(manifest_path: Path) -> List[str]:
    """Second column (filename) per row; skip header."""
    lines = Path(manifest_path).read_text(encoding="utf-8").strip().splitlines()
    out: List[str] = []
    for line in lines[1:]:
        parts = line.split("\t")
        if len(parts) >= 2:
            out.append(parts[1].strip())
    return out


# Re-export for notebooks / callers that imported from ``student_clip`` before split.
from nehm_pipeline.manifest_io import (  # noqa: E402
    load_manifest_keys,
    material_column_aligned_to_manifest,
)


def load_hdbscan_labels(path: Path) -> np.ndarray:
    arr = np.loadtxt(path, delimiter=",", dtype=np.int64)
    return arr.ravel()


def build_hdbscan_remap(labels: np.ndarray) -> Tuple[Dict[int, int], int]:
    """
    Map raw HDBSCAN cluster ids (excluding -1) to contiguous class indices 0..K-1.
    Returns (raw_id -> class_index, K).
    """
    u = sorted(int(x) for x in np.unique(labels))
    if -1 in u:
        u.remove(-1)
    raw_to_class = {rid: i for i, rid in enumerate(u)}
    return raw_to_class, len(u)


def hdbscan_remap_to_json_dict(raw_to_class: Dict[int, int], n_clusters: int) -> dict:
    return {
        "noise_label": -1,
        "n_clusters": n_clusters,
        "raw_to_class": {str(k): v for k, v in sorted(raw_to_class.items())},
    }


def build_label_remap_json(labels: np.ndarray) -> dict:
    raw_to_class, k = build_hdbscan_remap(labels)
    return hdbscan_remap_to_json_dict(raw_to_class, k)


def valid_indices_for_distillation(labels: np.ndarray) -> np.ndarray:
    """Row indices where label != -1 (noise)."""
    lab = np.asarray(labels).ravel()
    return np.where(lab != -1)[0]


def y_class_for_rows(
    row_indices: np.ndarray,
    labels_raw: np.ndarray,
    raw_to_class: Dict[int, int],
) -> np.ndarray:
    ri = np.asarray(row_indices, dtype=np.int64)
    return np.array([raw_to_class[int(labels_raw[i])] for i in ri], dtype=np.int64)


def train_val_split_rows(
    row_indices: np.ndarray,
    y_class: np.ndarray,
    val_fraction: float,
    random_state: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Stratified split on ``y_class`` when every class has ≥2 samples; otherwise unstratified shuffle.
    ``row_indices`` and ``y_class`` must have the same length.
    """
    from sklearn.model_selection import train_test_split

    row_indices = np.asarray(row_indices, dtype=np.int64)
    y_class = np.asarray(y_class)
    n = len(row_indices)
    if n == 0:
        return row_indices.copy(), np.array([], dtype=np.int64)
    if val_fraction <= 0 or n < 2:
        return row_indices.copy(), np.array([], dtype=np.int64)

    pos = np.arange(n)
    stratify: Optional[np.ndarray] = None
    _, counts = np.unique(y_class, return_counts=True)
    if len(counts) >= 1 and bool(np.all(counts >= 2)) and int(round(n * val_fraction)) >= 1:
        stratify = y_class

    try:
        pos_tr, pos_va = train_test_split(
            pos,
            test_size=val_fraction,
            random_state=random_state,
            shuffle=True,
            stratify=stratify,
        )
    except ValueError as err:
        logger.warning("Train/val stratified split failed (%s); using shuffle only.", err)
        pos_tr, pos_va = train_test_split(
            pos,
            test_size=val_fraction,
            random_state=random_state,
            shuffle=True,
            stratify=None,
        )
    if stratify is None:
        logger.info("Train/val split: stratify disabled (small classes or val set too small).")
    return row_indices[pos_tr], row_indices[pos_va]


def class_balanced_sampler_weights(
    row_indices: np.ndarray,
    labels_raw: np.ndarray,
    raw_to_class: Dict[int, int],
    n_classes: int,
) -> torch.Tensor:
    """
    One weight per dataset row (same order as ``row_indices``) for ``WeightedRandomSampler``:
    inverse class frequency on the training subset.
    """
    y = y_class_for_rows(row_indices, labels_raw, raw_to_class)
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    inv = 1.0 / np.maximum(counts, 1.0)
    return torch.as_tensor(inv[y], dtype=torch.double)


def make_class_balanced_sampler(
    row_indices: np.ndarray,
    labels_raw: np.ndarray,
    raw_to_class: Dict[int, int],
    n_classes: int,
    num_samples: Optional[int] = None,
) -> WeightedRandomSampler:
    """
    ``num_samples`` defaults to ``len(row_indices)``: that many **weighted random draws**
    per epoch (with replacement). Increase to run more optimizer steps per epoch while
    keeping class rebalancing.
    """
    w = class_balanced_sampler_weights(row_indices, labels_raw, raw_to_class, n_classes)
    n = len(row_indices)
    draws = int(num_samples) if num_samples is not None else n
    if draws < 1:
        draws = n
    return WeightedRandomSampler(w, num_samples=draws, replacement=True)


class PLMImageEmbDistillDataset(Dataset):
    """Rows indexed into ``image_only``, ``teacher``, ``labels_raw`` (global row index ``i``)."""

    def __init__(
        self,
        row_indices: np.ndarray,
        image_only: np.ndarray,
        teacher: np.ndarray,
        labels_raw: np.ndarray,
        raw_to_class: Dict[int, int],
    ) -> None:
        self.row_indices = np.asarray(row_indices, dtype=np.int64)
        self.image_only = image_only.astype(np.float32)
        self.teacher = teacher.astype(np.float32)
        self.labels_raw = np.asarray(labels_raw).ravel()
        self.raw_to_class = raw_to_class

    def __len__(self) -> int:
        return len(self.row_indices)

    def __getitem__(self, j: int):
        i = int(self.row_indices[j])
        e = torch.from_numpy(self.image_only[i].copy())
        t = torch.from_numpy(self.teacher[i].copy())
        y = int(self.raw_to_class[int(self.labels_raw[i])])
        return e, t, y


class _DecoderExpansionBlock(nn.Module):
    """One stage of a vector ``decoder`` tower: linear → LayerNorm → activation → [dropout]."""

    def __init__(
        self,
        d_in: int,
        d_out: int,
        dropout: float,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(d_in, d_out)
        self.norm = nn.LayerNorm(d_out)
        a = activation.lower().strip()
        if a == "gelu":
            self.act: nn.Module = nn.GELU()
        elif a in ("relu",):
            self.act = nn.ReLU(inplace=True)
        else:
            raise ValueError(f"activation must be 'gelu' or 'relu', got {activation!r}")
        self.drop = nn.Dropout(float(dropout)) if float(dropout) > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        x = self.norm(x)
        x = self.act(x)
        return self.drop(x)


class _DecoderExpansionHead(nn.Module):
    """
    Expands a bottleneck vector (e.g. CLIP image 768-D) toward ``embed_dim`` with monotonic
    width growth — analogous to a fully-connected **decoder** (no final norm/act on the code).
    """

    def __init__(
        self,
        image_dim: int,
        embed_dim: int,
        hidden_dims: Tuple[int, ...],
        dropout: float,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        dims = [int(image_dim)] + [int(h) for h in hidden_dims] + [int(embed_dim)]
        blocks: List[nn.Module] = []
        for i in range(len(dims) - 1):
            if i == len(dims) - 2:
                blocks.append(nn.Linear(dims[i], dims[i + 1]))
            else:
                blocks.append(
                    _DecoderExpansionBlock(
                        dims[i], dims[i + 1], dropout=dropout, activation=activation
                    )
                )
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x


class ImageEmbeddingStudent(nn.Module):
    """
    **Decoder-style** expansion (default): ViT pooled image vector (**768-D** `ln_post` CLS for
    LongCLIP-B pre-projection pipeline, or 512-D joint if using legacy checkpoints) → progressively wider
    stages (LayerNorm + GELU) → **2304-D** ``z`` for regression against the fused teacher, plus a
    linear classifier on ``z``.

    The **input is still the precomputed / pooled visual embedding** in that space.
    Use ``head_style=\"mlp\"`` for the older shallow ReLU MLP.
    """

    def __init__(
        self,
        image_dim: int,
        num_classes: int,
        embed_dim: int = 2304,
        mlp_hidden: Tuple[int, int] = (1024, 1536),
        decoder_hidden: Tuple[int, ...] = (1024, 1536, 2048),
        dropout: float = 0.0,
        head_style: str = "decoder",
        decoder_activation: str = "gelu",
        input_dropout: Optional[float] = None,
        decoder_dropout: Optional[float] = None,
        classifier_dropout: Optional[float] = None,
    ) -> None:
        """
        Three-way dropout split (use these instead of the legacy single ``dropout`` knob):

        - ``input_dropout``: applied to the image embedding before ``self.head``. Encourages
          the head to not over-rely on any one input dim. Suggested 0.05.
        - ``decoder_dropout``: dropout inside the head's expansion blocks. **Set 0.0 when the
          regression target (z) is the primary objective** (L1 distillation) — non-zero here
          adds noise to the very signal the L1 loss is trying to recover.
        - ``classifier_dropout``: dropout applied to ``z`` only on the path to ``self.classifier``.
          Regularizes the CE branch while leaving ``z`` clean for the L1/cosine objective.
          Suggested 0.1–0.3.

        For backwards compatibility, if any of the three new knobs is ``None``, ``dropout`` is
        used as a default (so old configs keep working). ``nn.Dropout`` and ``nn.Identity`` have
        no learnable params so existing checkpoints load with strict=True.
        """
        super().__init__()
        style = head_style.lower().strip()
        d = float(dropout)
        in_drop = float(input_dropout) if input_dropout is not None else 0.0
        dec_drop = float(decoder_dropout) if decoder_dropout is not None else d
        cls_drop = float(classifier_dropout) if classifier_dropout is not None else 0.0
        self.input_drop = nn.Dropout(in_drop) if in_drop > 0 else nn.Identity()
        if style == "decoder":
            self.head = _DecoderExpansionHead(
                image_dim=image_dim,
                embed_dim=embed_dim,
                hidden_dims=decoder_hidden,
                dropout=dec_drop,
                activation=decoder_activation,
            )
        elif style == "mlp":
            h1, h2 = mlp_hidden
            layers: List[nn.Module] = [
                nn.Linear(image_dim, h1),
                nn.ReLU(inplace=True),
            ]
            if dec_drop > 0:
                layers.append(nn.Dropout(dec_drop))
            layers.extend(
                [
                    nn.Linear(h1, h2),
                    nn.ReLU(inplace=True),
                ]
            )
            if dec_drop > 0:
                layers.append(nn.Dropout(dec_drop))
            layers.append(nn.Linear(h2, embed_dim))
            self.head = nn.Sequential(*layers)
        else:
            raise ValueError(f"head_style must be 'decoder' or 'mlp', got {head_style!r}")
        self.classifier_drop = nn.Dropout(cls_drop) if cls_drop > 0 else nn.Identity()
        self.classifier = nn.Linear(embed_dim, num_classes)

    def set_head_trainable(self, trainable: bool) -> None:
        for p in self.head.parameters():
            p.requires_grad = trainable

    def set_classifier_trainable(self, trainable: bool) -> None:
        for p in self.classifier.parameters():
            p.requires_grad = trainable

    def forward(self, image_emb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        f = image_emb.float()
        f = self.input_drop(f)
        z = self.head(f)
        logits = self.classifier(self.classifier_drop(z))
        return z, logits


class PLMMaterialDistillDataset(Dataset):
    """
    Like ``PLMImageEmbDistillDataset`` but targets are **material** ``logit`` ids from a spreadsheet
    (``-1`` = no CE target). Indexed by global pipeline row ``i``.
    """

    def __init__(
        self,
        row_indices: np.ndarray,
        image_only: np.ndarray,
        teacher: np.ndarray,
        material_logits: np.ndarray,
    ) -> None:
        self.row_indices = np.asarray(row_indices, dtype=np.int64)
        self.image_only = image_only.astype(np.float32)
        self.teacher = teacher.astype(np.float32)
        self.material_logits = np.asarray(material_logits, dtype=np.int64).ravel()

    def __len__(self) -> int:
        return len(self.row_indices)

    def __getitem__(self, j: int):
        i = int(self.row_indices[j])
        e = torch.from_numpy(self.image_only[i].copy())
        t = torch.from_numpy(self.teacher[i].copy())
        y = int(self.material_logits[i])
        return e, t, y


def load_material_logits_from_excel(
    path: Path,
    n_rows: int,
    sheet_name: str = "Sheet2",
    column: str = "logit",
) -> np.ndarray:
    """Load labels for pipeline rows ``0 .. n_rows-1`` from ``Material_Database.xlsx``.

    The sheet may list **all** DB rows (e.g. 5410) while embeddings align to a **prefix**
    (e.g. 5351). After sorting by ``Image Index``, we keep the first ``n_rows`` rows and
    require ``Image Index == 0, 1, …, n_rows-1`` for that prefix.
    """
    import pandas as pd

    df = pd.read_excel(Path(path), sheet_name=sheet_name)
    if "Image Index" not in df.columns or column not in df.columns:
        raise ValueError(f"Expected 'Image Index' and '{column}' in {path}")
    df = df.sort_values("Image Index").reset_index(drop=True)
    if len(df) < n_rows:
        raise ValueError(
            f"Excel rows {len(df)} < pipeline rows {n_rows}: spreadsheet must cover indices 0..{n_rows - 1}"
        )
    df = df.iloc[: int(n_rows)].copy()
    idx = df["Image Index"].astype("int64", copy=False).to_numpy()
    expected = np.arange(int(n_rows), dtype=np.int64)
    if not np.array_equal(idx, expected):
        raise ValueError(
            f"After sort by Image Index, first {n_rows} rows must be indices 0..{n_rows - 1}; "
            f"got Image Index head {idx[: min(12, len(idx))].tolist()}{'…' if len(idx) > 12 else ''}"
        )
    return df[column].to_numpy(dtype=np.int64)


def reconstruction_loss(
    z: torch.Tensor,
    teacher: torch.Tensor,
    kind: str = "mse",
) -> torch.Tensor:
    """
    Decoder-style target on the 2304-D vector.

    - ``mse``: mean squared error over all elements (common "L2" / Gauss likelihood objective).
    - ``l1``: mean absolute error (more robust / sparse errors).
    """
    k = kind.lower().strip()
    if k == "mse":
        return F.mse_loss(z, teacher)
    if k in ("l1", "mae"):
        return F.l1_loss(z, teacher)
    raise ValueError(f"kind must be 'mse' or 'l1', got {kind!r}")


def joint_material_loss(
    z: torch.Tensor,
    teacher: torch.Tensor,
    logits: torch.Tensor,
    y_material: torch.Tensor,
    lambda_emb: float = 1.0,
    lambda_ce: float = 1.0,
    emb_kind: str = "mse",
    label_smoothing: float = 0.0,
    ignore_index: int = -1,
    ce_class_weights: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Reconstruction on **all** batch rows; CE only where ``y_material != ignore_index``.
    If every target in the batch is ``ignore_index``, CE is treated as 0 (PyTorch CE would be NaN).

    ``ce_class_weights``: optional ``(num_classes,)`` tensor for ``F.cross_entropy(..., weight=…)``
    (e.g. inverse-frequency / balanced weights on the **training** label distribution).
    """
    l_emb = reconstruction_loss(z, teacher, kind=emb_kind)
    ign = int(ignore_index)
    labeled = y_material != ign
    if labeled.any():
        if ce_class_weights is not None:
            l_ce = F.cross_entropy(
                logits,
                y_material,
                weight=ce_class_weights,
                ignore_index=ign,
                label_smoothing=float(label_smoothing),
            )
        else:
            l_ce = F.cross_entropy(
                logits,
                y_material,
                ignore_index=ign,
                label_smoothing=float(label_smoothing),
            )
    else:
        l_ce = z.new_zeros(())
    total = float(lambda_emb) * l_emb + float(lambda_ce) * l_ce
    return total, l_emb.detach(), l_ce.detach()


def student_embedding_joint_epoch(
    student: nn.Module,
    loader: DataLoader,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
    *,
    train: bool,
    lambda_emb: float,
    lambda_ce: float,
    emb_kind: str,
    label_smoothing: float,
    ignore_index: int = -1,
    ce_class_weights: Optional[torch.Tensor] = None,
    grad_clip: float = 0.0,
) -> Tuple[float, float, float, float]:
    """
    One epoch over ``PLMMaterialDistillDataset`` batches ``(image_emb, fused_teacher, y)``.

    Same joint objective as :func:`joint_material_loss` (recon on all rows; CE with
    ``ignore_index``). Returns mean **total**, **recon**, **CE**, and **classification
    accuracy counting only rows with** ``y != ignore_index`` (typically material ≥ 0).
    """
    if train:
        student.train()
    else:
        student.eval()
    tot = tot_emb = tot_ce = 0.0
    n = 0
    corr = 0
    n_labeled = 0
    params = [p for p in student.parameters() if p.requires_grad]
    ctx = torch.enable_grad() if train else torch.inference_mode()
    ign = int(ignore_index)
    with ctx:
        for e, t, y in loader:
            e = e.to(device)
            t = t.to(device, dtype=torch.float32)
            y = y.to(device)
            if train and optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            z, logits = student(e)
            loss, l_emb, l_ce = joint_material_loss(
                z,
                t,
                logits,
                y,
                lambda_emb=lambda_emb,
                lambda_ce=lambda_ce,
                emb_kind=emb_kind,
                label_smoothing=label_smoothing,
                ignore_index=ign,
                ce_class_weights=ce_class_weights if train else None,
            )
            if train and optimizer is not None:
                loss.backward()
                if grad_clip > 0 and params:
                    torch.nn.utils.clip_grad_norm_(params, grad_clip)
                optimizer.step()
            bs = int(y.size(0))
            tot += float(loss.detach()) * bs
            tot_emb += float(l_emb) * bs
            tot_ce += float(l_ce) * bs
            n += bs
            labeled = y != ign
            if labeled.any():
                pred = logits.argmax(dim=-1)
                corr += int((pred[labeled] == y[labeled]).sum().item())
                n_labeled += int(labeled.sum().item())
    denom = max(n, 1)
    acc = float(corr / max(n_labeled, 1)) if n_labeled else float("nan")
    return tot / denom, tot_emb / denom, tot_ce / denom, acc


def class_balanced_sampler_weights_from_labels(
    y_class: np.ndarray,
    n_classes: int,
) -> torch.Tensor:
    """Inverse-frequency weights for labels in ``0..n_classes-1`` (training subset rows)."""
    y = np.asarray(y_class, dtype=np.int64).ravel()
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    inv = 1.0 / np.maximum(counts, 1.0)
    return torch.as_tensor(inv[y], dtype=torch.double)


def make_class_balanced_sampler_from_y(
    y_per_row: np.ndarray,
    n_classes: int,
    num_samples: Optional[int] = None,
) -> WeightedRandomSampler:
    w = class_balanced_sampler_weights_from_labels(y_per_row, n_classes)
    n = len(y_per_row)
    draws = int(num_samples) if num_samples is not None else n
    if draws < 1:
        draws = n
    return WeightedRandomSampler(w, num_samples=draws, replacement=True)


def material_ce_class_weights_tensor(
    y_train: np.ndarray,
    num_classes: int,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Per-class weights for ``cross_entropy(..., weight=…)``: sklearn **balanced** style
    ``n_samples / (n_classes * count_c)`` for classes that appear in ``y_train``; classes
    with zero count get weight ``0``. Scaled so the mean over classes with positive weight
    is ``1`` (keeps loss scale comparable to unweighted CE).
    """
    y = np.asarray(y_train, dtype=np.int64).ravel()
    counts = np.bincount(y, minlength=int(num_classes)).astype(np.float64)
    n = float(len(y))
    k = float(int(num_classes))
    w = np.zeros(int(num_classes), dtype=np.float64)
    for c in range(int(num_classes)):
        if counts[c] > 0:
            w[c] = n / (k * counts[c])
    pos = w > 0
    if pos.any():
        w[pos] /= w[pos].mean()
    return torch.as_tensor(w, device=device, dtype=dtype)


@torch.inference_mode()
def pooled_clip_visual_embedding(
    pil_crops: List[Image.Image],
    preprocess,
    visual: nn.Module,
    device: torch.device,
    *,
    ln_post_only: bool = False,
) -> torch.Tensor:
    """
    Encode each crop with CLIP ``visual`` (or ``ln_post`` CLS only when ``ln_post_only``), L2-normalize per row,
    mean-pool, L2-normalize vector.
    Returns shape (D,) float32 on ``device``.
    """
    if not pil_crops:
        raise ValueError("pil_crops must be non-empty")
    from nehm_pipeline.vit_ln_post_utils import vit_visual_ln_post_forward

    clip_dtype = next(visual.parameters()).dtype
    batch = torch.stack([preprocess(p) for p in pil_crops]).to(device=device, dtype=clip_dtype)
    if ln_post_only:
        e = vit_visual_ln_post_forward(visual, batch).float()
    else:
        e = visual(batch).float()
    e = e / e.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    pooled = e.mean(dim=0)
    pooled = pooled / pooled.norm().clamp(min=1e-12)
    return pooled


def infer_from_pil(
    pil: Image.Image,
    student: ImageEmbeddingStudent,
    visual: nn.Module,
    preprocess,
    device: torch.device,
    num_random_crops: int = 8,
    rng: Optional[np.random.Generator] = None,
    *,
    visual_ln_post_only: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Inference: random 336² crops after height→336, pooled ViT embedding, then student.
    Returns ``(z, logits)`` each shape (embed_dim,) and (num_classes,).
    """
    from nehm_pipeline.preprocess_image import random_square_crops_after_height_336

    crops = random_square_crops_after_height_336(pil, target=336, n=num_random_crops, rng=rng)
    emb = pooled_clip_visual_embedding(
        crops, preprocess, visual, device, ln_post_only=visual_ln_post_only
    )
    # ViT path uses @inference_mode → inference tensor; clone so Linear (trainable weights) can run.
    emb = emb.clone()
    student.eval()
    with torch.inference_mode():
        z, logits = student(emb.unsqueeze(0))
    return z.squeeze(0), logits.squeeze(0)


def distill_loss(
    z: torch.Tensor,
    teacher: torch.Tensor,
    logits: torch.Tensor,
    y_class: torch.Tensor,
    lambda_emb: float = 1.0,
    lambda_ce: float = 1.0,
    label_smoothing: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """MSE on embedding + cross-entropy on cluster logits."""
    l_emb = F.mse_loss(z, teacher)
    l_ce = F.cross_entropy(logits, y_class, label_smoothing=float(label_smoothing))
    total = lambda_emb * l_emb + lambda_ce * l_ce
    return total, l_emb.detach(), l_ce.detach()


def load_distillation_bundle(
    pipeline_out: Path,
    images_dir: Optional[Path] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Load ``fused_embeddings.npy``, ``image_embeddings.npy``, ``hdbscan_labels.csv``, ``manifest.tsv``.

    ``images_dir`` is accepted for call-site compatibility; it is not read here.
    """
    pipeline_out = Path(pipeline_out)
    fused = np.load(pipeline_out / "fused_embeddings.npy")
    image_only = np.load(pipeline_out / "image_embeddings.npy")
    labels = load_hdbscan_labels(pipeline_out / "hdbscan_labels.csv")
    fnames = load_manifest_filenames(pipeline_out / "manifest.tsv")
    if not (len(fnames) == len(labels) == len(fused) == len(image_only)):
        raise ValueError(
            f"Length mismatch: manifest {len(fnames)}, labels {len(labels)}, "
            f"fused {len(fused)}, image_only {len(image_only)}"
        )
    return fused, image_only, labels, fnames
