"""
Classical approximate K-SVD for pooled embeddings.

Dictionary update follows the **approximate K-SVD** loop from Karl Skretting's
Dictionary Learning Tools (Rubinstein et al., batch OMP technical report), as
documented at https://www.ux.uis.no/~karlsk/dle/ — atom ``k`` is updated from
the active residual patch via a normalized least-squares direction instead of a
full rank-1 SVD.

Sparse encoding supports OMP, Lasso, or Elastic Net. When
``sparse_codes_nonnegative`` is enabled with Elastic Net, coefficients are
constrained to be non-negative (atoms may still be zeroed by pruning).

Layout (hybrid convention):
    X: (n_samples, n_features)
    D: (n_atoms, n_features)   row atoms
    R: (n_samples, n_atoms)    sparse codes
    X_recon = R @ D
"""

from __future__ import annotations

import json
import warnings
from contextlib import contextmanager
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import sparse_encode
from sklearn.exceptions import ConvergenceWarning
from tqdm.auto import tqdm


@dataclass
class ClassicalKsvdConfig:
    input_dim: int = 2304
    source_input_dim: int = 2304
    embedding_block: str = "full"  # full | dates_period_512 | culture_artist_512 | name_desc_512 | image_768 | custom
    block_start: int = 0
    block_end: int = 2304
    n_atoms: int = 400
    max_iter: int = 30
    ksvd_iter: int = 1
    sparse_method: str = "omp"  # omp | lasso | elastic_net
    omp_n_nonzero: int = 40
    lasso_alpha: float = 0.5
    lasso_algorithm: str = "lasso_cd"
    elastic_net_alpha: float = 1e-5
    elastic_net_l1_ratio: float = 0.5
    elastic_net_max_iter: int = 1000
    sparse_codes_nonnegative: bool = False
    refine_r_iters: int = 0  # optional post-pass like hybrid step 3
    refine_r_lr: float = 0.01
    refine_r_lambda: float = 0.5
    preprocess_mode: str = "raw"  # raw | standardize
    subsample_fraction: float = 1.0  # 1.0 = full batch; e.g. 0.05 = online mini-batch (5% random paintings/step)
    active_eps: float = 1e-4
    reinit_dead_atoms: bool = True
    min_dict_change: float = 1e-5
    patience: int = 3
    seed: int = 42
    # Underperforming-atom pruning (specimen memorization guard)
    prune_enabled: bool = True
    min_atom_support: int = 20
    max_top5_concentration: float = 0.4
    prune_every_iters: int = 3
    replacement_mode: str = "residual_svd"  # residual_svd | weighted_centroid | max_residual
    residual_pool_size: int = 200
    max_prune_per_cycle: int = 50
    max_abs_code: float = 1e4

    # aliases for older configs
    @property
    def outer_iters(self) -> int:
        return int(self.max_iter)


def load_classical_ksvd_config(path: Path | str) -> ClassicalKsvdConfig:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    allowed = {f.name for f in fields(ClassicalKsvdConfig)}
    kwargs = {k: v for k, v in data.items() if k in allowed}
    if "outer_iters" in data and "max_iter" not in kwargs:
        kwargs["max_iter"] = int(data["outer_iters"])
    return ClassicalKsvdConfig(**kwargs)


def save_classical_ksvd_config(cfg: ClassicalKsvdConfig, path: Path | str) -> None:
    Path(path).write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")


@contextmanager
def _suppress_sparse_convergence_warnings():
    """Silence sklearn coordinate-descent 'did not converge' noise during sparse coding."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        yield


def normalize_rows(d: np.ndarray) -> np.ndarray:
    d = np.nan_to_num(np.asarray(d, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    norms = np.linalg.norm(d, axis=1, keepdims=True)
    out = d / np.maximum(norms, 1e-12)
    out[norms[:, 0] <= 1e-12] = 0.0
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def _sanitize_matrix(x: np.ndarray, *, clip_abs: float | None = None) -> np.ndarray:
    out = np.nan_to_num(np.asarray(x, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if clip_abs is not None and float(clip_abs) > 0:
        out = np.clip(out, -float(clip_abs), float(clip_abs), out=out)
    return out.astype(np.float32, copy=False)


def initial_dictionary_from_samples(x: np.ndarray, n_atoms: int, *, seed: int = 42) -> np.ndarray:
    """X (n_samples, features) -> D (n_atoms, features)."""
    rng = np.random.default_rng(seed)
    n_samples = x.shape[0]
    idx = rng.choice(n_samples, size=n_atoms, replace=n_atoms > n_samples)
    return normalize_rows(x[idx].copy()).astype(np.float32)


def _elastic_net_coef_at_alpha(alphas: np.ndarray, coefs: np.ndarray, target_alpha: float) -> np.ndarray:
    """Pick coefficients from ``enet_path`` output at ``target_alpha``."""
    alphas = np.asarray(alphas, dtype=np.float64)
    coefs = np.asarray(coefs, dtype=np.float32)
    if coefs.ndim != 3:
        raise ValueError(f"expected coefs (n_samples, n_atoms, n_alphas), got {coefs.shape}")
    target = float(target_alpha)
    if target >= float(alphas[0]):
        return coefs[:, :, 0]
    if target <= float(alphas[-1]):
        return coefs[:, :, -1]
    idx = int(np.searchsorted(-alphas, -target, side="left"))
    idx = min(max(idx, 0), int(coefs.shape[2]) - 1)
    return coefs[:, :, idx]


def _nonnegative_sparse_codes(cfg: ClassicalKsvdConfig) -> bool:
    return bool(getattr(cfg, "sparse_codes_nonnegative", False))


def _active_code_mask(r: np.ndarray, active_eps: float, *, nonnegative: bool) -> np.ndarray:
    r = np.asarray(r, dtype=np.float32)
    if nonnegative:
        return r > float(active_eps)
    return np.abs(r) > float(active_eps)


def _clamp_nonnegative_codes(r: np.ndarray) -> np.ndarray:
    return np.maximum(np.asarray(r, dtype=np.float32), 0.0)


def _fit_positive_elastic_net_row(
    xi: np.ndarray,
    d_T: np.ndarray,
    *,
    alpha: float,
    l1_ratio: float,
    max_iter: int,
) -> np.ndarray:
    from sklearn.linear_model import ElasticNet

    en = ElasticNet(
        alpha=float(alpha),
        l1_ratio=float(l1_ratio),
        positive=True,
        fit_intercept=False,
        max_iter=int(max_iter),
    )
    with _suppress_sparse_convergence_warnings():
        en.fit(d_T, np.asarray(xi, dtype=np.float64))
    return np.maximum(en.coef_, 0.0).astype(np.float32, copy=False)


def elastic_net_codes(
    x: np.ndarray,
    d: np.ndarray,
    *,
    alpha: float,
    l1_ratio: float,
    max_iter: int = 1000,
    max_abs_code: float = 1e4,
    positive: bool = False,
    show_progress: bool = False,
    center: np.ndarray | None = None,
) -> np.ndarray:
    """Batch Elastic Net sparse coding with fixed dictionary rows.

    ``center`` (optional): mean vector subtracted from every row of ``x`` before
    coding, for dictionaries trained on mean-centered embeddings.

    Solves ``y ≈ r @ D`` for every row of ``x`` against fixed atom rows ``d``.
    This is a **spectral / atom-dimension** problem: spatial ``70×70`` structure is
    already flattened into independent patch rows. Cost is dominated by
    ``n_patch_rows × embedding_dim`` (e.g. 4900×2304 per image), not by browsing
    atoms one-at-a-time in the UI.

    With a **fixed alpha**, runtime is weakly dependent on ``n_atoms`` (typically
    20 vs 200 is similar). Avoid ``enet_path`` regularization grids at inference.
    """
    from sklearn.linear_model import enet_path

    x = _sanitize_matrix(x, clip_abs=float(max_abs_code))
    if center is not None:
        center_row = np.asarray(center, dtype=np.float32).reshape(1, -1)
        if center_row.shape[1] != x.shape[1]:
            raise ValueError(f"center dim {center_row.shape[1]} != x dim {x.shape[1]}")
        x = (x - center_row).astype(np.float32, copy=False)
    d = normalize_rows(d).astype(np.float32, copy=False)
    if x.ndim != 2:
        raise ValueError(f"x must be 2-D, got {x.shape}")
    if d.ndim != 2 or d.shape[1] != x.shape[1]:
        raise ValueError(f"d shape {d.shape} incompatible with x shape {x.shape}")
    if x.shape[0] == 0:
        return np.zeros((0, d.shape[0]), dtype=np.float32)

    if positive:
        from joblib import Parallel, delayed

        d_T = d.T.astype(np.float64, copy=False)
        n_jobs = -1 if x.shape[0] > 32 else 1
        row_indices = range(x.shape[0])
        if show_progress:
            row_indices = tqdm(
                row_indices,
                desc="non-neg elastic net encode",
                unit="sample",
                leave=False,
            )
        rows = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(_fit_positive_elastic_net_row)(
                x[i],
                d_T,
                alpha=float(alpha),
                l1_ratio=float(l1_ratio),
                max_iter=int(max_iter),
            )
            for i in row_indices
        )
        return _sanitize_matrix(np.vstack(rows), clip_abs=float(max_abs_code))

    if show_progress:
        tqdm.write(f"elastic net encode: {x.shape[0]:,} samples (batched)")
    target_alpha = float(alpha)
    with _suppress_sparse_convergence_warnings():
        alphas, coefs, _ = enet_path(
            d.T,
            x.T,
            l1_ratio=float(l1_ratio),
            alphas=[target_alpha],
            max_iter=int(max_iter),
        )
    if coefs.ndim == 3:
        r = np.asarray(coefs[:, :, 0], dtype=np.float32)
    else:
        r = _elastic_net_coef_at_alpha(alphas, coefs, target_alpha)
    return _sanitize_matrix(r, clip_abs=float(max_abs_code))


def elastic_net_patch_codes_for_image_ranks(
    flat_mmap: np.ndarray,
    image_ranks: np.ndarray,
    *,
    dictionary: np.ndarray,
    block_start: int,
    block_end: int,
    patch_grid: int,
    alpha: float,
    l1_ratio: float,
    max_iter: int = 1000,
    positive: bool = False,
    show_progress: bool = False,
    center: np.ndarray | None = None,
) -> dict[int, np.ndarray]:
    """Batch sparse coding for many images' flattened patch rows against one ``D_sub``."""
    image_ranks = np.sort(np.unique(np.asarray(image_ranks, dtype=np.int64)))
    if image_ranks.size == 0:
        return {}
    patches_per_image = int(patch_grid) * int(patch_grid)
    chunks: list[np.ndarray] = []
    for image_rank in image_ranks:
        start = int(image_rank) * patches_per_image
        end = start + patches_per_image
        patches = np.asarray(flat_mmap[start:end, int(block_start) : int(block_end)], dtype=np.float32)
        patches = np.nan_to_num(patches, nan=0.0, posinf=0.0, neginf=0.0)
        chunks.append(patches)
    codes = elastic_net_codes(
        np.vstack(chunks),
        dictionary,
        alpha=float(alpha),
        l1_ratio=float(l1_ratio),
        max_iter=int(max_iter),
        positive=bool(positive),
        show_progress=bool(show_progress),
        center=center,
    )
    return {
        int(image_rank): np.asarray(codes[i * patches_per_image : (i + 1) * patches_per_image], dtype=np.float32)
        for i, image_rank in enumerate(image_ranks)
    }


def elastic_net_image_codes(
    x: np.ndarray,
    d: np.ndarray,
    *,
    alpha: float = 1e-5,
    l1_ratio: float = 0.5,
    max_iter: int = 1000,
    positive: bool = False,
    show_progress: bool = False,
    center: np.ndarray | None = None,
) -> np.ndarray:
    """Fit image-level Elastic Net codes for all rows of ``x`` in one batched solve."""
    return elastic_net_codes(
        x,
        d,
        alpha=float(alpha),
        l1_ratio=float(l1_ratio),
        max_iter=int(max_iter),
        positive=bool(positive),
        show_progress=bool(show_progress),
        center=center,
    )


def sparse_codes(
    x: np.ndarray,
    d: np.ndarray,
    cfg: ClassicalKsvdConfig,
    *,
    show_progress: bool = False,
) -> np.ndarray:
    """X (n_samples, features), D (n_atoms, features) -> R (n_samples, n_atoms)."""
    x = _sanitize_matrix(x, clip_abs=float(cfg.max_abs_code))
    d = normalize_rows(d).astype(np.float32, copy=False)
    method = str(cfg.sparse_method).lower().strip()
    with _suppress_sparse_convergence_warnings():
        if method == "omp":
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="Orthogonal matching pursuit ended prematurely.*",
                    category=RuntimeWarning,
                )
                r = sparse_encode(
                    x,
                    d,
                    algorithm="omp",
                    n_nonzero_coefs=int(cfg.omp_n_nonzero),
                )
        elif method == "lasso":
            r = sparse_encode(
                x,
                d,
                algorithm=str(cfg.lasso_algorithm),
                alpha=float(cfg.lasso_alpha),
                max_iter=5000,
                positive=_nonnegative_sparse_codes(cfg),
            )
            if show_progress:
                tqdm.write(
                    f"lasso encode: {x.shape[0]:,} samples "
                    f"(positive={_nonnegative_sparse_codes(cfg)}, alpha={cfg.lasso_alpha:g})"
                )
        elif method == "elastic_net":
            r = elastic_net_codes(
                x,
                d,
                alpha=float(cfg.elastic_net_alpha),
                l1_ratio=float(cfg.elastic_net_l1_ratio),
                max_iter=int(cfg.elastic_net_max_iter),
                max_abs_code=float(cfg.max_abs_code),
                positive=_nonnegative_sparse_codes(cfg),
                show_progress=show_progress,
            )
        else:
            raise ValueError(f"unknown sparse_method {cfg.sparse_method!r}; use omp, lasso, or elastic_net")
    r = _sanitize_matrix(r, clip_abs=float(cfg.max_abs_code))
    if _nonnegative_sparse_codes(cfg):
        r = _clamp_nonnegative_codes(r)
    return r


def approximate_ksvd_update_rows(
    x: np.ndarray,
    d: np.ndarray,
    r: np.ndarray,
    *,
    active_eps: float,
    reinit_dead_atoms: bool,
    max_abs_code: float = 1e4,
    nonnegative_codes: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Approximate K-SVD on row atoms (Skretting DLE / Rubinstein batch-OMP variant).

    Hybrid / Student_Embed residual update with X = R @ D.
    """
    x = _sanitize_matrix(x, clip_abs=max_abs_code)
    d = normalize_rows(d).astype(np.float32, copy=False)
    r = _sanitize_matrix(r, clip_abs=max_abs_code)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        residual = x - r @ d
    residual = _sanitize_matrix(residual)
    n_atoms = d.shape[0]

    active_mask = _active_code_mask(r, active_eps, nonnegative=nonnegative_codes)

    for k in range(n_atoms):
        active = np.flatnonzero(active_mask[:, k])
        if len(active) == 0:
            if reinit_dead_atoms:
                j = int(np.argmax(np.linalg.norm(residual, axis=1)))
                atom = residual[j].copy()
                norm = np.linalg.norm(atom)
                if norm > 1e-12:
                    d[k] = atom / norm
            continue

        rk = r[active, k]
        ri = residual[active] + np.outer(rk, d[k])
        ri = _sanitize_matrix(ri)
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            dk = _sanitize_matrix(rk.T @ ri)
        norm = np.linalg.norm(dk)
        if norm <= 1e-12:
            continue
        d[k] = dk / norm
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            rk_new = _sanitize_matrix(ri @ d[k], clip_abs=max_abs_code)
        if nonnegative_codes:
            rk_new = _clamp_nonnegative_codes(rk_new)
        r[active, k] = rk_new
        residual[active] = ri - np.outer(r[active, k], d[k])
        residual[active] = _sanitize_matrix(residual[active])

    if nonnegative_codes:
        r = _clamp_nonnegative_codes(r)
    return normalize_rows(d).astype(np.float32, copy=False), _sanitize_matrix(r, clip_abs=max_abs_code)


def atom_usage_metrics(
    r: np.ndarray,
    *,
    active_eps: float,
    nonnegative: bool = False,
) -> dict[str, np.ndarray]:
    """Per-atom support counts and top-5 activation concentration."""
    r = np.asarray(r, dtype=np.float32)
    active_mask = _active_code_mask(r, active_eps, nonnegative=nonnegative)
    support = active_mask.sum(axis=0).astype(np.int64)
    mass = np.abs(r)
    n_atoms = r.shape[1]
    concentration = np.ones(n_atoms, dtype=np.float32)
    for k in range(n_atoms):
        col = mass[:, k]
        total = float(col.sum())
        if total <= 0.0:
            continue
        n_top = min(5, int((col > 0).sum()))
        if n_top <= 0:
            continue
        top_vals = np.partition(col, -n_top)[-n_top:]
        concentration[k] = float(top_vals.sum() / total)
    return {
        "support": support,
        "concentration_top5": concentration,
        "mean_abs_code": mass.mean(axis=0),
        "max_abs_code": mass.max(axis=0),
    }


def identify_underperforming_atoms(
    r: np.ndarray,
    cfg: ClassicalKsvdConfig,
) -> np.ndarray:
    """Atom indices failing support and/or concentration thresholds."""
    metrics = atom_usage_metrics(
        r,
        active_eps=float(cfg.active_eps),
        nonnegative=_nonnegative_sparse_codes(cfg),
    )
    support = metrics["support"]
    concentration = metrics["concentration_top5"]
    bad = (support < int(cfg.min_atom_support)) | (
        concentration > float(cfg.max_top5_concentration)
    )
    prune_ix = np.flatnonzero(bad)
    if prune_ix.size == 0:
        return prune_ix
    order = np.lexsort((concentration[prune_ix], support[prune_ix]))
    prune_ix = prune_ix[order]
    cap = int(cfg.max_prune_per_cycle)
    if cap > 0 and prune_ix.size > cap:
        prune_ix = prune_ix[:cap]
    return prune_ix


def _orthogonalize_atom(
    atom: np.ndarray,
    basis_rows: np.ndarray,
    *,
    min_cosine: float = 0.85,
) -> np.ndarray:
    out = atom.astype(np.float32, copy=True)
    for row in basis_rows:
        row = row.astype(np.float32, copy=False)
        cos = float(np.dot(out, row))
        if abs(cos) >= float(min_cosine):
            out = out - cos * row
    norm = float(np.linalg.norm(out))
    if norm <= 1e-12:
        return np.zeros_like(out, dtype=np.float32)
    return (out / norm).astype(np.float32, copy=False)


def _atom_from_residual_batch(
    residual_batch: np.ndarray,
    existing_rows: np.ndarray,
    *,
    mode: str,
    rng: np.random.Generator,
) -> np.ndarray:
    batch = np.asarray(residual_batch, dtype=np.float32)
    if batch.shape[0] == 0:
        return np.zeros(batch.shape[1] if batch.ndim == 2 else 0, dtype=np.float32)
    mode = str(mode).lower().strip()
    if mode == "max_residual":
        j = int(np.argmax(np.linalg.norm(batch, axis=1)))
        atom = batch[j]
    elif mode == "weighted_centroid":
        weights = np.linalg.norm(batch, axis=1).astype(np.float32)
        w_sum = float(weights.sum())
        if w_sum <= 1e-12:
            j = int(rng.integers(batch.shape[0]))
            atom = batch[j]
        else:
            atom = (batch * weights[:, None]).sum(axis=0) / w_sum
    else:
        _u, _s, vt = np.linalg.svd(batch, full_matrices=False)
        atom = vt[0]
    return _orthogonalize_atom(atom, existing_rows)


def prune_and_reinit_atoms(
    x: np.ndarray,
    d: np.ndarray,
    r: np.ndarray,
    cfg: ClassicalKsvdConfig,
    *,
    prune_ix: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, int, dict[str, Any]]:
    """
    Zero underperforming atoms and re-seed from high-error residual directions.

    Returns updated D, R, number pruned, and diagnostic info.
    """
    x = _sanitize_matrix(x, clip_abs=float(cfg.max_abs_code))
    d = normalize_rows(d).astype(np.float32, copy=True)
    r = _sanitize_matrix(r, clip_abs=float(cfg.max_abs_code)).copy()
    if rng is None:
        rng = np.random.default_rng(int(cfg.seed))

    if prune_ix is None:
        prune_ix = identify_underperforming_atoms(r, cfg)
    else:
        prune_ix = np.asarray(prune_ix, dtype=np.int64)

    if prune_ix.size == 0:
        return d, r, 0, {"pruned_atoms": [], "mse_before": None, "mse_after": None}

    active_eps = float(cfg.active_eps)
    nonnegative = _nonnegative_sparse_codes(cfg)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        mse_before = float(np.mean(_sanitize_matrix((x - r @ d) ** 2)))
    pool_size = max(10, int(cfg.residual_pool_size))
    replacement_mode = str(cfg.replacement_mode)

    new_rows: list[np.ndarray] = []
    for k in prune_ix:
        users = np.flatnonzero(_active_code_mask(r[:, k], active_eps, nonnegative=nonnegative))
        old_atom = d[k].copy()
        old_coef = r[:, k].copy()
        r[:, k] = 0.0
        d[k] = 0.0

        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            residual = _sanitize_matrix(x - r @ d)
        if users.size > 0 and replacement_mode == "weighted_centroid":
            batch = residual[users] + np.outer(old_coef[users], old_atom)
        else:
            err_norm = np.linalg.norm(residual, axis=1)
            top_ix = np.argpartition(err_norm, -min(pool_size, err_norm.size))[-pool_size:]
            top_ix = top_ix[np.argsort(err_norm[top_ix])[::-1]]
            batch = residual[top_ix]

        basis = np.vstack([d, np.vstack(new_rows)]) if new_rows else d
        atom = _atom_from_residual_batch(batch, basis, mode=replacement_mode, rng=rng)
        if float(np.linalg.norm(atom)) <= 1e-12:
            j = int(np.argmax(np.linalg.norm(residual, axis=1)))
            atom = _orthogonalize_atom(residual[j], basis)
        if float(np.linalg.norm(atom)) > 1e-12:
            d[k] = atom
            new_rows.append(atom)

    n_pruned = int(prune_ix.size)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        mse_after = float(np.mean(_sanitize_matrix((x - r @ d) ** 2)))
    info = {
        "pruned_atoms": prune_ix.astype(int).tolist(),
        "mse_before": mse_before,
        "mse_after": mse_after,
    }
    return d.astype(np.float32, copy=False), r, n_pruned, info


def reconstruction_stats(
    x: np.ndarray,
    d: np.ndarray,
    r: np.ndarray,
    *,
    active_eps: float,
    cfg: ClassicalKsvdConfig | None = None,
) -> dict[str, float]:
    x = _sanitize_matrix(x, clip_abs=float(cfg.max_abs_code) if cfg is not None else 1e4)
    d = normalize_rows(d).astype(np.float32, copy=False)
    r = _sanitize_matrix(r, clip_abs=float(cfg.max_abs_code) if cfg is not None else 1e4)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        recon = _sanitize_matrix(r @ d)
    nonnegative = bool(cfg is not None and _nonnegative_sparse_codes(cfg))
    active_mask = _active_code_mask(r, active_eps, nonnegative=nonnegative)
    active = active_mask.sum(axis=1)
    dead = active_mask.sum(axis=0) == 0
    stats: dict[str, float] = {
        "mse": float(np.mean((x - recon) ** 2)),
        "sparsity": float(np.mean(np.abs(r) <= active_eps)),
        "mean_active": float(active.mean()),
        "min_active": int(active.min()),
        "max_active": int(active.max()),
        "dead_atoms": int(dead.sum()),
    }
    if cfg is not None:
        metrics = atom_usage_metrics(r, active_eps=active_eps, nonnegative=nonnegative)
        under = identify_underperforming_atoms(r, cfg)
        stats["underperforming_atoms"] = int(under.size)
        stats["min_atom_support"] = int(metrics["support"].min())
        stats["median_atom_support"] = float(np.median(metrics["support"]))
    return stats


def refine_codes_fixed_dictionary(
    x: np.ndarray,
    d: np.ndarray,
    r: np.ndarray,
    cfg: ClassicalKsvdConfig,
    *,
    verbose: bool = False,
) -> np.ndarray:
    """Gradient refinement of R with fixed D (hybrid step 3, signed)."""
    x = _sanitize_matrix(x, clip_abs=float(cfg.max_abs_code))
    d = normalize_rows(d).astype(np.float32, copy=False)
    r = _sanitize_matrix(r, clip_abs=float(cfg.max_abs_code)).copy()
    lr = float(cfg.refine_r_lr)
    lam = float(cfg.refine_r_lambda)
    for it in range(1, int(cfg.refine_r_iters) + 1):
        err = _sanitize_matrix(x - r @ d)
        grad = -2.0 * (err @ d.T) + lam * np.sign(r)
        grad = _sanitize_matrix(grad)
        r = _sanitize_matrix(r - lr * grad, clip_abs=float(cfg.max_abs_code))
        if _nonnegative_sparse_codes(cfg):
            r = _clamp_nonnegative_codes(r)
        if verbose and it % 10 == 0:
            mse = float(np.mean(err**2))
            print(f"  refine R {it}: mse={mse:.6f}")
    return r.astype(np.float32, copy=False)


def fit_approx_dictionary(
    x: np.ndarray,
    cfg: ClassicalKsvdConfig,
    *,
    verbose: bool = True,
    checkpoint_dir: Path | str | None = None,
    checkpoint_every_iters: int = 1,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """
    Train on all samples X (n_samples, n_features).

    Returns D (n_atoms, n_features) and history.
    """
    x = _sanitize_matrix(x, clip_abs=float(cfg.max_abs_code))
    finite_rows = np.isfinite(x).all(axis=1)
    if not finite_rows.all():
        x = x[finite_rows]
    if x.ndim != 2 or x.shape[0] == 0:
        raise ValueError(f"x must be non-empty 2-D finite matrix, got {x.shape}")
    rng = np.random.default_rng(int(cfg.seed))
    d = initial_dictionary_from_samples(x, int(cfg.n_atoms), seed=int(cfg.seed))
    history: list[dict[str, Any]] = []
    stall = 0

    def save_checkpoint(reason: str, iteration: int) -> None:
        if checkpoint_dir is None:
            return
        out_dir = Path(checkpoint_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / "ksvd_checkpoint_dictionary.npy", d.astype(np.float32, copy=False))
        history_to_dataframe(history).to_csv(out_dir / "ksvd_checkpoint_history.csv", index=False)
        (out_dir / "ksvd_checkpoint_config.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")
        checkpoint_meta = {
            "reason": reason,
            "iteration": int(iteration),
            "n_history_rows": int(len(history)),
            "dictionary_shape": list(d.shape),
        }
        (out_dir / "ksvd_checkpoint_meta.json").write_text(json.dumps(checkpoint_meta, indent=2), encoding="utf-8")
        if verbose:
            tqdm.write(
                f"  checkpoint saved ({reason}) @ iter {iteration}: "
                f"{out_dir / 'ksvd_checkpoint_dictionary.npy'}"
            )

    def _draw_minibatch() -> tuple[np.ndarray, np.ndarray]:
        """Fresh random subsample for one online encode+update step."""
        if float(cfg.subsample_fraction) < 1.0:
            n_sub = max(int(cfg.n_atoms), int(round(x.shape[0] * float(cfg.subsample_fraction))))
            sub_ix = rng.choice(x.shape[0], size=min(n_sub, x.shape[0]), replace=False)
            return x[sub_ix], sub_ix
        return x, np.arange(x.shape[0], dtype=np.int64)

    try:
        iter_range = range(1, int(cfg.max_iter) + 1)
        outer_pbar = tqdm(iter_range, desc="approx K-SVD", unit="iter") if verbose else iter_range
        for it in outer_pbar:
            d_prev = d.copy()
            n_ksvd = max(1, int(cfg.ksvd_iter))
            last_sub_ix: np.ndarray | None = None
            x_iter = x
            r_iter = np.zeros((0, int(cfg.n_atoms)), dtype=np.float32)

            atom_iter = range(n_ksvd)
            if verbose and n_ksvd > 1:
                atom_iter = tqdm(atom_iter, desc="online mini-batches", unit="batch", leave=False)
            for k_pass in atom_iter:
                # Every encode+update loop draws a new random mini-batch.
                x_iter, sub_ix = _draw_minibatch()
                last_sub_ix = sub_ix
                if verbose and hasattr(outer_pbar, "set_description"):
                    pct = 100.0 * float(cfg.subsample_fraction) if float(cfg.subsample_fraction) < 1.0 else 100.0
                    outer_pbar.set_description(
                        f"online K-SVD {it}/{cfg.max_iter} batch {k_pass + 1}/{n_ksvd} "
                        f"({x_iter.shape[0]:,}/{x.shape[0]:,} = {pct:.1f}%)"
                    )
                r_iter = sparse_codes(x_iter, d, cfg, show_progress=verbose and n_ksvd == 1)
                d, r_iter = approximate_ksvd_update_rows(
                    x_iter,
                    d,
                    r_iter,
                    active_eps=float(cfg.active_eps),
                    reinit_dead_atoms=bool(cfg.reinit_dead_atoms),
                    max_abs_code=float(cfg.max_abs_code),
                    nonnegative_codes=_nonnegative_sparse_codes(cfg),
                )
                d = normalize_rows(d).astype(np.float32, copy=False)

            atoms_pruned = 0
            if bool(cfg.prune_enabled) and int(cfg.prune_every_iters) > 0 and it % int(cfg.prune_every_iters) == 0:
                # Prune on a fresh mini-batch so dead-atom replacement is not stuck on one draw.
                x_iter, sub_ix = _draw_minibatch()
                last_sub_ix = sub_ix
                r_iter = sparse_codes(x_iter, d, cfg, show_progress=verbose)
                d, r_iter, atoms_pruned, prune_info = prune_and_reinit_atoms(
                    x_iter, d, r_iter, cfg, rng=rng
                )
                d = normalize_rows(d).astype(np.float32, copy=False)
                if atoms_pruned > 0:
                    r_iter = sparse_codes(x_iter, d, cfg, show_progress=verbose)
                    if verbose:
                        tqdm.write(
                            f"  prune @ iter {it}: replaced {atoms_pruned} atoms | "
                            f"mse {prune_info['mse_before']:.6f} -> {prune_info['mse_after']:.6f} "
                            f"(pre-reencode)"
                        )

            stats = reconstruction_stats(x_iter, d, r_iter, active_eps=float(cfg.active_eps), cfg=cfg)
            d = normalize_rows(d).astype(np.float32, copy=False)
            r_iter = _sanitize_matrix(r_iter, clip_abs=float(cfg.max_abs_code))
            dict_change = float(np.linalg.norm(d - d_prev) / max(np.linalg.norm(d_prev), 1e-12))
            row: dict[str, Any] = {
                "iteration": it,
                "samples": int(x_iter.shape[0]),
                "subsample_fraction": float(cfg.subsample_fraction),
                "ksvd_batches": int(n_ksvd),
                "minibatch_first_idx": int(last_sub_ix[0]) if last_sub_ix is not None and last_sub_ix.size else -1,
                "dict_change": dict_change,
                "atoms_pruned": int(atoms_pruned),
                **stats,
            }
            history.append(row)

            if verbose:
                if hasattr(outer_pbar, "set_postfix"):
                    outer_pbar.set_postfix(
                        mse=f"{stats['mse']:.4f}",
                        active=f"{stats['mean_active']:.1f}",
                        dead=int(stats["dead_atoms"]),
                        d_chg=f"{dict_change:.2e}",
                        refresh=True,
                    )
                tqdm.write(
                    f"iter {it:03d}/{cfg.max_iter} | batches={n_ksvd} | "
                    f"mse={stats['mse']:.6f} | "
                    f"active={stats['mean_active']:.1f} | sparsity={stats['sparsity']:.3f} | "
                    f"dead={stats['dead_atoms']} | under={stats.get('underperforming_atoms', 0)} | "
                    f"min_sup={stats.get('min_atom_support', 0)} | d_change={dict_change:.3e}"
                )

            if checkpoint_dir is not None and int(checkpoint_every_iters) > 0 and it % int(checkpoint_every_iters) == 0:
                save_checkpoint("iteration", it)

            if dict_change < float(cfg.min_dict_change):
                stall += 1
                stall_needed = int(cfg.patience)
                if float(cfg.subsample_fraction) < 1.0:
                    stall_needed = max(stall_needed, int(cfg.patience) * 2)
                if stall >= stall_needed:
                    if verbose:
                        tqdm.write(
                            f"early stop at iter {it}: dict_change < {cfg.min_dict_change} "
                            f"for {stall_needed} consecutive mini-batches"
                        )
                    save_checkpoint("early_stop", it)
                    break
            else:
                stall = 0
    except KeyboardInterrupt:
        save_checkpoint("keyboard_interrupt", history[-1]["iteration"] if history else 0)
        raise

    return d, history


def encode_samples(
    x: np.ndarray,
    d: np.ndarray,
    cfg: ClassicalKsvdConfig,
    *,
    refine: bool | None = None,
    show_progress: bool = False,
) -> np.ndarray:
    """X (n_samples, features), D (n_atoms, features) -> R (n_samples, n_atoms)."""
    x = _sanitize_matrix(x, clip_abs=float(cfg.max_abs_code))
    if x.ndim != 2:
        raise ValueError(f"x must be 2-D, got {x.shape}")
    d = normalize_rows(d).astype(np.float32, copy=False)
    r = sparse_codes(x, d, cfg, show_progress=show_progress)
    do_refine = bool(cfg.refine_r_iters > 0) if refine is None else bool(refine)
    if do_refine:
        r = refine_codes_fixed_dictionary(x, d, r, cfg)
    return _sanitize_matrix(r, clip_abs=float(cfg.max_abs_code))


def atom_usage_dataframe(
    r: np.ndarray,
    cfg: ClassicalKsvdConfig,
) -> pd.DataFrame:
    r = _sanitize_matrix(r, clip_abs=float(cfg.max_abs_code))
    metrics = atom_usage_metrics(
        r,
        active_eps=float(cfg.active_eps),
        nonnegative=_nonnegative_sparse_codes(cfg),
    )
    under = identify_underperforming_atoms(r, cfg)
    under_set = set(int(i) for i in under.tolist())
    return pd.DataFrame(
        {
            "atom": np.arange(r.shape[1], dtype=np.int32),
            "support_count": metrics["support"],
            "concentration_top5": metrics["concentration_top5"],
            "mean_abs_code": metrics["mean_abs_code"],
            "max_abs_code": metrics["max_abs_code"],
            "underperforming": [int(a in under_set) for a in range(r.shape[1])],
        }
    )


# Back-compat aliases (column-atom layout used briefly)
def fit_classical_ksvd(y_train: np.ndarray, cfg: ClassicalKsvdConfig, **kwargs) -> tuple[np.ndarray, list]:
    d, hist = fit_approx_dictionary(y_train.T, cfg, verbose=kwargs.get("verbose", True))
    return d.T, hist


def sparse_code_matrix(y: np.ndarray, d_cols: np.ndarray, cfg: ClassicalKsvdConfig) -> np.ndarray:
    return sparse_codes(y.T, d_cols.T, cfg).T


def history_to_dataframe(history: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(history)
