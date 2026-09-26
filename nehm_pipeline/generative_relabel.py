"""
Neighborhood-aware generative relabeling for the NEHM corpus.

For each row of ``Material_Database.xlsx`` we:

1. Find the top-K nearest neighbors in a chosen embedding space (student Z by default).
2. Compute Gaussian-kernel weights ``w_k`` from L2 distances; build a Dirichlet-smoothed
   posterior over selected categorical fields (logitb, Class, Subclass, …).
3. Render a structured prompt that contains the target row's metadata + each neighbor's
   metadata (esp. ``microscopy_description`` and ``Extended_description``), labeled with
   the kernel weight, plus the categorical posteriors.
4. Call an LLM (HKU gateway by default) with strict JSON output and append the result to
   a JSONL file. The file is incremental and resumable.
5. Optionally merge the JSONL columns back into a copy of the spreadsheet.

The defaults match ``distiller_CLIP_ViT_inference_embeddings_student.ipynb`` exactly
(``alpha0=0.55``, ``TAU_FRAC=0.5``, ``NEIGHBOR_TEMP=1.0``).

Usage from a notebook
---------------------
>>> from nehm_pipeline.generative_relabel import RelabelConfig, run_relabel
>>> cfg = RelabelConfig(
...     project_root=Path("/Users/marc/Desktop/NEHM_Project"),
...     student_z_npz=Path(".../inference_student.npz"),
...     output_jsonl=Path(".../relabel.jsonl"),
...     llm_engine="gpt-5.5",
...     row_filter="all",
... )
>>> run_relabel(cfg)
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Literal, Optional

import numpy as np
import pandas as pd

from nehm_pipeline.llm_gateway import HKUGateway, HKUGatewayError

__all__ = [
    "RelabelConfig",
    "RelabelResult",
    "run_relabel",
    "preview_relabel",
    "merge_jsonl_into_xlsx",
]

_LOG = logging.getLogger(__name__)


@dataclass
class RelabelConfig:
    project_root: Path

    # Excel
    xlsx_path: Optional[Path] = None
    sheet_name: str = "Sheet2"
    image_index_col: str = "Image Index"

    # Identity-level fields: things that ARE the answer to "what is this?". These are pulled
    # from neighbors as evidence and hidden from the target when assume_unknown_query=True.
    field_columns: tuple[str, ...] = (
        "Source",
        "Specimen Name",
        "Illumination Modality",
        "Class",
        "Subclass",
        "Description",
        "Stain",
        "Magnification",
        "composition",
        "refractive index",
        "Birefringence",
        "Optical Sign",
        "microscopy description",
        "Extended description",
    )
    bayes_field_columns: tuple[str, ...] = ("logitb", "Class", "Subclass")

    # When True, treat the target as an UNKNOWN specimen submitted for identification.
    # The target block only exposes capture-level metadata (microscope settings you'd know
    # about your own image) — Class/Subclass/descriptions/composition are HIDDEN from the
    # target and only shown for neighbors.
    assume_unknown_query: bool = True
    # Capture-level fields kept in the target block when assume_unknown_query=True.
    # These are things a microscopist knows about their own image regardless of identification.
    target_capture_fields: tuple[str, ...] = (
        "Illumination Modality",
        "Stain",
        "Magnification",
    )

    # Embeddings: pass either an in-memory array OR a path (.npy or .npz). When both are
    # provided, the in-memory array wins. .npy is a single matrix; .npz uses ``student_z_key``.
    student_z_array: Optional[np.ndarray] = None
    student_z_npz: Optional[Path] = None
    student_z_key: str = "Z"
    longclip_image_npy: Optional[Path] = None
    longclip_text_micro_npy: Optional[Path] = None
    longclip_text_ext_npy: Optional[Path] = None

    retrieval_space: Literal["student_z", "longclip_image", "longclip_text", "ensemble"] = "student_z"

    # Bayesian neighborhood
    k_neighbors: int = 10
    alpha0: float = 0.55
    tau_frac: float = 0.5
    neighbor_temp: float = 1.0

    # Text trimming (token control)
    max_target_chars_per_field: int = 2000
    max_neighbor_chars_per_field: int = 500
    neighbor_text_fields: tuple[str, ...] = ("microscopy description", "Extended description", "Description")

    # LLM
    llm_engine: str = "gpt-5.5"
    llm_api_key_env: str = "HKU_API_KEY"
    llm_api_base: str = "https://api.hku.hk"
    llm_api_version: str = "2024-06-01"
    temperature: float = 0.2
    max_output_tokens: int = 1500

    # Throughput / robustness
    concurrency: int = 4
    min_request_interval_s: float = 0.4
    max_retries: int = 6
    timeout_s: float = 120.0

    # Scope
    row_filter: Literal["all", "labeled_only", "unlabeled_only", "low_confidence_only"] = "all"
    low_confidence_threshold: float = 0.6
    student_predictions_csv: Optional[Path] = None
    only_image_indices: Optional[Iterable[int]] = None
    max_rows: Optional[int] = None
    dry_run_first_n: int = 3

    # Output
    output_jsonl: Optional[Path] = None
    audit_dir: Optional[Path] = None

    def resolve(self) -> "RelabelConfig":
        """Fill in defaults that depend on ``project_root``."""
        pr = Path(self.project_root).resolve()
        if self.xlsx_path is None:
            self.xlsx_path = pr / "Database" / "Material_Database.xlsx"
        if self.audit_dir is None:
            self.audit_dir = pr / "NEHM_RESULTS" / "generative_relabel"
        if self.output_jsonl is None:
            self.audit_dir.mkdir(parents=True, exist_ok=True)
            self.output_jsonl = self.audit_dir / "relabel.jsonl"
        else:
            Path(self.output_jsonl).parent.mkdir(parents=True, exist_ok=True)
        return self


@dataclass
class RelabelResult:
    n_rows_processed: int
    n_rows_skipped: int
    n_rows_failed: int
    output_jsonl: Path
    usage: dict
    elapsed_s: float


_SYSTEM_PROMPT_UNKNOWN_QUERY = (
    "You are a meticulous identification engine for a microscopy reference database.\n"
    "You receive an UNKNOWN specimen — only its capture metadata (microscope settings) and "
    "a neighborhood of visually-similar specimens drawn from a fine-tuned image-text manifold "
    "are provided. Each neighbor carries a kernel weight w in [0,1]; treat higher weights as "
    "stronger evidence.\n\n"
    "Your job is to PROPOSE the most likely identification of the unknown specimen using ONLY "
    "the neighborhood evidence and the kernel-weighted posteriors. You may not assume any "
    "metadata about the unknown specimen beyond the capture settings explicitly shown.\n\n"
    "RULES:\n"
    "- Treat the target as unknown. Do not invent attributes that are not supported by neighbors.\n"
    "- If the top neighbors disagree, lower confidence and document the disagreement.\n"
    "- The `microscopy_signature` paragraph must read like a polished catalog entry that "
    "  someone could use to identify the same kind of specimen later. Stay concrete; cite "
    "  optical features, habit, color, magnification, illumination modality, etc., as "
    "  reported by the neighborhood.\n"
    "- `agreement_with_logitb` describes consistency among the top neighbors' logitb values, "
    "  not consistency with any pre-existing label of the unknown.\n"
    "- Output STRICT JSON conforming exactly to the schema below. No prose outside the JSON."
)

_SYSTEM_PROMPT_KNOWN_TARGET = (
    "You are a meticulous specimen-cataloguer for a microscopy reference database.\n"
    "You synthesize one specimen's metadata with neighborhood evidence drawn from the "
    "nearest visually-similar specimens in a fine-tuned image-text manifold. Each neighbor "
    "carries a kernel weight w in [0,1]; treat higher weights as stronger evidence and "
    "lower weights as weaker corroboration.\n\n"
    "RULES:\n"
    "- Do not invent provenance, ages, locations, or chemistry that are not in the evidence.\n"
    "- If neighbors disagree with the target's current classification, say so in "
    "  `disagreement_notes` and lower `confidence_0_1` accordingly.\n"
    "- The `microscopy_signature` paragraph must read like a polished catalog entry that "
    "  someone could use to identify a similar specimen later. Stay concrete; cite optical "
    "  features, habit, color, magnification, illumination modality, etc.\n"
    "- Output STRICT JSON conforming exactly to the schema below. No prose outside the JSON."
)

# Back-compat alias (older callers).
_SYSTEM_PROMPT = _SYSTEM_PROMPT_UNKNOWN_QUERY

_JSON_SCHEMA_TEXT = """\
Required JSON schema (return EXACTLY these keys, no additional top-level keys):
{
  "refined_class_hypothesis": "<string: best single label combining target + neighbors>",
  "agreement_with_logitb":    "<one of: high | medium | low | disagree>",
  "confidence_0_1":           <number in [0,1]>,
  "microscopy_signature":     "<string, <= 1000 chars; polished catalog paragraph>",
  "key_features":             ["<string>", "..."],
  "disagreement_notes":       "<string, '' if no meaningful disagreement>",
  "evidence_rows_used":       [<int Image Index>, ...],
  "uncertain_attributes":     ["<string>", "..."]
}
"""


def _kernel_weights(dists: np.ndarray, *, tau_frac: float, neighbor_temp: float) -> np.ndarray:
    d = np.asarray(dists, dtype=np.float64)
    pos = d[d > 1e-12]
    med = float(np.median(pos)) if pos.size else 1.0
    tau = max(med * float(tau_frac), 1e-9)
    t = max(float(neighbor_temp), 1e-9)
    z = -((d / tau) ** 2) / t
    if z.size:
        z -= float(np.max(z))
    w = np.exp(z)
    s = float(np.sum(w))
    return w / s if s > 0 else w


def _l2_normalize_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, 1e-12)


def _build_index(matrix: np.ndarray):
    import faiss

    xb = np.ascontiguousarray(np.asarray(matrix, dtype=np.float32))
    idx = faiss.IndexFlatL2(xb.shape[1])
    idx.add(xb)
    return idx, xb


def _load_retrieval_matrix(cfg: RelabelConfig, n_expected: int) -> np.ndarray:
    if cfg.retrieval_space == "student_z":
        if cfg.student_z_array is not None:
            z = np.asarray(cfg.student_z_array, dtype=np.float32)
        elif cfg.student_z_npz is not None and Path(cfg.student_z_npz).is_file():
            p = Path(cfg.student_z_npz)
            if p.suffix.lower() == ".npy":
                z = np.asarray(np.load(p), dtype=np.float32)
            else:
                pack = np.load(p, allow_pickle=False)
                if cfg.student_z_key not in pack.files:
                    raise KeyError(f"npz {p} missing array '{cfg.student_z_key}'; keys={pack.files}")
                z = np.asarray(pack[cfg.student_z_key], dtype=np.float32)
        else:
            raise FileNotFoundError(
                "retrieval_space='student_z' requires student_z_array (in-memory) or "
                f"student_z_npz (.npy or .npz on disk); got {cfg.student_z_npz}"
            )
    elif cfg.retrieval_space == "longclip_image":
        if cfg.longclip_image_npy is None or not Path(cfg.longclip_image_npy).is_file():
            raise FileNotFoundError(f"longclip_image_npy required; got {cfg.longclip_image_npy}")
        z = np.asarray(np.load(cfg.longclip_image_npy), dtype=np.float32)
    elif cfg.retrieval_space == "longclip_text":
        parts = []
        if cfg.longclip_text_micro_npy is not None and Path(cfg.longclip_text_micro_npy).is_file():
            parts.append(np.asarray(np.load(cfg.longclip_text_micro_npy), dtype=np.float32))
        if cfg.longclip_text_ext_npy is not None and Path(cfg.longclip_text_ext_npy).is_file():
            parts.append(np.asarray(np.load(cfg.longclip_text_ext_npy), dtype=np.float32))
        if not parts:
            raise FileNotFoundError(
                "retrieval_space='longclip_text' requires at least one of "
                "longclip_text_micro_npy / longclip_text_ext_npy"
            )
        z = np.concatenate(parts, axis=1)
    else:
        raise ValueError(f"unsupported retrieval_space: {cfg.retrieval_space}")

    if z.shape[0] != n_expected:
        raise ValueError(
            f"retrieval matrix rows {z.shape[0]} != spreadsheet rows {n_expected}"
        )
    return _l2_normalize_rows(z)


def _ensemble_neighbors(
    rows_a: tuple[np.ndarray, np.ndarray],
    rows_b: tuple[np.ndarray, np.ndarray],
    *,
    k: int,
    rrf_const: float = 60.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Reciprocal-rank fusion of two (idx, dist) lists; returns merged top-k."""
    score: dict[int, float] = {}
    dist_lookup: dict[int, float] = {}
    for ranks_idx, dists in (rows_a, rows_b):
        for rank, (i, d) in enumerate(zip(ranks_idx, dists)):
            score[int(i)] = score.get(int(i), 0.0) + 1.0 / (rrf_const + rank)
            dist_lookup.setdefault(int(i), float(d))
    ordered = sorted(score.items(), key=lambda kv: -kv[1])[:k]
    out_ix = np.array([i for i, _ in ordered], dtype=np.int64)
    out_d = np.array([dist_lookup[i] for i, _ in ordered], dtype=np.float64)
    return out_ix, out_d


def _bayesian_posterior(
    df: pd.DataFrame,
    column: str,
    *,
    neighbor_row_indices: np.ndarray,
    weights: np.ndarray,
    alpha0: float,
) -> dict[str, float]:
    vocab = {}
    for _, v in df[column].items():
        if pd.isna(v):
            continue
        s = str(v).strip()
        if s:
            vocab[s] = vocab.get(s, 0) + 1
    soft: dict[str, float] = {}
    for ridx, w in zip(neighbor_row_indices, weights):
        v = df.at[int(ridx), column] if column in df.columns else None
        if pd.isna(v):
            continue
        s = str(v).strip()
        if not s:
            continue
        soft[s] = soft.get(s, 0.0) + float(w)
    K = max(len(vocab), 1)
    a = float(alpha0) / K
    denom = float(alpha0) + sum(soft.values())
    if denom <= 0:
        return {}
    return {c: (a + soft.get(c, 0.0)) / denom for c in vocab}


def _truncate(s: object, n: int) -> str:
    if pd.isna(s):
        return ""
    text = str(s).strip()
    if len(text) <= n:
        return text
    return text[: max(0, n - 1)].rstrip() + "…"


def _format_target_block(row: pd.Series, *, fields: tuple[str, ...], cap: int) -> str:
    lines = []
    for col in fields:
        if col not in row.index:
            continue
        v = _truncate(row[col], cap)
        if v:
            lines.append(f"  {col}: {v}")
    return "\n".join(lines) if lines else "  (no metadata available — identity to be inferred from neighborhood)"


def _format_neighbor_block(
    df: pd.DataFrame,
    *,
    neighbor_row_indices: np.ndarray,
    weights: np.ndarray,
    fields: tuple[str, ...],
    text_fields: tuple[str, ...],
    cap_text: int,
    cap_field: int,
    image_index_col: str,
) -> str:
    out_lines = []
    for rank, (ridx, w) in enumerate(zip(neighbor_row_indices, weights), 1):
        ridx = int(ridx)
        ii = int(df.at[ridx, image_index_col]) if image_index_col in df.columns else ridx
        header = f"[#{rank} w={float(w):.3f} ImageIndex={ii}]"
        meta_bits = []
        for c in fields:
            if c in text_fields or c not in df.columns:
                continue
            v = _truncate(df.at[ridx, c], cap_field)
            if v:
                meta_bits.append(f"{c}={v}")
        text_bits = []
        for c in text_fields:
            if c not in df.columns:
                continue
            v = _truncate(df.at[ridx, c], cap_text)
            if v:
                text_bits.append(f"{c}: {v}")
        block = header
        if meta_bits:
            block += "\n  " + " | ".join(meta_bits)
        for tb in text_bits:
            block += "\n  " + tb
        out_lines.append(block)
    return "\n".join(out_lines) if out_lines else "  (no neighbors)"


def _format_posterior_block(posteriors: dict[str, dict[str, float]], *, top: int = 3) -> str:
    out = []
    for col, post in posteriors.items():
        if not post:
            continue
        ranked = sorted(post.items(), key=lambda kv: -kv[1])[:top]
        items = [f"{name} (p={p:.3f})" for name, p in ranked]
        out.append(f"  {col}: " + " | ".join(items))
    return "\n".join(out) if out else "  (no posteriors)"


def _build_user_prompt(
    *,
    target_image_index: int,
    target_block: str,
    posterior_block: str,
    neighbor_block: str,
    schema_text: str,
    assume_unknown_query: bool,
) -> str:
    if assume_unknown_query:
        target_header = (
            f"UNKNOWN SPECIMEN (Image Index = {target_image_index}; identity to be inferred)\n"
            f"  Capture metadata (only what the user knows about their own image):\n"
            f"{target_block}\n\n"
        )
        task = (
            f"TASK\n"
            f"Identify the unknown specimen using ONLY the neighborhood evidence above and "
            f"the kernel-weighted posteriors. The unknown specimen has no pre-existing label; "
            f"do not assume any property of it beyond the capture metadata listed. Cite which "
            f"neighbor Image Indices you relied on in `evidence_rows_used`.\n\n"
        )
    else:
        target_header = (
            f"TARGET SAMPLE (Image Index = {target_image_index})\n"
            f"{target_block}\n\n"
        )
        task = (
            f"TASK\n"
            f"Synthesize a refined specimen label that integrates the target's metadata with "
            f"the neighborhood evidence weighted by the kernel weights. Note disagreements "
            f"explicitly. Cite which neighbor Image Indices you relied on in "
            f"`evidence_rows_used`.\n\n"
        )
    return (
        f"{target_header}"
        f"NEIGHBOR-INFORMED POSTERIORS (top-3 per field; weights are kernel-weighted)\n"
        f"{posterior_block}\n\n"
        f"NEIGHBORHOOD EVIDENCE (top-K nearest in the active embedding space)\n"
        f"{neighbor_block}\n\n"
        f"{task}"
        f"{schema_text}"
    )


def _hash_prompt(system: str, user: str) -> str:
    h = hashlib.sha256()
    h.update(system.encode("utf-8"))
    h.update(b"\x00")
    h.update(user.encode("utf-8"))
    return h.hexdigest()


def _read_existing_jsonl(path: Path) -> dict[int, str]:
    if not path.is_file():
        return {}
    out: dict[int, str] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            ii = rec.get("image_index")
            ph = rec.get("prompt_hash")
            if ii is not None and ph:
                out[int(ii)] = str(ph)
    return out


def _select_target_rows(df: pd.DataFrame, cfg: RelabelConfig) -> np.ndarray:
    n = len(df)
    mask = np.ones(n, dtype=bool)
    if cfg.row_filter == "labeled_only" and "logitb" in df.columns:
        mask &= df["logitb"].fillna(-1).astype(int).to_numpy() >= 0
    elif cfg.row_filter == "unlabeled_only" and "logitb" in df.columns:
        mask &= df["logitb"].fillna(-1).astype(int).to_numpy() < 0
    elif cfg.row_filter == "low_confidence_only":
        if cfg.student_predictions_csv is None or not Path(cfg.student_predictions_csv).is_file():
            raise FileNotFoundError(
                "row_filter='low_confidence_only' requires student_predictions_csv pointing at "
                "the inference run's predictions CSV with a 'top1_prob' column."
            )
        preds = pd.read_csv(cfg.student_predictions_csv)
        prob_col = next(
            (c for c in ("top1_prob", "y_pred_prob", "pred_prob", "p_top1") if c in preds.columns),
            None,
        )
        if prob_col is None:
            raise ValueError(
                "student_predictions_csv must contain one of: top1_prob / y_pred_prob / pred_prob / p_top1"
            )
        # Try to align by Image Index if present, else by row order.
        ii_lower_map = {c.lower(): c for c in preds.columns}
        ii_alias = ii_lower_map.get(cfg.image_index_col.lower()) or ii_lower_map.get("image_index")
        if ii_alias is not None and cfg.image_index_col in df.columns:
            preds = preds.set_index(ii_alias)
            probs = preds.reindex(df[cfg.image_index_col].astype(int).values)[prob_col].to_numpy()
        else:
            probs = preds[prob_col].to_numpy()
        mask &= np.asarray(probs, dtype=np.float64) < float(cfg.low_confidence_threshold)
    if cfg.only_image_indices is not None and cfg.image_index_col in df.columns:
        wanted = set(int(x) for x in cfg.only_image_indices)
        mask &= df[cfg.image_index_col].astype(int).isin(wanted).to_numpy()
    rows = np.flatnonzero(mask)
    if cfg.max_rows is not None:
        rows = rows[: int(cfg.max_rows)]
    return rows


def run_relabel(cfg: RelabelConfig) -> RelabelResult:
    cfg.resolve()
    if not Path(cfg.xlsx_path).is_file():
        raise FileNotFoundError(f"xlsx_path not found: {cfg.xlsx_path}")
    df = pd.read_excel(cfg.xlsx_path, sheet_name=cfg.sheet_name)
    if cfg.image_index_col in df.columns:
        df = df.sort_values(cfg.image_index_col).reset_index(drop=True)
    n = len(df)

    matrix_a = _load_retrieval_matrix(cfg, n)
    index_a, xb_a = _build_index(matrix_a)
    _LOG.info("FAISS index built: D=%d N=%d (%s)", xb_a.shape[1], xb_a.shape[0], cfg.retrieval_space)

    use_ensemble = cfg.retrieval_space == "ensemble"
    matrix_b = index_b = xb_b = None
    if use_ensemble:
        text_cfg = RelabelConfig(**{**cfg.__dict__, "retrieval_space": "longclip_text"})
        matrix_b = _load_retrieval_matrix(text_cfg, n)
        index_b, xb_b = _build_index(matrix_b)

    target_rows = _select_target_rows(df, cfg)
    _LOG.info("Selected %d / %d rows for relabeling (filter=%s)", len(target_rows), n, cfg.row_filter)

    seen_hashes = _read_existing_jsonl(cfg.output_jsonl)
    if seen_hashes:
        _LOG.info("Resuming: %d rows already in %s", len(seen_hashes), cfg.output_jsonl)

    gateway = HKUGateway(
        engine=cfg.llm_engine,
        api_base=cfg.llm_api_base,
        api_version=cfg.llm_api_version,
        api_key_env=cfg.llm_api_key_env,
        timeout_s=cfg.timeout_s,
        max_retries=cfg.max_retries,
        min_request_interval_s=cfg.min_request_interval_s,
    )

    write_lock = threading.Lock()
    out_path = Path(cfg.output_jsonl)
    err_path = out_path.with_suffix(".errors.jsonl")

    n_done = n_skip = n_fail = 0
    t0 = time.time()

    def _process_row(row_idx: int) -> tuple[int, str]:
        row = df.iloc[row_idx]
        ii_val = row.get(cfg.image_index_col, row_idx)
        ii = int(ii_val) if not pd.isna(ii_val) else int(row_idx)

        k_search = max(int(cfg.k_neighbors) + 2, 5)
        xq_a = xb_a[row_idx : row_idx + 1]
        d2_a, idx_a = index_a.search(xq_a, k_search)
        idx_a, d2_a = idx_a[0], d2_a[0]
        keep = idx_a != row_idx
        idx_a, d2_a = idx_a[keep][: cfg.k_neighbors], d2_a[keep][: cfg.k_neighbors]

        if use_ensemble and index_b is not None:
            xq_b = xb_b[row_idx : row_idx + 1]
            d2_b, idx_b = index_b.search(xq_b, k_search)
            idx_b, d2_b = idx_b[0], d2_b[0]
            keep_b = idx_b != row_idx
            idx_b, d2_b = idx_b[keep_b][: cfg.k_neighbors], d2_b[keep_b][: cfg.k_neighbors]
            neigh_ix, neigh_d = _ensemble_neighbors((idx_a, d2_a), (idx_b, d2_b), k=cfg.k_neighbors)
        else:
            neigh_ix, neigh_d = idx_a, d2_a

        order = np.argsort(neigh_d)
        neigh_ix = neigh_ix[order]
        neigh_d = neigh_d[order]
        weights = _kernel_weights(neigh_d, tau_frac=cfg.tau_frac, neighbor_temp=cfg.neighbor_temp)

        posteriors: dict[str, dict[str, float]] = {}
        for col in cfg.bayes_field_columns:
            if col in df.columns:
                posteriors[col] = _bayesian_posterior(
                    df, col,
                    neighbor_row_indices=neigh_ix,
                    weights=weights,
                    alpha0=cfg.alpha0,
                )

        target_fields = cfg.target_capture_fields if cfg.assume_unknown_query else cfg.field_columns
        target_block = _format_target_block(
            row, fields=target_fields, cap=cfg.max_target_chars_per_field
        )
        posterior_block = _format_posterior_block(posteriors)
        neighbor_block = _format_neighbor_block(
            df,
            neighbor_row_indices=neigh_ix,
            weights=weights,
            fields=cfg.field_columns,
            text_fields=cfg.neighbor_text_fields,
            cap_text=cfg.max_neighbor_chars_per_field,
            cap_field=120,
            image_index_col=cfg.image_index_col,
        )
        user_prompt = _build_user_prompt(
            target_image_index=ii,
            target_block=target_block,
            posterior_block=posterior_block,
            neighbor_block=neighbor_block,
            schema_text=_JSON_SCHEMA_TEXT,
            assume_unknown_query=cfg.assume_unknown_query,
        )
        sys_prompt = (
            _SYSTEM_PROMPT_UNKNOWN_QUERY if cfg.assume_unknown_query else _SYSTEM_PROMPT_KNOWN_TARGET
        )
        prompt_hash = _hash_prompt(sys_prompt, user_prompt)

        if seen_hashes.get(ii) == prompt_hash:
            return ii, "skip"

        try:
            parsed = gateway.chat_json(
                system=sys_prompt,
                user=user_prompt,
                temperature=cfg.temperature,
                max_tokens=cfg.max_output_tokens,
            )
        except HKUGatewayError as exc:
            with write_lock:
                with err_path.open("a", encoding="utf-8") as ef:
                    ef.write(json.dumps({
                        "image_index": ii,
                        "row_idx": int(row_idx),
                        "error": str(exc),
                        "prompt_hash": prompt_hash,
                        "ts_utc": datetime.now(timezone.utc).isoformat(),
                    }) + "\n")
            return ii, "fail"

        record = {
            "image_index": ii,
            "row_idx": int(row_idx),
            "prompt_hash": prompt_hash,
            "model": cfg.llm_engine,
            "ts_utc": datetime.now(timezone.utc).isoformat(),
            "k_neighbors": int(cfg.k_neighbors),
            "retrieval_space": cfg.retrieval_space,
            "neighbor_image_indices": [
                int(df.at[int(r), cfg.image_index_col]) if cfg.image_index_col in df.columns else int(r)
                for r in neigh_ix
            ],
            "neighbor_weights": [float(w) for w in weights],
            "posteriors_top3": {
                col: sorted(post.items(), key=lambda kv: -kv[1])[:3]
                for col, post in posteriors.items()
            },
            "response": parsed,
        }
        with write_lock:
            with out_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return ii, "ok"

    if cfg.dry_run_first_n > 0 and target_rows.size > 0:
        head = list(target_rows[: min(int(cfg.dry_run_first_n), len(target_rows))])
        _LOG.info("Running dry preview on %d row(s) before fan-out…", len(head))
        for r in head:
            ii, status = _process_row(int(r))
            _LOG.info("  preview row %s -> Image Index %d : %s", r, ii, status)
            if status == "ok":
                n_done += 1
            elif status == "skip":
                n_skip += 1
            else:
                n_fail += 1
        target_rows = target_rows[len(head):]

    if target_rows.size == 0:
        _LOG.info("No rows remaining after preview / filter.")
    else:
        with ThreadPoolExecutor(max_workers=max(1, int(cfg.concurrency))) as pool:
            futs = {pool.submit(_process_row, int(r)): int(r) for r in target_rows}
            for i_done, fut in enumerate(as_completed(futs), 1):
                try:
                    ii, status = fut.result()
                except Exception as exc:
                    n_fail += 1
                    _LOG.exception("Worker raised: %s", exc)
                    continue
                if status == "ok":
                    n_done += 1
                elif status == "skip":
                    n_skip += 1
                else:
                    n_fail += 1
                if i_done % 25 == 0 or i_done == len(futs):
                    u = gateway.usage.snapshot()
                    elapsed = time.time() - t0
                    _LOG.info(
                        "progress %d/%d  ok=%d skip=%d fail=%d  tokens(prompt/completion/total)=%d/%d/%d  retries=%d  rate=%.2f rows/s",
                        i_done, len(futs), n_done, n_skip, n_fail,
                        u["prompt_tokens"], u["completion_tokens"], u["total_tokens"],
                        u["retry_count"], i_done / max(elapsed, 1e-9),
                    )

    return RelabelResult(
        n_rows_processed=n_done,
        n_rows_skipped=n_skip,
        n_rows_failed=n_fail,
        output_jsonl=out_path,
        usage=gateway.usage.snapshot(),
        elapsed_s=time.time() - t0,
    )


def preview_relabel(cfg: RelabelConfig, *, n_rows: int = 3, row_indices: Optional[Iterable[int]] = None) -> list[dict]:
    """
    Dry-run helper: build prompts + call the LLM for a few rows and return the results
    in memory. **Does not write anything to disk** (no JSONL append, no error log).

    Parameters
    ----------
    cfg : RelabelConfig
        Same config object used for ``run_relabel``. ``output_jsonl`` / ``audit_dir`` are
        ignored here.
    n_rows : int, optional
        Number of rows to preview. Used only when ``row_indices`` is None. The first
        ``n_rows`` rows passing the configured filter are previewed.
    row_indices : iterable of int, optional
        Explicit list of dataframe row indices (NOT ``Image Index`` values). Bypasses
        the configured filter when provided.

    Returns
    -------
    list of dict
        One dict per row with keys: ``image_index``, ``row_idx``, ``neighbor_image_indices``,
        ``neighbor_weights``, ``posteriors_top3``, ``system_prompt``, ``user_prompt``,
        ``response``, ``usage_after_call``.
    """
    cfg.resolve()
    df = pd.read_excel(cfg.xlsx_path, sheet_name=cfg.sheet_name)
    if cfg.image_index_col in df.columns:
        df = df.sort_values(cfg.image_index_col).reset_index(drop=True)

    matrix = _load_retrieval_matrix(cfg, len(df))
    index, xb = _build_index(matrix)

    if row_indices is not None:
        target_rows = np.asarray(list(row_indices), dtype=np.int64)
    else:
        target_rows = _select_target_rows(df, cfg)[: int(n_rows)]

    gateway = HKUGateway(
        engine=cfg.llm_engine,
        api_base=cfg.llm_api_base,
        api_version=cfg.llm_api_version,
        api_key_env=cfg.llm_api_key_env,
        timeout_s=cfg.timeout_s,
        max_retries=cfg.max_retries,
        min_request_interval_s=cfg.min_request_interval_s,
    )

    out: list[dict] = []
    for row_idx in target_rows:
        row_idx = int(row_idx)
        row = df.iloc[row_idx]
        ii_val = row.get(cfg.image_index_col, row_idx)
        ii = int(ii_val) if not pd.isna(ii_val) else int(row_idx)

        k_search = max(int(cfg.k_neighbors) + 2, 5)
        d2, idx = index.search(xb[row_idx : row_idx + 1], k_search)
        idx, d2 = idx[0], d2[0]
        keep = idx != row_idx
        neigh_ix = idx[keep][: cfg.k_neighbors]
        neigh_d = d2[keep][: cfg.k_neighbors]
        order = np.argsort(neigh_d)
        neigh_ix = neigh_ix[order]
        neigh_d = neigh_d[order]
        weights = _kernel_weights(neigh_d, tau_frac=cfg.tau_frac, neighbor_temp=cfg.neighbor_temp)

        posteriors: dict[str, dict[str, float]] = {}
        for col in cfg.bayes_field_columns:
            if col in df.columns:
                posteriors[col] = _bayesian_posterior(
                    df, col,
                    neighbor_row_indices=neigh_ix,
                    weights=weights,
                    alpha0=cfg.alpha0,
                )

        target_fields = cfg.target_capture_fields if cfg.assume_unknown_query else cfg.field_columns
        target_block = _format_target_block(
            row, fields=target_fields, cap=cfg.max_target_chars_per_field
        )
        posterior_block = _format_posterior_block(posteriors)
        neighbor_block = _format_neighbor_block(
            df,
            neighbor_row_indices=neigh_ix,
            weights=weights,
            fields=cfg.field_columns,
            text_fields=cfg.neighbor_text_fields,
            cap_text=cfg.max_neighbor_chars_per_field,
            cap_field=120,
            image_index_col=cfg.image_index_col,
        )
        user_prompt = _build_user_prompt(
            target_image_index=ii,
            target_block=target_block,
            posterior_block=posterior_block,
            neighbor_block=neighbor_block,
            schema_text=_JSON_SCHEMA_TEXT,
            assume_unknown_query=cfg.assume_unknown_query,
        )
        sys_prompt = (
            _SYSTEM_PROMPT_UNKNOWN_QUERY if cfg.assume_unknown_query else _SYSTEM_PROMPT_KNOWN_TARGET
        )

        try:
            parsed = gateway.chat_json(
                system=sys_prompt,
                user=user_prompt,
                temperature=cfg.temperature,
                max_tokens=cfg.max_output_tokens,
            )
        except HKUGatewayError as exc:
            parsed = {"_error": str(exc)}

        out.append({
            "image_index": ii,
            "row_idx": int(row_idx),
            "neighbor_image_indices": [
                int(df.at[int(r), cfg.image_index_col]) if cfg.image_index_col in df.columns else int(r)
                for r in neigh_ix
            ],
            "neighbor_weights": [float(w) for w in weights],
            "posteriors_top3": {
                col: sorted(post.items(), key=lambda kv: -kv[1])[:3]
                for col, post in posteriors.items()
            },
            "system_prompt": sys_prompt,
            "user_prompt": user_prompt,
            "response": parsed,
            "usage_after_call": gateway.usage.snapshot(),
        })
    return out


_REFINED_COLUMN_PREFIX = "refined_"
_REFINED_KEYS = (
    "refined_class_hypothesis",
    "agreement_with_logitb",
    "confidence_0_1",
    "microscopy_signature",
    "key_features",
    "disagreement_notes",
    "evidence_rows_used",
    "uncertain_attributes",
)


def merge_jsonl_into_xlsx(
    *,
    jsonl_path: Path,
    src_xlsx: Path,
    dst_xlsx: Path,
    sheet_name: str = "Sheet2",
    image_index_col: str = "Image Index",
) -> Path:
    """Merge the relabel JSONL into a copy of the spreadsheet."""
    df = pd.read_excel(src_xlsx, sheet_name=sheet_name)
    if image_index_col not in df.columns:
        raise ValueError(f"src xlsx missing '{image_index_col}'")

    by_ii: dict[int, dict] = {}
    with Path(jsonl_path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            ii = int(rec.get("image_index"))
            resp = rec.get("response", {}) or {}
            by_ii[ii] = {f"{_REFINED_COLUMN_PREFIX}{k}": resp.get(k) for k in _REFINED_KEYS}
            by_ii[ii][f"{_REFINED_COLUMN_PREFIX}model"] = rec.get("model")

    new_cols = {col: [] for col in next(iter(by_ii.values())).keys()} if by_ii else {}
    for col in new_cols:
        col_values: list = []
        for ii in df[image_index_col].astype(int).tolist():
            v = by_ii.get(int(ii), {}).get(col)
            if isinstance(v, (list, dict)):
                v = json.dumps(v, ensure_ascii=False)
            col_values.append(v)
        new_cols[col] = col_values

    out_df = df.copy()
    for col, vals in new_cols.items():
        out_df[col] = vals

    Path(dst_xlsx).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_excel(dst_xlsx, sheet_name=sheet_name, index=False)
    return Path(dst_xlsx)
