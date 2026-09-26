"""End-to-end inference for microscopy images outside the NEHM database."""

from __future__ import annotations

import hashlib
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
from PIL import Image

from nehm_pipeline.clip_vit_multiview_finetune import (
    batched_pooled_visual_ln_post_embeddings,
    multiview_crops_finetune,
)
from nehm_pipeline.clip_visual_student import (
    ClipVisualStudent,
    load_clip_visual_student_checkpoint,
)
from nehm_pipeline.device import clip_visual_preprocess_without_channel_normalize
from nehm_pipeline.student_clip import ImageEmbeddingStudent

from . import atom_interpretability as ai
from .classical_ksvd import load_classical_ksvd_config, normalize_rows

ProgressCallback = Callable[[str, int, str], None]


@dataclass(frozen=True)
class UploadArtifacts:
    project_root: Path
    database_xlsx: Path
    corpus_embeddings: Path
    corpus_codes: Path
    dictionary_atoms: Path
    embedding_mean: Path
    dictionary_config: Path
    atom_labels: Path
    student_checkpoint: Path
    longclip_root: Path
    longclip_checkpoint: Path

    @classmethod
    def from_project_root(cls, project_root: Path) -> "UploadArtifacts":
        root = Path(project_root).expanduser().resolve()
        embedding_dir = root / "NEHM_RESULTS" / "student_inference_B"
        current_image_embeddings = (
            embedding_dir / "student_z_2304_B_current_images.npy"
        )
        interp = (
            embedding_dir
            / "global_dictionary_interpretability"
        )
        return cls(
            project_root=root,
            database_xlsx=root / "Database" / "Material_Database.xlsx",
            corpus_embeddings=(
                current_image_embeddings
                if current_image_embeddings.is_file()
                else embedding_dir / "student_z_2304_B.npy"
            ),
            corpus_codes=interp / "global_sparse_codes.npy",
            dictionary_atoms=interp / "global_dictionary_atoms.npy",
            embedding_mean=interp / "global_embedding_mean.npy",
            dictionary_config=interp / "global_ksvd_config.json",
            atom_labels=interp / "atom_microscopy_labels.json",
            student_checkpoint=(
                root / "NEHM_RESULTS" / "vit_student_finetune_best_B_logitb.pth"
            ),
            longclip_root=root / "LongCLIP",
            longclip_checkpoint=root / "LongCLIP" / "checkpoints" / "longclip-B.pt",
        )

    def required_paths(self) -> list[Path]:
        return [
            self.database_xlsx,
            self.corpus_embeddings,
            self.corpus_codes,
            self.dictionary_atoms,
            self.embedding_mean,
            self.dictionary_config,
            self.atom_labels,
            self.student_checkpoint,
            self.longclip_checkpoint,
        ]


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if (
        getattr(torch.backends, "mps", None) is not None
        and torch.backends.mps.is_available()
    ):
        return torch.device("mps")
    return torch.device("cpu")


def _normalized_frequency(
    neighbors: list[dict[str, Any]],
    field: str,
) -> list[dict[str, Any]]:
    groups: dict[str, dict[str, Any]] = {}
    for rank, neighbor in enumerate(neighbors, start=1):
        value = ai.safe_text(neighbor.get(field, "")).strip()
        key = re.sub(r"\s+", "", value).casefold()
        if not key or key in {"unknown", "unspecified", "n/a", "na", "none", "-"}:
            continue
        group = groups.setdefault(
            key,
            {"value": value, "count": 0, "ranks": [], "d2": []},
        )
        group["count"] += 1
        group["ranks"].append(rank)
        group["d2"].append(float(neighbor["d2"]))
    ranked: list[dict[str, Any]] = []
    for group in groups.values():
        distances = group.pop("d2")
        group["mean_d2"] = float(sum(distances) / len(distances))
        ranked.append(group)
    return sorted(ranked, key=lambda item: (-int(item["count"]), item["mean_d2"]))


def _positive_elastic_net_fista(
    target: np.ndarray,
    dictionary: np.ndarray,
    *,
    alpha: float,
    l1_ratio: float,
    max_iter: int,
    tolerance: float = 1e-6,
) -> np.ndarray:
    """Single-row nonnegative Elastic Net without sklearn/OpenMP runtime coupling."""
    design = np.ascontiguousarray(
        normalize_rows(dictionary).T,
        dtype=np.float64,
    )
    y = np.asarray(target, dtype=np.float64).reshape(-1)
    n_features = float(design.shape[0])
    l1 = float(alpha) * float(l1_ratio)
    l2 = float(alpha) * (1.0 - float(l1_ratio))

    def forward(vector: np.ndarray) -> np.ndarray:
        return np.einsum("ij,j->i", design, vector, optimize=False)

    def transpose(vector: np.ndarray) -> np.ndarray:
        return np.einsum("ij,i->j", design, vector, optimize=False)

    # Power iteration estimates the smooth gradient Lipschitz constant.
    probe = np.full(design.shape[1], 1.0 / np.sqrt(design.shape[1]))
    for _ in range(30):
        probe_next = transpose(forward(probe)) / n_features + l2 * probe
        norm = float(np.linalg.norm(probe_next))
        if norm <= 1e-12:
            break
        probe = probe_next / norm
    lipschitz = float(
        probe @ (transpose(forward(probe)) / n_features + l2 * probe)
    )
    step = 1.0 / max(lipschitz, 1e-8)

    coefficients = np.zeros(design.shape[1], dtype=np.float64)
    accelerated = coefficients.copy()
    momentum = 1.0
    for _ in range(max(50, min(int(max_iter), 2000))):
        gradient = transpose(forward(accelerated) - y) / n_features + l2 * accelerated
        updated = np.maximum(accelerated - step * gradient - step * l1, 0.0)
        if np.linalg.norm(updated - coefficients) <= tolerance * max(
            1.0, float(np.linalg.norm(coefficients))
        ):
            coefficients = updated
            break
        next_momentum = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * momentum * momentum))
        accelerated = updated + ((momentum - 1.0) / next_momentum) * (
            updated - coefficients
        )
        coefficients = updated
        momentum = next_momentum
    return coefficients.astype(np.float32)


class PLMUploadInference:
    """Loads model/artifacts once and describes arbitrary uploaded images."""

    def __init__(
        self,
        artifacts: UploadArtifacts,
        *,
        device: torch.device | None = None,
        neighbor_count: int = 10,
        crop_seed: int = 42,
        identity_max_nearest_d2: float = 8.0,
        excluded_evidence_terms: tuple[str, ...] = ("vivianite",),
    ) -> None:
        self.artifacts = artifacts
        self.device = device or choose_device()
        self.neighbor_count = int(neighbor_count)
        self.crop_seed = int(crop_seed)
        self.identity_max_nearest_d2 = float(identity_max_nearest_d2)
        self.excluded_evidence_terms = tuple(
            term.strip().casefold()
            for term in excluded_evidence_terms
            if term.strip()
        )
        missing = [path for path in artifacts.required_paths() if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "Missing PLM web artifacts:\n  " + "\n  ".join(str(path) for path in missing)
            )

        self.z, self.df, _ = ai.load_aligned_corpus(
            z_path=artifacts.corpus_embeddings,
            xlsx_path=artifacts.database_xlsx,
        )
        self.codes = np.asarray(np.load(artifacts.corpus_codes), dtype=np.float32)
        self.dictionary = np.asarray(
            np.load(artifacts.dictionary_atoms), dtype=np.float32
        )
        self.mean = np.asarray(np.load(artifacts.embedding_mean), dtype=np.float32)
        self.cfg = load_classical_ksvd_config(artifacts.dictionary_config)
        self.labels = ai.load_atom_labels(artifacts.atom_labels)
        self.excluded_atom_ids = {
            atom_id
            for atom_id, record in self.labels.items()
            if self._contains_excluded_term(
                record.get("label", ""),
                record.get("microscopy_description", ""),
            )
        }
        self._load_student()

    def _contains_excluded_term(self, *values: Any) -> bool:
        text = " ".join(ai.safe_text(value) for value in values).casefold()
        return any(term in text for term in self.excluded_evidence_terms)

    def _load_student(self) -> None:
        longclip_root = str(self.artifacts.longclip_root)
        if longclip_root not in sys.path:
            sys.path.insert(0, longclip_root)
        from model import longclip  # type: ignore

        clip_model, preprocess_with_normalize = longclip.load(
            str(self.artifacts.longclip_checkpoint),
            device=str(self.device),
        )
        clip_model = clip_model.float().to(self.device)
        self.preprocess = clip_visual_preprocess_without_channel_normalize(
            preprocess_with_normalize
        )
        visual = clip_model.visual

        checkpoint = torch.load(
            self.artifacts.student_checkpoint,
            map_location="cpu",
        )
        meta = dict(checkpoint.get("meta", {}))
        n_classes = int(meta.get("n_material_classes") or 0)
        teacher_dim = int(meta.get("teacher_embed_dim") or 0)
        state_dict = checkpoint.get("state_dict") or {}
        if n_classes <= 0 or teacher_dim <= 0:
            for key, value in state_dict.items():
                if str(key).endswith("student.classifier.weight"):
                    n_classes = int(value.shape[0])
                    teacher_dim = int(value.shape[1])
                    break
        if n_classes <= 0 or teacher_dim != 2304:
            raise ValueError(
                f"Unexpected Pipeline B checkpoint dimensions: classes={n_classes}, "
                f"teacher_dim={teacher_dim}"
            )

        student = ImageEmbeddingStudent(
            image_dim=768,
            num_classes=n_classes,
            embed_dim=teacher_dim,
            decoder_hidden=(1024, 1536),
            dropout=0.0,
            head_style="decoder",
        )
        self.model = ClipVisualStudent(visual, student).to(self.device).eval()
        load_clip_visual_student_checkpoint(
            self.artifacts.student_checkpoint,
            self.model,
            map_location=self.device,
            strict=True,
        )
        for parameter in self.model.parameters():
            parameter.requires_grad = False
        self.visual = self.model.visual
        self.student = self.model.student

    def artifact_status(self) -> dict[str, Any]:
        return {
            "ready": True,
            "device": str(self.device),
            "corpus_samples": int(self.z.shape[0]),
            "embedding_dim": int(self.z.shape[1]),
            "dictionary_atoms": int(self.dictionary.shape[0]),
            "atom_labels": len(self.labels),
            "excluded_evidence_terms": list(self.excluded_evidence_terms),
            "excluded_atom_count": len(self.excluded_atom_ids),
        }

    def _image_seed(self, image_path: Path) -> int:
        digest = hashlib.sha256(Path(image_path).read_bytes()).digest()
        return (self.crop_seed + int.from_bytes(digest[:4], "big")) % (2**32)

    def embed_image(
        self,
        image_path: Path,
        *,
        seed: int | None = None,
    ) -> np.ndarray:
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        rng = np.random.default_rng(self._image_seed(image_path) if seed is None else seed)
        crops = multiview_crops_finetune(
            image,
            n_native_random=16,
            rng=rng,
            target=224,
            tile_px=0,
            global_kind="short_side",
            pre_crop_margin_px=1,
            tile_min_px=16,
            tile_max_px=224,
            tile_scale_ref_px=3000,
        )
        chunk_size = 4 if self.device.type == "mps" else 32
        with torch.inference_mode():
            embedding = batched_pooled_visual_ln_post_embeddings(
                self.visual,
                self.preprocess,
                [crops],
                self.device,
                encode_chunk_size=chunk_size,
            )
            z, _ = self.student(embedding)
        result = z.detach().to("cpu", dtype=torch.float32).numpy()[0]
        if result.shape != (2304,):
            raise ValueError(f"Expected 2304-D upload embedding, got {result.shape}")
        return result

    def encode_atoms(self, embedding: np.ndarray) -> np.ndarray:
        centered = np.asarray(embedding, dtype=np.float32).reshape(-1) - self.mean
        if (
            self.cfg.sparse_method != "elastic_net"
            or not self.cfg.sparse_codes_nonnegative
        ):
            raise ValueError(
                "The upload runtime requires the trained nonnegative Elastic Net dictionary."
            )
        return _positive_elastic_net_fista(
            centered,
            self.dictionary,
            alpha=float(self.cfg.elastic_net_alpha),
            l1_ratio=float(self.cfg.elastic_net_l1_ratio),
            max_iter=int(self.cfg.elastic_net_max_iter),
        )

    def nearest_neighbors(self, embedding: np.ndarray) -> list[dict[str, Any]]:
        return ai.faiss_neighbors_by_vector(
            self.z,
            embedding,
            self.df,
            k=self.neighbor_count,
            excluded_terms=self.excluded_evidence_terms,
        )

    def filter_atom_evidence(self, code: np.ndarray) -> np.ndarray:
        filtered = np.asarray(code, dtype=np.float32).copy()
        if self.excluded_atom_ids:
            atom_ids = np.fromiter(self.excluded_atom_ids, dtype=np.int64)
            filtered[atom_ids] = 0.0
        return filtered

    def inferred_context(
        self,
        neighbors: list[dict[str, Any]],
    ) -> dict[str, Any]:
        illumination = _normalized_frequency(neighbors, "illumination")
        magnification = _normalized_frequency(neighbors, "magnification")
        return {
            "illumination_frequencies": illumination,
            "magnification_frequencies": magnification,
            "illumination": illumination[0]["value"] if illumination else "microscopy",
            "magnification": (
                magnification[0]["value"] if magnification else "estimated magnification"
            ),
            "basis": "nearest-neighbor frequency with mean squared distance tie-break",
        }

    def describe_upload(
        self,
        image_path: Path,
        out_path: Path,
        *,
        coordinator_backend: str,
        coordinator_model: str,
        progress: ProgressCallback | None = None,
    ) -> dict[str, Any]:
        emit = progress or (lambda _stage, _percent, _message: None)
        emit("embedding", 15, "Encoding the uploaded image with Pipeline B v3.1.")
        embedding = self.embed_image(image_path)

        emit("neighbors", 35, "Searching the nearest microscopy neighbors.")
        neighbors = self.nearest_neighbors(embedding)
        inferred = self.inferred_context(neighbors)

        emit("atoms", 50, "Encoding the image with the global atom dictionary.")
        code = self.filter_atom_evidence(self.encode_atoms(embedding))

        # Reuse the notebook's exact synthesis path by appending one synthetic query row.
        synthetic = {column: "" for column in self.df.columns}
        synthetic.update(
            {
                "Image": Path(image_path).name,
                "Specimen Name": "Uploaded microscopy sample",
                "Source": "Uploaded image",
                "Illumination Modality": inferred["illumination"],
                "Magnification": inferred["magnification"],
            }
        )
        augmented_df = pd.concat(
            [self.df, pd.DataFrame([synthetic], columns=self.df.columns)],
            ignore_index=True,
        )
        augmented_z = np.vstack([self.z, embedding.reshape(1, -1)]).astype(
            np.float32, copy=False
        )
        augmented_codes = np.vstack([self.codes, code.reshape(1, -1)]).astype(
            np.float32, copy=False
        )

        emit("vision", 62, "Inspecting visible microscopy features.")
        emit("synthesis", 72, "Synthesizing neighbors, vision, and atom evidence.")
        result = ai.describe_sample(
            query_row=len(self.df),
            z=augmented_z,
            df=augmented_df,
            codes=augmented_codes,
            labels=self.labels,
            images_dir=Path(image_path).parent,
            out_path=out_path,
            cfg=self.cfg,
            coordinator_backend=coordinator_backend,
            coordinator_model=coordinator_model,
            faiss_k=self.neighbor_count,
            identity_max_nearest_d2=self.identity_max_nearest_d2,
            excluded_neighbor_terms=self.excluded_evidence_terms,
        )
        result["target"] = {
            "kind": "uploaded_image",
            "image": Path(image_path).name,
            "inferred_illumination_modality": inferred["illumination"],
            "inferred_magnification": inferred["magnification"],
        }
        result["upload_inference"] = {
            "embedding_dim": int(embedding.shape[0]),
            "crop_policy": "Pipeline B v3.1: 1 global + 16 adaptive native tiles",
            "context_inference": inferred,
            "retrieval_identity_max_nearest_d2": self.identity_max_nearest_d2,
            "excluded_evidence_terms": list(self.excluded_evidence_terms),
            "excluded_atom_ids": sorted(self.excluded_atom_ids),
            "coordinator_backend": coordinator_backend,
            "coordinator_model": coordinator_model,
        }
        out_path.write_text(
            json.dumps(result, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        emit("complete", 100, "Microscopy interpretation complete.")
        return result

