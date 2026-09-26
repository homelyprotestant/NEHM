from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from PLM_Agent import atom_interpretability as ai
from PLM_Agent.upload_inference import (
    PLMUploadInference,
    UploadArtifacts,
    _normalized_frequency,
)


def test_minor_contrastive_phrases_do_not_reject_synthesis() -> None:
    ai._validate_positive_text(
        "The non-fibrous aggregate contains mineral shards rather than long filaments.",
        stage="sample synthesis",
    )


def test_vector_neighbors_and_consensus() -> None:
    corpus = np.array([[0.0, 0.0], [1.0, 0.0], [4.0, 0.0]], dtype=np.float32)
    frame = pd.DataFrame(
        [
            {
                "Specimen Name": "A",
                "composition": "X",
                "Illumination Modality": "polarized",
                "Magnification": "40X",
            },
            {
                "Specimen Name": "A",
                "composition": "X",
                "Illumination Modality": "polarized",
                "Magnification": "40X",
            },
            {
                "Specimen Name": "B",
                "composition": "Y",
                "Illumination Modality": "brightfield",
                "Magnification": "100X",
            },
        ]
    )
    hits = ai.faiss_neighbors_by_vector(
        corpus,
        np.array([0.2, 0.0], dtype=np.float32),
        frame,
        k=3,
    )
    assert [hit["database_row"] for hit in hits] == [0, 1, 2]
    assert all("chemical_formula" in hit for hit in hits)
    assert all("magnification" in hit for hit in hits)
    assert _normalized_frequency(hits, "illumination")[0]["value"] == "polarized"
    assert _normalized_frequency(hits, "magnification")[0]["value"] == "40X"


def test_formula_consensus_requires_same_specimen_agreement() -> None:
    neighbors = [
        {
            "database_row": row,
            "d2": 0.5 + row,
            "specimen": specimen,
            "chemical_formula": "X2Y3",
        }
        for row, specimen in enumerate(["Rare A", "Rare B", "Rare C"])
    ]
    consensus = ai._l2_neighbor_consensus(
        neighbors,
        identity_max_nearest_d2=8.0,
    )
    assert consensus["repeated_formula_suggestion"] is None

    neighbors[1]["specimen"] = "Rare A"
    consensus = ai._l2_neighbor_consensus(
        neighbors,
        identity_max_nearest_d2=8.0,
    )
    assert consensus["most_repeated_specimen"]["value"] == "Rare A"
    assert consensus["repeated_formula_suggestion"]["value"] == "X2Y3"


def test_ultramarine_lazurite_family_consensus_overrides_distance_gate() -> None:
    names = ["lazurite"] * 8 + ["ultramarine", "smalt"]
    neighbors = [
        {
            "database_row": row,
            "d2": 18.0 + row,
            "specimen": name,
            "chemical_formula": "Na7Al6Si6O24S3" if name != "smalt" else "",
        }
        for row, name in enumerate(names)
    ]
    consensus = ai._l2_neighbor_consensus(
        neighbors,
        identity_max_nearest_d2=8.0,
    )
    assert consensus["distance_identity_allowed"] is False
    assert consensus["identity_inference_allowed"] is True
    assert consensus["strong_identity_family_consensus"]["value"] == (
        "ultramarine/lazurite"
    )
    assert consensus["strong_identity_family_consensus"]["count"] == 9

    ai._validate_identity_family_consensus(
        {
            "microscopy_description": (
                "The nearest neighbors support ultramarine, whose characteristic blue "
                "mineral phase is lazurite."
            )
        },
        consensus,
    )


def test_at_least_ten_atoms_are_considered() -> None:
    ranked_atoms = [{"atom_id": atom_id} for atom_id in range(12)]
    payload = {
        "atom_contributions": [
            {"atom_id": atom_id, "contribution": "transferable attribute"}
            for atom_id in range(10)
        ]
    }
    ai._validate_minimum_atom_contributions(payload, ranked_atoms, minimum=10)

    with pytest.raises(ValueError, match="at least 10"):
        ai._validate_minimum_atom_contributions(
            {"atom_contributions": payload["atom_contributions"][:3]},
            ranked_atoms,
            minimum=10,
        )


def test_vivianite_is_excluded_from_neighbors_and_atom_evidence() -> None:
    corpus = np.array(
        [[0.0, 0.0], [0.1, 0.0], [0.2, 0.0], [0.3, 0.0]],
        dtype=np.float32,
    )
    frame = pd.DataFrame(
        [
            {"Specimen Name": "vivianite"},
            {"Specimen Name": "CHAL_vivianite_pp"},
            {"Specimen Name": "ultramarine"},
            {"Specimen Name": "azurite"},
        ]
    )
    hits = ai.faiss_neighbors_by_vector(
        corpus,
        np.array([0.0, 0.0], dtype=np.float32),
        frame,
        k=2,
        excluded_terms=("vivianite",),
    )
    assert [hit["specimen"] for hit in hits] == ["ultramarine", "azurite"]

    augmented_corpus = np.vstack(
        [corpus, np.array([[0.0, 0.0]], dtype=np.float32)]
    )
    augmented_frame = pd.concat(
        [frame, pd.DataFrame([{"Specimen Name": "Uploaded sample"}])],
        ignore_index=True,
    )
    synthesis_hits = ai.faiss_neighbors(
        augmented_corpus,
        4,
        augmented_frame,
        k=2,
        excluded_terms=("vivianite",),
    )
    assert [hit["specimen"] for hit in synthesis_hits] == [
        "ultramarine",
        "azurite",
    ]

    engine = object.__new__(PLMUploadInference)
    engine.excluded_atom_ids = {1, 3}
    filtered = engine.filter_atom_evidence(
        np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    )
    assert filtered.tolist() == [1.0, 0.0, 3.0, 0.0]


@pytest.mark.skipif(
    os.getenv("PLM_RUN_MODEL_TESTS") != "1",
    reason="Set PLM_RUN_MODEL_TESTS=1 for the checkpoint smoke test.",
)
def test_known_corpus_image_matches_saved_embedding() -> None:
    root = Path(__file__).resolve().parents[2]
    artifacts = UploadArtifacts.from_project_root(root)
    engine = PLMUploadInference(artifacts)
    row = 0
    image_name = str(engine.df.iloc[row]["Image"])
    image_path = root / "NEHM" / image_name
    inferred = engine.embed_image(image_path, seed=42 + row * 1_000_003)
    expected = engine.z[row]
    cosine = float(
        np.dot(inferred, expected)
        / (np.linalg.norm(inferred) * np.linalg.norm(expected) + 1e-12)
    )
    assert inferred.shape == (2304,)
    assert cosine > 0.999
    code = engine.encode_atoms(inferred)
    assert code.shape == (400,)
    assert np.isfinite(code).all()
    assert float(code.max()) > 0
    expected_code = engine.codes[row]
    code_cosine = float(
        np.dot(code, expected_code)
        / (np.linalg.norm(code) * np.linalg.norm(expected_code) + 1e-12)
    )
    assert code_cosine > 0.98

