from __future__ import annotations

import io
import json
import time
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

from PLM_Agent.web.api import create_app
from PLM_Agent.web.settings import WebSettings


class FakeEngine:
    def artifact_status(self):
        return {
            "ready": True,
            "device": "test",
            "corpus_samples": 3,
            "embedding_dim": 2304,
            "dictionary_atoms": 400,
            "atom_labels": 400,
        }

    def describe_upload(
        self,
        image_path,
        out_path,
        *,
        coordinator_backend,
        coordinator_model,
        progress,
    ):
        progress("embedding", 20, "Embedding test image.")
        progress("neighbors", 40, "Finding test neighbors.")
        progress("atoms", 55, "Coding test atoms.")
        progress("vision", 70, "Inspecting test image.")
        progress("synthesis", 85, "Writing test description.")
        result = {
            "target": {
                "kind": "uploaded_image",
                "image": Path(image_path).name,
                "inferred_illumination_modality": "polarized",
            },
            "pipeline": {"synthesis_fallback_used": False},
            "established_type": "particle",
            "suggested_chemical_formula": "CdS",
            "magnification_estimate": "approximately 40×",
            "microscopy_description": "Test microscopy description.",
            "faiss_neighbors": [
                {
                    "database_row": 1,
                    "d2": 0.2,
                    "specimen": "Cadmium yellow",
                    "chemical_formula": "CdS",
                    "illumination": "polarized",
                    "magnification": "40X",
                }
            ],
            "direct_visual_observations": {},
            "weighted_atoms": [],
            "atom_contributions": [],
            "upload_inference": {
                "context_inference": {
                    "illumination": "polarized",
                    "magnification": "40X",
                }
            },
        }
        Path(out_path).write_text(json.dumps(result), encoding="utf-8")
        progress("complete", 100, "Complete.")
        return result


def _png_bytes() -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (64, 64), (220, 180, 40)).save(output, format="PNG")
    return output.getvalue()


def test_upload_job_lifecycle(tmp_path: Path) -> None:
    static_dir = Path(__file__).resolve().parents[1] / "web" / "static"
    settings = WebSettings(
        project_root=Path(__file__).resolve().parents[2],
        data_dir=tmp_path,
        static_dir=static_dir,
        max_upload_bytes=2 * 1024 * 1024,
        result_ttl_seconds=3600,
        max_queue_size=2,
    )
    app = create_app(settings=settings, engine=FakeEngine())
    with TestClient(app) as client:
        health = client.get("/api/health")
        assert health.status_code == 200
        response = client.post(
            "/api/jobs",
            files={"image": ("test.png", _png_bytes(), "image/png")},
            data={"coordinator": "openai"},
        )
        assert response.status_code == 202
        job = response.json()
        deadline = time.time() + 5
        while time.time() < deadline:
            status = client.get(job["status_url"]).json()
            if status["status"] in {"complete", "failed"}:
                break
            time.sleep(0.05)
        assert status["status"] == "complete"
        result = client.get(job["result_url"])
        assert result.status_code == 200
        assert result.json()["suggested_chemical_formula"] == "CdS"
        download = client.get(job["download_url"])
        assert download.status_code == 200
        assert download.headers["content-type"].startswith("application/json")

    restarted_app = create_app(settings=settings, engine=FakeEngine())
    with TestClient(restarted_app) as restarted_client:
        recovered = restarted_client.get(job["status_url"])
        recovered_result = restarted_client.get(job["result_url"])
        recovered_download = restarted_client.get(job["download_url"])
    assert recovered.status_code == 200
    assert recovered.json()["status"] == "complete"
    assert recovered_result.json()["suggested_chemical_formula"] == "CdS"
    assert recovered_download.status_code == 200


def test_rejects_non_image(tmp_path: Path) -> None:
    static_dir = Path(__file__).resolve().parents[1] / "web" / "static"
    settings = WebSettings(
        project_root=Path(__file__).resolve().parents[2],
        data_dir=tmp_path,
        static_dir=static_dir,
        max_upload_bytes=1024,
        result_ttl_seconds=3600,
        max_queue_size=1,
    )
    app = create_app(settings=settings, engine=FakeEngine())
    with TestClient(app) as client:
        response = client.post(
            "/api/jobs",
            files={"image": ("bad.txt", b"not an image", "text/plain")},
            data={"coordinator": "openai"},
        )
    assert response.status_code == 415


def test_rejects_non_openai_coordinator(tmp_path: Path) -> None:
    static_dir = Path(__file__).resolve().parents[1] / "web" / "static"
    settings = WebSettings(
        project_root=Path(__file__).resolve().parents[2],
        data_dir=tmp_path,
        static_dir=static_dir,
        max_upload_bytes=2 * 1024 * 1024,
        result_ttl_seconds=3600,
        max_queue_size=1,
    )
    app = create_app(settings=settings, engine=FakeEngine())
    with TestClient(app) as client:
        response = client.post(
            "/api/jobs",
            files={"image": ("test.png", _png_bytes(), "image/png")},
            data={"coordinator": "local-qwen"},
        )
    assert response.status_code == 422


def test_public_runtime_headers_hide_docs_and_secrets(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-secret-must-not-leak")
    static_dir = Path(__file__).resolve().parents[1] / "web" / "static"
    settings = WebSettings(
        project_root=Path(__file__).resolve().parents[2],
        data_dir=tmp_path,
        static_dir=static_dir,
        max_upload_bytes=2 * 1024 * 1024,
        result_ttl_seconds=3600,
        max_queue_size=1,
        public_prototype=True,
    )
    app = create_app(settings=settings, engine=FakeEngine())
    with TestClient(app) as client:
        health = client.get("/api/health")
        docs = client.get("/docs")

    assert health.status_code == 200
    assert health.headers["cache-control"] == "no-store"
    assert health.headers["x-content-type-options"] == "nosniff"
    assert health.headers["x-frame-options"] == "DENY"
    assert "test-secret-must-not-leak" not in health.text
    assert docs.status_code == 404

