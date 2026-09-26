"""
Google Colab helpers: Drive mount and project root.

Use from a Colab cell after copying this file under your Drive project root, e.g.:

    from Colab.colab_env import colab_setup
    PROJECT_ROOT = colab_setup("NEHM_Project")  # folder name under MyDrive

Or set PROJECT_ROOT manually to an absolute Path.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional


def in_colab() -> bool:
    try:
        import google.colab  # noqa: F401

        return True
    except ImportError:
        return False


def mount_google_drive(force_remount: bool = False) -> None:
    """No-op if not running in Colab."""
    if not in_colab():
        return
    from google.colab import drive

    drive.mount("/content/drive", force_remount=force_remount)


def colab_setup(
    drive_folder_name: str = "NEHM_Project",
    *,
    mount: bool = True,
    force_remount: bool = False,
) -> Path:
    """
    Optionally mount Drive, then return ``/content/drive/MyDrive/<drive_folder_name>``
    and insert it on ``sys.path`` for ``nehm_pipeline`` imports.
    """
    if mount:
        mount_google_drive(force_remount=force_remount)
    root = Path("/content/drive/MyDrive") / drive_folder_name
    root = root.resolve()
    s = str(root)
    if s not in sys.path:
        sys.path.insert(0, s)
    return root


def assert_training_artifacts(
    project_root: Path,
    *,
    pipeline: str = "auto",
    images_subdir: str = "NEHM",
    xlsx_relative: str = "Database/Material_Database.xlsx",
) -> None:
    """Raise FileNotFoundError with a clear message if something obvious is missing.

    ``pipeline`` selects which teacher artifact set to require:
      - ``"A"``    : legacy LongCLIP-B fused-teacher bundle under ``NEHM_RESULTS/pipeline_notebook/``.
      - ``"B"``    : B v2 OpenAI ViT-L/14@336 global-only teacher under
                     ``LongCLIP_Embeddings_v1/output_random_local_global_B/``.
      - ``"auto"`` *(default)*: pass if **either** set is fully present.
    """
    p = Path(project_root)

    common = [
        p / "nehm_pipeline" / "clip_vit_multiview_finetune.py",
        p / xlsx_relative,
    ]

    pipeline_a = [
        p / "NEHM_RESULTS" / "pipeline_notebook" / "fused_embeddings.npy",
        p / "NEHM_RESULTS" / "pipeline_notebook" / "image_embeddings.npy",
        p / "NEHM_RESULTS" / "pipeline_notebook" / "manifest.tsv",
        p / "NEHM_RESULTS" / "pipeline_notebook" / "hdbscan_labels.csv",
    ]

    b_root = p / "LongCLIP_Embeddings_v1" / "output_random_local_global_B"
    pipeline_b = [
        b_root / "image_embeddings" / "image_embeddings_B_global.npy",
        b_root / "column_text_embeddings" / "columns_manifest.csv",
    ]

    def _missing(group: list[Path]) -> list[str]:
        return [str(x) for x in group if not x.is_file()]

    miss_common = _missing(common)
    miss_a = _missing(pipeline_a)
    miss_b = _missing(pipeline_b)

    pipe = pipeline.lower()
    if pipe == "a":
        missing = miss_common + miss_a
    elif pipe == "b":
        missing = miss_common + miss_b
    elif pipe == "auto":
        a_ok = not miss_a
        b_ok = not miss_b
        if a_ok or b_ok:
            missing = miss_common
        else:
            missing = miss_common + ["(neither pipeline A nor B v2 teacher artifacts are present)"] + miss_a + miss_b
    else:
        raise ValueError(f"pipeline must be 'A', 'B', or 'auto'; got {pipeline!r}")

    if missing:
        raise FileNotFoundError(
            "Missing training artifacts (sync from your Mac):\n  " + "\n  ".join(missing)
        )

    img_dir = p / images_subdir
    if not img_dir.is_dir():
        raise FileNotFoundError(f"Images directory not found: {img_dir}")
