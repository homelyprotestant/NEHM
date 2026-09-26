"""
Resolve where to save large ``.pth`` checkpoints: prefer Google Drive when available.

- **Colab:** ``/content/drive/MyDrive/<folder>/NEHM_RESULTS`` when Drive is mounted.
- **macOS + Google Drive for desktop:** ``~/Library/CloudStorage/GoogleDrive-*/My Drive/<folder>/NEHM_RESULTS``
- Otherwise: ``project_root / "NEHM_RESULTS"``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional


def resolve_checkpoint_dir(
    project_root: Path,
    *,
    drive_project_folder: str = "NEHM_Project",
    prefer_cloud: bool = True,
    results_subdir: str = "NEHM_RESULTS",
) -> Path:
    """
    Return a directory for ViT / student ``.pth`` files.

    ``prefer_cloud``: try cloud-backed paths first so checkpoints sync to Google Drive
    without manual copying (Colab or macOS Drive client).
    """
    project_root = Path(project_root).resolve()

    if prefer_cloud:
        # Google Colab
        colab_drive = Path("/content/drive/MyDrive")
        if colab_drive.is_dir():
            out = (colab_drive / drive_project_folder / results_subdir).resolve()
            out.mkdir(parents=True, exist_ok=True)
            return out

        # macOS — Google Drive for desktop
        cloud = Path.home() / "Library/CloudStorage"
        if cloud.is_dir():
            candidates = sorted(
                p
                for p in cloud.iterdir()
                if p.is_dir() and p.name.startswith("GoogleDrive-")
            )
            for g in candidates:
                out = (g / "My Drive" / drive_project_folder / results_subdir).resolve()
                # Require My Drive to exist; create NEHM_RESULTS (and parents) under it
                my_drive = g / "My Drive"
                if my_drive.is_dir():
                    out.mkdir(parents=True, exist_ok=True)
                    return out

    out = (project_root / results_subdir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    return out
