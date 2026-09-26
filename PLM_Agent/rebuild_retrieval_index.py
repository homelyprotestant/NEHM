"""Rebuild retrieval vectors for the remapped trained Pigment Compendium rows."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import replace
from pathlib import Path

import numpy as np

from .upload_inference import PLMUploadInference, UploadArtifacts


PIGMENT_COMPENDIUM_START = 3743
PIGMENT_COMPENDIUM_STOP = 5177
WINDOW_SAMPLE_SEED = 42
ROW_SEED_STRIDE = 1_000_003


def rebuild(
    project_root: Path,
    *,
    start: int = PIGMENT_COMPENDIUM_START,
    stop: int = PIGMENT_COMPENDIUM_STOP,
    output: Path | None = None,
) -> Path:
    root = Path(project_root).expanduser().resolve()
    artifacts = UploadArtifacts.from_project_root(root)
    base_path = (
        root / "NEHM_RESULTS" / "student_inference_B" / "student_z_2304_B.npy"
    )
    artifacts = replace(artifacts, corpus_embeddings=base_path)
    engine = PLMUploadInference(artifacts)

    destination = output or base_path.with_name(
        "student_z_2304_B_current_images.npy"
    )
    destination = Path(destination).expanduser().resolve()
    temporary = destination.with_suffix(destination.suffix + ".partial")
    state_path = destination.with_suffix(destination.suffix + ".progress.json")

    first = max(0, int(start))
    last = min(int(stop), int(engine.z.shape[0]))
    if first >= last:
        raise ValueError(f"Empty rebuild range: [{first}, {last})")

    next_row = first
    if temporary.is_file() and state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if (
            int(state.get("start", -1)) == first
            and int(state.get("stop", -1)) == last
            and list(state.get("shape", [])) == list(engine.z.shape)
        ):
            rebuilt = np.lib.format.open_memmap(temporary, mode="r+")
            next_row = max(first, int(state.get("next_row", first)))
        else:
            raise RuntimeError(
                f"Existing partial rebuild has incompatible state: {state_path}"
            )
    else:
        rebuilt = np.lib.format.open_memmap(
            temporary,
            mode="w+",
            dtype=np.float32,
            shape=engine.z.shape,
        )
        rebuilt[:] = engine.z
        rebuilt.flush()

    total = last - first
    for row in range(next_row, last):
        image_path = root / "NEHM" / str(engine.df.iloc[row]["Image"])
        if not image_path.is_file():
            raise FileNotFoundError(f"Missing trained image for row {row}: {image_path}")
        seed = WINDOW_SAMPLE_SEED + row * ROW_SEED_STRIDE
        rebuilt[row] = engine.embed_image(image_path, seed=seed)

        completed = row - first + 1
        if completed % 25 == 0 or row + 1 == last:
            rebuilt.flush()
            state_path.write_text(
                json.dumps(
                    {
                        "start": first,
                        "stop": last,
                        "shape": list(engine.z.shape),
                        "next_row": row + 1,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            print(
                f"Rebuilt {completed}/{total} rows "
                f"({100.0 * completed / total:.1f}%)",
                flush=True,
            )

    rebuilt.flush()
    del rebuilt
    os.replace(temporary, destination)
    state_path.unlink(missing_ok=True)
    print(f"Wrote retrieval index: {destination}", flush=True)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--start", type=int, default=PIGMENT_COMPENDIUM_START)
    parser.add_argument("--stop", type=int, default=PIGMENT_COMPENDIUM_STOP)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rebuild(
        args.project_root,
        start=args.start,
        stop=args.stop,
        output=args.output,
    )


if __name__ == "__main__":
    main()
