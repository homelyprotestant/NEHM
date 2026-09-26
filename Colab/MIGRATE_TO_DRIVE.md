# Migrating NEHM ViT fine-tune to Google Drive (for Colab)

**Pre-built folder to upload:** run `bash colab/build_NEHM_Colab_Drive_bundle.sh` from the repo root, then upload **`NEHM_Colab_Drive/`** to Drive (see **`NEHM_Colab_Drive/PATHS_AND_UPLOAD.md`**). The script **rsyncs `NEHM_RESULTS/pipeline_notebook/`** and copies ViT-related **`.pth`** files; you only add **`NEHM/`** images.

---

Copy the items below into a **single folder tree on Drive** (example: `MyDrive/NEHM_Project/`). Paths in the Colab notebook assume **`PROJECT_ROOT`** points at that folder.

## 1. Python package (required)

| Path on Drive | Purpose |
|---------------|---------|
| `nehm_pipeline/` | **Entire directory** — all `*.py` files and `__init__.py` |

Without this, `import nehm_pipeline...` fails.

## 2. Pipeline embeddings + manifest (required)

Under `NEHM_RESULTS/pipeline_notebook/` (or whatever you set as `PIPELINE_OUT`):

| File | Used by |
|------|---------|
| `fused_embeddings.npy` | Teacher vector for `joint_material_loss` (L1 on `z`) |
| `image_embeddings.npy` | Student `image_dim` / consistency checks |
| `hdbscan_labels.csv` | Loaded by bundle (length check); not the CE target if you use `logitb` from Excel |
| `manifest.tsv` | Row order + **filenames** column → must match `N` rows of `.npy` files |

## 3. Images (required)

| Path | Purpose |
|------|---------|
| `NEHM/` (or your `IMAGES_DIR`) | Image files named exactly as in `manifest.tsv` (second column) |

Same relative layout as on your Mac if `manifest` stores bare filenames like `0001.jpg`.

## 4. Spreadsheet (required)

| Path | Purpose |
|------|---------|
| `Database/Material_Database.xlsx` | Sheet **`Sheet2`**, columns **`Image Index`**, **`logitb`** (and other metadata) |

## 5. Checkpoints (optional but typical)

Under `NEHM_RESULTS/`:

| File | Purpose |
|------|---------|
| `student_clip_student.pth` | Fallback **decoder-only** warm start if no ViT `.pth` exists |
| `vit_student_finetune_best.pth` / `vit_student_finetune_best_logitb.pth` | Preferred unified resume |
| `vit_finetune_stage_B_last2.pth`, `vit_finetune_stage_A_readout.pth`, … | Stage checkpoints for resume |

## 6. Not required on Drive

- Local-only `Notebooks/` copy is optional if you only run **`colab/distiller_CLIP_ViT_finetune.ipynb`**.
- `LongCLIP/`, other notebooks, and git history are optional.
- CLIP ViT-L/14 weights download automatically on first `clip.load` (needs network on Colab).

## 7. Checkpoint directory (`nehm_pipeline.checkpoint_paths`)

The ViT notebook sets **`FINETUNE_CKPT_DIR`** with **`resolve_checkpoint_dir(..., prefer_cloud=True)`**:

- **Colab** (Drive mounted): `My Drive/<DRIVE_PROJECT_FOLDER>/NEHM_RESULTS/` — `.pth` files land there directly.
- **macOS + Google Drive for desktop:** same path under `Library/CloudStorage/GoogleDrive-*/`.
- Otherwise: `PROJECT_ROOT/NEHM_RESULTS`.

Set **`PREFER_SAVE_CHECKPOINTS_TO_GOOGLE_DRIVE = False`** in the notebook to force local project-only saves.

## 8. After the run

On **ephemeral** Colab disk, your weights still live on **Drive** if `FINETUNE_CKPT_DIR` resolved there. Otherwise download **`NEHM_RESULTS/*.pth`** manually.

## 9. Colab runtime: GPU vs TPU

**Use a GPU runtime, not TPU.**

- This stack is **PyTorch + OpenAI CLIP** on **CUDA** (or CPU). **TPU** would require `torch_xla`, different data loading, and is a poor fit for this notebook without a full rewrite.
- **Free tier:** **T4** (16 GB) — often tight for **batch_size=5** with **17 ViT forwards** per sample; start with **`BATCH_SIZE=1` or `2`** and increase if memory allows.
- **Colab Pro / Pro+:** Prefer **L4** or **A100** (more VRAM, faster). **A100 40 GB** is the most comfortable for larger batch sizes with ViT-L/14@336.

See the Colab notebook’s intro cell for the same summary.
