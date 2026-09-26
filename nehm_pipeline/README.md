# NEHM pipeline

Multimodal embeddings (local CLIP ViT-L/14 at 224×224 with sliding-window image pooling + text fields) and clustering (UMAP + HDBSCAN), optimized for Apple MPS when available.

Behavior is aligned with [Notebooks/Clip_clusterb.ipynb](Notebooks/Clip_clusterb.ipynb): same text fields, fusion in the spirit of `tst`/`ts2`, UMAP (`n_neighbors=20`, `min_dist=0`, `cosine`, `random_state` unset like the notebook), HDBSCAN on **2D UMAP** coordinates (`min_cluster_size=20`, `min_samples=5`), and matplotlib figures matching cells 44–45 (gray background, `s=2`, `tab20` cluster colors + legend; optional thumbnails).

## Setup

From the project root:

```bash
pip install -r requirements-nehm.txt
```

Install **OpenAI CLIP** the same way as in your notebook (not pinned in this file), e.g. `pip install git+https://github.com/openai/CLIP.git`.

## Paths (defaults)

| Item | Default |
|------|---------|
| Excel | `Database/Material_Database.xlsx` |
| JSON | `Database/Material_Database.json` |
| Images | `NEHM/` |
| Outputs | `NEHM_RESULTS/pipeline/` |

## Commands

```bash
# 1) Export Excel → JSON (NaN-safe)
python -m nehm_pipeline export

# 2) CLIP embeddings (MPS/CUDA/CPU) + sliding windows + fusion
python -m nehm_pipeline embed --images /path/to/NEHM

# 3) UMAP + HDBSCAN on fused vectors (+ umap_scatter.png by default)
python -m nehm_pipeline cluster

# Optional: large figure with image thumbnails (cell 45 style; subsampled)
python -m nehm_pipeline cluster --plot-thumbnails --max-thumbnails 400

# Regenerate plots only
python -m nehm_pipeline plot --plot-thumbnails

# Or full chain
python -m nehm_pipeline all --images /path/to/NEHM
```

Use `--device cpu` if MPS/CUDA causes issues.

**Patches:** default **`center_biased_random`** uses **one** tile that is the **entire image downsampled to 224×224**, plus **random** 224×224 crops with a **center-biased** Gaussian (see `center_bias_sigma_frac` in config). Legacy **`sliding_grid`** is available via `--patch-mode sliding_grid`. **`max_patches_per_image`** caps the **total** count (including the global tile when enabled). Reproducible crops: `window_sample_seed` in config or `--window-seed`; `-1` = random each run.

Use `--no-plot` to skip PNGs. `--plot-monochrome` matches the notebook’s active scatter (uncolored).

## Outputs

After `embed`: `fused_embeddings.npy`, `image_embeddings.npy`, `text_embeddings_6xD.npy`, `manifest.tsv`.

After `cluster`: `umap_xy.csv`, `hdbscan_labels.csv`, `cluster_summary.json`, `per_image_clusters.tsv`, **`umap_scatter.png`**, and optionally **`umap_thumbnails.png`**.

## Jupyter test notebook

[Notebooks/NEHM_pipeline_test.ipynb](Notebooks/NEHM_pipeline_test.ipynb) runs export → embed → cluster → inline plot. Set `MAX_RECORDS = 32` for a quick test, or `None` for the full dataset. Run with the kernel’s cwd at **`NEHM_Project`** (or open the notebook from that folder).
