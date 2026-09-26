# Pipeline B — v3.1, **LongCLIP-B + 16+1 *scale-adaptive* multiview** (ViT-B/16, 512-D image + text, 2048-D fused teacher) + K-SVD dictionary

Pipeline **B v3.1** is the current production line for Pipeline B. It keeps
all of v3's design choices — LongCLIP-B backbone, 17-view multiview, fused
2048-D teacher, K-SVD dictionary — and changes only one thing: **the random
tiles are now sized per-image so that each tile samples a roughly constant
fraction of the image's physical field of view**, ranging from 16×16 native
pixels on the smallest images up to 224×224 (the full LongCLIP-B input) on
the largest.

What changed from B v3:

- **Random tiles are now scale-adaptive.** The per-image tile side is

  ```text
  tile_px = clip(round(TILE_MAX_PX × sqrt(W·H) / TILE_SCALE_REF_PX),
                 TILE_MIN_PX, TILE_MAX_PX)
  ```

  with defaults `TILE_MIN_PX = 16`, `TILE_MAX_PX = 224`,
  `TILE_SCALE_REF_PX = 3000`. Set `TILE_PX > 0` (or
  `cfg.random_crop_px > 0` on the teacher side) to revert to the legacy
  fixed-size B-v3 tiles.
- **Image manifest gains four columns**: `image_w_px`, `image_h_px`,
  `image_geo_px`, `tile_px_used`. Step A in the teacher notebook prints a
  histogram of the tile sizes actually used so you can sanity-check the
  scaling on your dataset.
- Everything else (LongCLIP-B backbone, 1-pixel scan-border trim, no
  per-channel normalization, 0.5/0.5 image-teacher weighting, 2048-D fused
  teacher, `512 → 1024 → 1536 → 2048` student head, K-SVD section in the
  inference notebook) is identical to B v3.

What changed from B v2 → B v3 (still applies, recap):

1. **Backbone reverted to LongCLIP-B (ViT-B/16, 512-D image + text).**
2. **Multiview re-enabled at the Pipeline-A view count (16 + 1).**
3. **Per-channel normalization stays off** (linear RGB in [0, 1]).
4. **A 1-pixel border is dropped before resize** (`pre_crop_margin_px = 1`).
5. **Student head re-shaped** for 512-D input → 2048-D fused teacher
   (`decoder_hidden = (1024, 1536)`).
6. **K-SVD dictionary learning** added to the inference notebook.

If you are coming from B v3: only the teacher's random-pool npy needs to be
regenerated (the global-view npy and all text npys are unchanged in dim and
preprocessing). The student must be retrained from `Z_embed` because the
random-pool teacher contribution changes per row. The student head shape is
unchanged.

If you are coming from B v2: re-encode the teacher (image + text) under
LongCLIP-B and retrain the student from `Z_embed`, because the head shape and
target dim both changed (3072 → 2048).

## Why scale-adaptive tiles

The dataset spans roughly 150 - 3020-pixel short sides across sources, but
every image is approximately the same physical FOV (a single microscopy /
Pigment Compendium frame). A *fixed* 16×16 tile therefore samples wildly
different *physical* fractions of that FOV depending on source:

| Source              | ≈ rows | Native short side | Fixed 16×16 tile covers… | Adaptive tile size (defaults) | Fraction of FOV per tile |
|---------------------|--------|-------------------|---------------------------|--------------------------------|--------------------------|
| Emrath              | 2%     | ~150 px           | ~1.1% of area             | **16×16** (clamped)            | ~1.1%                    |
| Pigment Compendium  | 27%    | ~302 px           | ~0.28% of area            | **22×22**                      | ~0.5%                    |
| Cameo-FRIL / TMI    | 46%    | 450–600 px        | ~0.07–0.13% of area       | **34–45×34–45**                | ~0.5%                    |
| McCrone / alaskafurid / Cameo | 25% | 900–3020 px | ~0.003–0.03% of area | **70–224×70–224**              | ~0.5%                    |

Adaptive sizing keeps that "fraction of the image per tile" within roughly an
order of magnitude across the entire dataset, while a fixed 16×16 tile spans
*four* orders of magnitude (~0.003% on McCrone vs. ~1.1% on Emrath). At
McCrone resolution, 16 random 16×16 tiles together cover ~0.05% of the image
— effectively a 16-pixel sub-thumbnail noise pool. Adaptive tiles at McCrone
resolution cover ~8% of the image (~50× more), and in particular make the
random pool genuinely additive over the global view (which already sees a
0.6× downscale on McCrone).

CLIP's `preprocess` always upsamples each tile to 224×224 before the ViT
sees it, so the *network* always sees a fixed-resolution input — the
variation is purely in **what fraction of the FOV that input was computed
from**.

## What's the same as A and B v1/v2

- L1 latent reconstruction is the **primary objective**; CE on `logitb` weak
  labels is a probe + light regularizer (`λ_emb = 1.0`, `λ_ce = 0.3`, label
  smoothing `0.10`).
- Train-time geometric augmentations (`hflip` / `vflip` p = 0.5; uniform ±5°
  rotation, bilinear, expanded canvas, black fill).
- WeightedRandomSampler (`inv_sqrt`) on labeled training rows balances both
  L1 and CE.
- Per-row Gaussian teacher noise scaled by per-dim std (`std = 0.005 × σ`).
- Three-way student dropout (input 0.05 / decoder 0.0 / classifier 0.20).
- Stochastic Weight Averaging in `D_full_decay` (last 3 epochs).
- Stage-wise progressive unfreezing schedule and layer-wise LR decay (0.75)
  in `D_full_decay`.

## A vs B v1 vs B v2 vs B v3 vs B v3.1 (current)

|                              | **A**                                  | **B v1**                              | **B v2** *(deprecated)*                                 | **B v3**                                                                       | **B v3.1** *(current)*                                                              |
|------------------------------|----------------------------------------|---------------------------------------|---------------------------------------------------------|--------------------------------------------------------------------------------|--------------------------------------------------------------------------------------|
| Visual backbone              | LongCLIP-B (ViT-B/16@224)              | LongCLIP-B (ViT-B/16@224)             | OpenAI ViT-L/14@336                                     | LongCLIP-B (ViT-B/16@224)                                                       | **LongCLIP-B (ViT-B/16@224)**                                                        |
| Text backbone                | LongCLIP-B text (512-D, 248 tokens)    | LongCLIP-B text (512-D, 248 tokens)   | OpenAI ViT-L/14@336 text (768-D, 77 tokens)             | LongCLIP-B text (512-D, 248 tokens)                                             | **LongCLIP-B text (512-D, 248 tokens)**                                              |
| Image emb dim                | 512                                    | 512                                   | 768                                                     | 512                                                                             | **512**                                                                              |
| Text emb dim                 | 512                                    | 512                                   | 768                                                     | 512                                                                             | **512**                                                                              |
| Fused teacher dim            | 4 × 512 = 2048                         | 4 × 512 = 2048                        | 4 × 768 = 3072                                          | 4 × 512 = 2048                                                                  | **4 × 512 = 2048**                                                                   |
| Student head                 | 512 → 1024 → 1536 → 2048               | 512 → 1024 → 1536 → 2048              | 768 → 1024 → 1536 → 2048 → 3072                         | 512 → 1024 → 1536 → 2048                                                        | **512 → 1024 → 1536 → 2048**                                                         |
| Global view                  | resize-height-336, center-crop 336×336 | short-side-224, center-crop 224×224   | drop 1-px border → short-side-336 → center-crop 336×336 | drop 1-px border → short-side-224 → center-crop 224×224                         | **drop 1-px border → short-side-224 → center-crop 224×224**                          |
| Random tile sizing           | fixed 336×336 native                   | fixed 16×16 native                    | n/a (no tiles)                                          | fixed 16×16 native                                                              | **scale-adaptive `[16-224]px` clipped, ref=3000**                                    |
| Local tiles per row          | 16                                     | 256                                   | 0                                                       | 16                                                                              | **16**                                                                               |
| Total views per row          | 17                                     | 257                                   | 1                                                       | 17                                                                              | **17**                                                                               |
| Image teacher fusion         | weighted avg(random, global)           | weighted avg(random, global)          | L2(global) only                                         | `0.5·L2(global) + 0.5·L2(mean(random))`                                         | **`0.5·L2(global) + 0.5·L2(mean(random))`**                                          |
| Per-channel normalization    | on (CLIP `preprocess`)                 | on (CLIP `preprocess`)                | off                                                     | off                                                                             | **off** (Normalize stripped; linear RGB in [0, 1])                                   |
| Teacher–student parity       | mismatched                             | parity ✅                              | parity ✅                                                | parity ✅                                                                        | **parity ✅**                                                                         |
| Dictionary section           | no                                     | no                                    | no                                                      | **K-SVD with logit + Bayesian per-atom decoding**                               | **K-SVD with logit + Bayesian per-atom decoding**                                    |
| Per-epoch GPU cost (relative)| 1×                                     | ~15× A                                | ~0.1× A (1 view, but ViT-L/14@336)                      | ~1× A (17 views × ViT-B/16@224 — same as A)                                     | **~1.0–1.3× A** (same view count; tiles are larger on hi-res images so per-tile preprocess + ViT cost grows slightly) |

## Geometry, in one paragraph (B v3.1)

For each image, B v3.1 generates **17 views**: drop a 1-pixel border, resize
so the short side is 224 px (preserving aspect), center-crop to 224×224 — that
is the global view. From the same trimmed image, compute the per-image tile
side `tile_px = clip(round(TILE_MAX_PX × sqrt(W·H) / TILE_SCALE_REF_PX), TILE_MIN_PX, TILE_MAX_PX)`
(defaults 16, 224, 3000), then sample 16 random `tile_px × tile_px`
native-resolution tiles. All 17 views are fed through `LongCLIP.encode_image`
(with `Normalize` stripped — RGB stays linear in [0, 1] so microscopy-relevant
channel ratios are not deformed; this is the framework default, also used in
the trainer / inference notebooks). The 16 tile embeddings are mean-pooled
and L2-normalized to give `random_pooled`; the global view is L2-normalized
to give `global`. The image-side teacher vector for the row is
`L2(0.5 × random_pooled + 0.5 × global)`.

The 1-pixel border trim removes the typical scan-border artifact on Pigment
Compendium book scans, which we measured at 0–1 pixels. A few Cameo-FRIL
images have much larger asymmetric blackout regions (hundreds of pixels)
that this trim does *not* address; if those start hurting validation, switch
to a content-aware bounding-box crop later.

## Fused teacher block layout (2048-D)

```text
Z = [ image_512  ⨁  illum_512  ⨁  avg(class+subclass)_512  ⨁  avg(desc+meta+composition×2)_512 ]
        image_512 = L2(0.5·L2(random_pool) + 0.5·L2(global))
```

Same masking and `composition × 2` weighting as A and B v1.

## Student head architecture (B v3.1)

```text
input  : 512-D LongCLIP-B image embedding (multiview pooled at training time, single-stream at inference)
head   : Linear+LayerNorm+GELU stack with widths 512 → 1024 → 1536 (final = plain Linear) → 2048
        (i.e. ImageEmbeddingStudent(image_dim=512, decoder_hidden=(1024, 1536), embed_dim=2048))
output : z ∈ ℝ^2048  — the L1 target against the fused teacher
classif: Linear(2048 → K)  — CE on `logitb`
```

Unchanged from B v3.

## File layout

|                                               | path                                                                                        |
|-----------------------------------------------|---------------------------------------------------------------------------------------------|
| LongCLIP package (vendored)                   | `LongCLIP/model/` (loader: `Notebooks/Teacher_inference_embeddings_B.ipynb` cell 4)         |
| LongCLIP checkpoint                           | `LongCLIP/checkpoints/longclip-B.pt`                                                        |
| Image embedding helpers                       | `nehm_pipeline/longclip_image_embeddings.py`                                                |
| Trim-aware preprocessing                      | `nehm_pipeline/preprocess_image.short_side_resize_center_crop`                              |
| Adaptive-tile helpers                         | `nehm_pipeline/preprocess_image.compute_adaptive_tile_px` + `random_square_crops_adaptive_native_resolution` |
| Multiview wrapper                             | `nehm_pipeline/clip_vit_multiview_finetune.py` (used with `n_native_random=16, tile_px=0`)  |
| Teacher notebook (B v3.1)                     | `Notebooks/Teacher_inference_embeddings_B.ipynb`                                            |
| Student trainer (B v3.1)                      | `Colab/Student_Trainer_B.ipynb`                                                             |
| Student inference (B v3.1)                    | `Notebooks/Student_inference_embeddings_B.ipynb` (incl. K-SVD section §8)                   |
| Image embedding (512-D) — global              | `LongCLIP_Embeddings_v1/output_random_local_global_B/image_embeddings/image_embeddings_B_global.npy`        |
| Image embedding (512-D) — random pooled       | `LongCLIP_Embeddings_v1/output_random_local_global_B/image_embeddings/image_embeddings_B_random_pooled.npy` |
| Per-row image manifest                        | `LongCLIP_Embeddings_v1/output_random_local_global_B/image_embeddings/image_manifest_B.csv` (now includes `image_w_px`, `image_h_px`, `image_geo_px`, `tile_px_used`) |
| Text embeddings (512-D, per column)           | `LongCLIP_Embeddings_v1/output_random_local_global_B/column_text_embeddings/*.npy` + `columns_manifest.csv` |
| Best student checkpoint                       | `NEHM_RESULTS/checkpoints/.../vit_student_finetune_best_B_logitb.pth`                       |
| Per-epoch log                                 | `NEHM_RESULTS/checkpoints/.../vit_finetune_training_history_B.jsonl`                        |
| Inference outputs                             | `NEHM_RESULTS/student_inference_B/student_image_emb_512_B.npy`, `student_z_2048_B.npy`, `student_logits_B.npy`, `student_predictions_B.csv` |
| K-SVD outputs                                 | `NEHM_RESULTS/student_inference_B/ksvd_dictionary/ksvd_dictionary.npy`, `ksvd_sparse_codes_all.npy`, `atom_usage.csv`, `atom_logit_footprint.csv`, `atom_metadata_posterior.csv`, `ksvd_manifest.json` |

## Running B v3.1 end-to-end

1. **Regenerate teacher embeddings (image + text)** — `Notebooks/Teacher_inference_embeddings_B.ipynb`:
   - Wires the vendored `LongCLIP/model/` package, then loads `longclip-B.pt`.
     Strips `Normalize` from the CLIP preprocess (linear RGB in [0, 1]).
     Probes `visual.output_dim == 512`.
   - Step A: encodes 17 views per image (16 random scale-adaptive native
     tiles + 1 global short-side-224 + `pre_crop_margin_px=1`). Writes
     `image_embeddings_B_random_pooled.npy`, `image_embeddings_B_global.npy`,
     and `image_manifest_B.csv` (with the four new size columns). Prints a
     histogram of the tile sizes actually used at the end.
   - Step B: encodes every text column through LongCLIP-B's 248-token text
     tower (512-D per row per column, with a `columns_manifest.csv`).
   - Step C: builds the 2048-D fused teacher with
     `image_only = L2(0.5·random + 0.5·global)`.
   - Step D: writes `hdbscan_clusters_B.csv` with B v3.1 UMAP coordinates.
   - Smoke-test first: set `MAX_ROWS = 50, FORCE_RECOMPUTE = True`.

2. **Train the student** — `Colab/Student_Trainer_B.ipynb`:
   - Validates that the on-disk image/text embeddings are 512-D (fails fast
     on accidental B v2 / 768-D paths).
   - Loads LongCLIP-B via the same package loader as the teacher notebook.
   - Builds the 2048-D fused teacher, the `512 → 1024 → 1536 → 2048` student
     head, and runs `Z_embed → A_readout → B_last2 → C_last4/8/12 →
     D_full_decay`.
   - Default batch sizes (A100-80 GB, 17 adaptive views): `A_readout = 16`,
     `B_last2 ≤ 12`, `C_* ≤ 8`, `D_full_decay ≤ 8`. (Slightly tighter than
     fixed-tile B v3 because high-resolution images now produce up to 16 ×
     224×224 tiles plus the global view, which pushes peak per-sample memory
     a little higher.)
   - Saves best `.pth` as `vit_student_finetune_best_B_logitb.pth`. The
     run-config manifest records `pipeline_variant = "B_v3.1"` plus
     `tile_min_px`, `tile_max_px`, `tile_scale_ref_px`.

3. **Run student inference** — `Notebooks/Student_inference_embeddings_B.ipynb`:
   - Auto-discovers the best B-suffixed checkpoint, asserts
     `teacher_embed_dim == 2048` (fails fast on B v2 3072-D `.pth`).
   - Loads LongCLIP-B and runs the same 17-view geometry as the trainer
     (`pre_crop_margin_px=1`, short-side 224, 224×224 center crop, 16 random
     scale-adaptive native tiles with the same `TILE_MIN_PX/TILE_MAX_PX/TILE_SCALE_REF_PX`).
   - Default batch sizes: `INFER_BATCH_SIZE = 16`, `VISUAL_ENCODE_CHUNK_SIZE
     = 32`, `DATALOADER_NUM_WORKERS = 4` on CUDA.
   - Writes `student_image_emb_512_B.npy`, `student_z_2048_B.npy`,
     `student_logits_B.npy`, then drives UMAP / HDBSCAN / FAISS / Bayesian /
     generative-relabel + the K-SVD section.
   - **Section 8 — K-SVD dictionary learning**: standardizes the 2048-D
     student `z`, fits an approximate K-SVD dictionary (`N_ATOMS = 400`,
     `KSVD_OUTER_ITERS = 30`, `OMP n_nonzero = 30`), sparse-codes every row,
     and writes per-atom diagnostics (block-energy footprint, logit
     footprint via `student.classifier`, and a Bayesian usage-weighted
     posterior over the manifest metadata).

## Tuning the tile-size knob

The defaults (`TILE_MIN_PX = 16`, `TILE_MAX_PX = 224`,
`TILE_SCALE_REF_PX = 3000`) are calibrated for the current NEHM dataset's
size distribution (median geo-mean ~500 px, 99th percentile ~3020 px). You
might want to change them in two situations:

- **You add a new high-res source above ~3000 px.** Increase
  `TILE_SCALE_REF_PX` to the new top-of-distribution geo-mean side (e.g.
  4000 if you start ingesting 4000-px Leica Z-stacks). This keeps the new
  source from saturating instantly at 224 — otherwise its tiles are no
  longer informative about the increased resolution.
- **You want every tile to be the same fixed pixel size again** (matches
  Pipeline A / B v3 exactly). Set the trainer / inference `TILE_PX` (or the
  teacher `B_TILE_FIXED_PX`) to a positive integer. The other adaptive knobs
  are then ignored. Useful for ablation runs.

The teacher's `image_manifest_B.csv` records `tile_px_used` per row, so you
can post-hoc analyze the actual tile distribution. The teacher cell prints a
histogram (binned to 16 px) at the end of Step A — eyeball it after the
first run; you should see a roughly tri-modal distribution (16, ~32-64, 224).

## Cost & smoke-testing tips

- **Teacher image regen** is now ~17 ViT-B/16 forwards per row, but the
  forward cost grows roughly linearly with the chosen tile size (because
  CLIP's `preprocess` upsamples every tile to 224×224 before the ViT — so
  the cost of a 16×16 tile and a 224×224 tile is the same on the GPU; the
  difference is purely the per-tile *PIL resize*). Plan a few minutes for
  the full DB on A100. With `MAX_ROWS = 50` it should finish in <1 minute.
- **Trainer** at default settings runs 17 views × ViT-B/16 per sample with
  `BATCH_SIZE = 16` for `A_readout`. Drop `BATCH_SIZE` to 12 if you OOM on
  hi-res-heavy minibatches.
- **Inference** at default settings is `INFER_BATCH_SIZE = 16` on CUDA,
  `VISUAL_ENCODE_CHUNK_SIZE = 32`, `DATALOADER_NUM_WORKERS = 4`.
- **K-SVD** at `N_ATOMS = 400` × 30 outer iters × OMP-30 sparse coding takes
  ~minutes on CPU for ~12k rows. For a smoke test, set `N_ATOMS = 64,
  KSVD_OUTER_ITERS = 5, SUBSAMPLE_FRACTION = 0.5`.
- If you are migrating mid-run from a B v3 checkpoint: the `.pth` head shape
  is unchanged, but the random-pool teacher target shifted, so retrain from
  `Z_embed`. If you are migrating from a B v2 checkpoint: the `.pth` is
  *not* compatible at all (the head saw a 3072-D teacher).

## Parity self-check (paste this into a fresh cell after teacher regen)

```python
import numpy as np
from PIL import Image
from nehm_pipeline.longclip_image_embeddings import LongClipBImageEmbedConfig
from nehm_pipeline.preprocess_image import (
    compute_adaptive_tile_px,
    short_side_resize_center_crop,
)

TEACHER = LongClipBImageEmbedConfig()
assert TEACHER.n_random_crops == 16,           "B v3.1 should have 16 random tiles"
assert TEACHER.random_crop_px == 0,            "B v3.1 should use scale-adaptive sizing (random_crop_px=0)"
assert TEACHER.tile_min_px == 16,              "B v3.1 default min tile = 16px"
assert TEACHER.tile_max_px == 224,             "B v3.1 default max tile = 224px"
assert TEACHER.tile_scale_ref_px == 3000,      "B v3.1 default scale ref = 3000px"
assert TEACHER.global_target_px == 224,        "B v3.1 global view is 224x224"
assert TEACHER.pre_crop_margin_px == 1,        "B v3.1 should drop a 1px scan border"

# Adaptive sizing sanity checks
assert compute_adaptive_tile_px((150, 150)) == 16,  "150x150 should clamp to tile_min_px=16"
assert compute_adaptive_tile_px((300, 300)) == 22,  "300x300 → 224 * 300/3000 = 22.4 → 22"
assert compute_adaptive_tile_px((1500, 1500)) == 112, "1500x1500 → 224 * 0.5 = 112"
assert compute_adaptive_tile_px((3000, 3000)) == 224, "3000x3000 hits tile_max_px=224"
assert compute_adaptive_tile_px((9999, 9999)) == 224, "Anything bigger clamps to tile_max_px=224"

# Round-trip the global view
img = Image.open(next((PROJECT_ROOT / "NEHM").iterdir())).convert("RGB")
g0 = short_side_resize_center_crop(img, 224)
g1 = short_side_resize_center_crop(img, 224, pre_crop_margin_px=1)
assert g0.size == (224, 224) and g1.size == (224, 224)
assert not np.array_equal(np.asarray(g0), np.asarray(g1)), \
    "trim=1 must produce a slightly different crop on a normal-sized image"

print("B v3.1 (LongCLIP-B + scale-adaptive 16+1 multiview + trim) preprocessing parity OK.")
```

## Paper-text corrections to apply when you next edit the manuscript

- "Pipeline A's 16 random 16×16 native tiles" / "fixed 16×16 tile pool" →
  *"16 random scale-adaptive native tiles per image, with per-image side
  `tile_px = clip(round(224 × sqrt(W·H) / 3000), 16, 224)`. The largest
  images in the dataset (~3000-pixel geometric-mean side) get full 224×224
  tiles (the entire LongCLIP-B input), the smallest images (~150-pixel side)
  saturate at 16×16 tiles (one ViT-B/16 patch token), and intermediate
  resolutions are sized so each tile samples a roughly constant fraction
  (~0.5%) of the image's physical field of view."*
- "fixed 16-pixel patch size on tiles" → **remove**. Replace with: *"the
  tile size scales with image resolution as described above; CLIP's
  `preprocess` then upsamples every tile to 224×224 before the ViT, so the
  network always sees a fixed-resolution input and the variation is purely
  in *what fraction of the FOV that input was computed from*."*
- "fused teacher dim 2048 = 512 image + 3 × 512 text" — unchanged.
- "Student head 512 → 1024 → 1536 → 2048" — unchanged.
- "ImageNet statistics for normalization" → **"no per-channel normalization
  (linear RGB in [0, 1])"** — unchanged.
- Add a sentence on the 1-pixel edge trim — unchanged from B v3.
- Add a sentence on dictionary learning — unchanged from B v3.
