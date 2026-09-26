# NEHM Stable — Pipeline B v3.1

Code for retraining the LongCLIP-B multiview student and running particle /
fiber microscopy inference (FAISS neighbors, global dictionary atoms, sample
descriptions).

**This repository does not include images or trained weights.** Place those on
Google Drive (or any local mirror) using the layout below.

## What is included

| Path | Purpose |
|------|---------|
| `Colab/Student_Trainer_B.ipynb` | Production student trainer (Pipeline B) |
| `Notebooks/Teacher_inference_embeddings_B.ipynb` | Teacher embedding regeneration |
| `Notebooks/Student_inference_embeddings_B.ipynb` | Student inference + FAISS / K-SVD |
| `Notebooks/NEHM_Global_Dictionary_Atom_Descriptions.ipynb` | Global dictionary + atom labels |
| `nehm_pipeline/` | Shared training / encode utilities |
| `LongCLIP/model/` | LongCLIP source + BPE vocab (no checkpoints) |
| `PLM_Agent/` | Dictionary interpretability + upload inference API |
| `Database/schema_sample.csv` | Sheet2 column schema (3 example rows) |
| `README_B.md` | Full Pipeline B v3.1 contract |

## Google Drive layout (expected)

Mirror these paths under your Drive project root (or set `PLM_PROJECT_ROOT`):

```text
NEHM/                                      # microscopy images (0001.jpg …)
Database/Material_Database.xlsx            # full Sheet2 workbook
LongCLIP/checkpoints/longclip-B.pt         # LongCLIP-B weights
LongCLIP_Embeddings_v1/output_random_local_global_B/   # teacher npys
NEHM_RESULTS/
  vit_student_finetune_best_B_logitb.pth   # trained student
  student_inference_B/
    student_z_2304_B.npy                   # or student_z_2304_B_current_images.npy
    global_dictionary_interpretability/    # atoms, codes, labels, config
```

Upload the trained model and `NEHM/` images to Drive yourself; keep this git
repo for source only.

## Retrain (Pipeline B)

1. Install deps from the project root:

   ```bash
   pip install -r requirements-nehm.txt
   pip install -r Colab/requirements-vit-finetune.txt
   ```

2. Put `longclip-B.pt`, `NEHM/`, and `Material_Database.xlsx` on disk / Drive.

3. Run `Notebooks/Teacher_inference_embeddings_B.ipynb` to regenerate teacher
   embeddings under `LongCLIP_Embeddings_v1/output_random_local_global_B/`.

4. Run `Colab/Student_Trainer_B.ipynb` (Colab + Drive mount via `colab_env.py`).

5. Run `Notebooks/Student_inference_embeddings_B.ipynb` to export student
   embeddings / FAISS corpus.

6. Optionally run `NEHM_Global_Dictionary_Atom_Descriptions.ipynb` to build the
   global dictionary and atom microscopy labels used by PLM.

See `README_B.md` for the scale-adaptive 17-view contract and dimension notes.
The live B stack fuses a 768-D image (`ln_post`) with 3×512 text into a
**2304-D** teacher; prefer the notebooks over older 2048-D prose in docs.

## Inference (notebooks / Python)

```bash
cp PLM_Agent/.env.example PLM_Agent/.env   # set OPENAI_API_KEY when using GPT
pip install -r PLM_Agent/requirements.txt
```

Use `Notebooks/Student_inference_embeddings_B.ipynb` and
`Notebooks/NEHM_Global_Dictionary_Atom_Descriptions.ipynb`, or call
`PLM_Agent.upload_inference.PLMUploadInference` from Python. Required artifacts
are listed in `PLM_Agent/README.md`.

## Tests

```bash
pip install -r PLM_Agent/requirements.txt
pytest PLM_Agent/tests -q
```

## Secrets

Never commit `PLM_Agent/.env`. Use `.env.example` as the template only.
