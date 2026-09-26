# PLM_Agent

Self-contained particle and fiber microscopy interpretability pipeline used by
`Notebooks/NEHM_Global_Dictionary_Atom_Descriptions.ipynb`.

It includes dictionary learning, metadata chunking, local Qwen RCS, per-atom labeling,
FAISS retrieval, and arbitrary-sample descriptions. It has no runtime dependency on
VisualHistory and no Azure/HKU/Gemini routing.

## Models

- `gpt-5.1`: direct OpenAI API via `OPENAI_API_KEY`
- `qwen3.8:27b`: local Ollama text and vision
- `qwen2.5:7b`: local Ollama retrieval-context scoring

Copy `.env.example` to `.env` and populate `OPENAI_API_KEY` when GPT is required.

## Arbitrary-image inference

`upload_inference.PLMUploadInference` accepts a microscopy image that is not already
in the database. It reproduces Pipeline B v3.1 inference, searches the ten nearest
corpus neighbors, computes the image's global-dictionary atom code, runs direct
vision, and returns the same evidence streams as the final notebook cell.

Illumination and magnification are inferred from nearest-neighbor frequency and direct
vision.

### Required artifacts

- `LongCLIP/checkpoints/longclip-B.pt`
- `NEHM_RESULTS/vit_student_finetune_best_B_logitb.pth`
- `NEHM_RESULTS/student_inference_B/student_z_2304_B.npy`
  (or `student_z_2304_B_current_images.npy` when present)
- `Database/Material_Database.xlsx`
- `NEHM_RESULTS/student_inference_B/global_dictionary_interpretability/`
  - `global_dictionary_atoms.npy`
  - `global_embedding_mean.npy`
  - `global_sparse_codes.npy`
  - `global_ksvd_config.json`
  - `atom_microscopy_labels.json`

### Install

From the repository root:

```bash
pip install -r PLM_Agent/requirements.txt
cp PLM_Agent/.env.example PLM_Agent/.env
# Set OPENAI_API_KEY only when GPT-5.1 will be used.
```

Per-atom labels are loaded from the stable cache and are never rebuilt by upload
inference. If coordinator synthesis exhausts its retries, the deterministic
nearest-neighbor and vision fallback still returns a report.
