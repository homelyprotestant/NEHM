# NEHM — Pipeline B v3.1

Code for regenerating LongCLIP-B teacher embeddings and running student
inference (FAISS neighbors, dictionary learning).

**This repository does not include images or trained weights.** Place those on
Google Drive (or any local mirror) using the layout below.

## What is included

| Path | Purpose |
|------|---------|
| `Notebooks/Teacher_inference_embeddings_B.ipynb` | Teacher embedding regeneration |
| `Notebooks/Student_inference_embeddings_B.ipynb` | Student inference + FAISS / K-SVD |
| `nehm_pipeline/` | Shared encode / student utilities |
| `LongCLIP/model/` | LongCLIP source + BPE vocab (no checkpoints) |
| `Database/Material_Database.json` | Full Sheet2 material database (11,095 rows) |
| `Database/schema_sample.csv` | Sheet2 column schema (3 example rows) |
| `README_B.md` | Full Pipeline B v3.1 contract |

## Google Drive layout (expected)

```text
NEHM/                                      # microscopy images (0001.jpg …)
LongCLIP/checkpoints/longclip-B.pt         # LongCLIP-B weights
LongCLIP_Embeddings_v1/output_random_local_global_B/   # teacher npys
NEHM_RESULTS/
  vit_student_finetune_best_B_logitb.pth   # trained student
  student_inference_B/
    student_z_2304_B.npy
```

The material database ships in-repo as `Database/Material_Database.json`
(`sheet_name`, `columns`, `row_count`, `records`). Notebooks that still expect
`Database/Material_Database.xlsx` can rebuild it with pandas:

```bash
python - <<'PY'
import json, pandas as pd
from pathlib import Path
data = json.loads(Path('Database/Material_Database.json').read_text())
pd.DataFrame(data['records']).to_excel(
    'Database/Material_Database.xlsx',
    sheet_name=data['sheet_name'],
    index=False,
)
PY
```

## Setup

```bash
pip install -r requirements-nehm.txt
```

Put `longclip-B.pt` and `NEHM/` on disk / Drive, then:

1. `Notebooks/Teacher_inference_embeddings_B.ipynb` — regenerate teacher embeddings
2. `Notebooks/Student_inference_embeddings_B.ipynb` — student inference / FAISS / K-SVD

See `README_B.md` for the scale-adaptive 17-view contract. The live B stack
fuses a 768-D image (`ln_post`) with 3×512 text into a **2304-D** teacher;
prefer the notebooks over older 2048-D prose in docs.
