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

## Arbitrary-image web app

The local FastAPI application accepts a microscopy image that is not already in the
database. It reproduces Pipeline B v3.1 inference, searches the ten nearest corpus
neighbors, computes the image's global-dictionary atom code, runs direct vision, and
returns the same evidence streams as the final notebook cell.

Illumination and magnification are inferred from nearest-neighbor frequency and direct
vision. The upload form does not request specimen metadata.

### Required artifacts

The app checks these files at startup:

- `LongCLIP/checkpoints/longclip-B.pt`
- `NEHM_RESULTS/vit_student_finetune_best_B_logitb.pth`
- `NEHM_RESULTS/student_inference_B/student_z_2304_B.npy`
- `Database/Material_Database.xlsx`
- `NEHM_RESULTS/student_inference_B/global_dictionary_interpretability/`
  - `global_dictionary_atoms.npy`
  - `global_embedding_mean.npy`
  - `global_sparse_codes.npy`
  - `global_ksvd_config.json`
  - `atom_microscopy_labels.json`

### Install and run

From `NEHM_Stable`:

```bash
uv venv ~/.venvs/nehm-plm-web
uv pip install --python ~/.venvs/nehm-plm-web/bin/python -r PLM_Agent/requirements.txt
cp PLM_Agent/.env.example PLM_Agent/.env
# Set OPENAI_API_KEY only when GPT-5.1 will be selected.
~/.venvs/nehm-plm-web/bin/python -m uvicorn PLM_Agent.web.api:app \
  --host 127.0.0.1 --port 8080
```

Open [http://127.0.0.1:8080](http://127.0.0.1:8080).

The web app uses GPT-5.1 through the direct OpenAI API. `OPENAI_API_KEY` is read only by
the Python backend from the private `PLM_Agent/.env`; it is never sent to browser code.
Per-atom labels are loaded from the stable cache and are never rebuilt by the web app.

### Upload behavior

- Accepted formats: JPEG, PNG, WebP, TIFF, and BMP.
- Default upload limit: 25 MB.
- Images are normalized to RGB PNG before inference.
- One queued worker serializes GPU/MPS work.
- Completed reports are available as downloadable JSON.
- Temporary uploaded images are removed after processing.
- If coordinator synthesis exhausts its retries, the deterministic nearest-neighbor and
  vision fallback still returns a report.

### API

- `GET /api/health`
- `POST /api/jobs` — multipart fields `image` and `coordinator`
- `GET /api/jobs/{id}?token=...`
- `GET /api/jobs/{id}/events?token=...` — server-sent progress events
- `GET /api/jobs/{id}/result?token=...`
- `GET /api/jobs/{id}/download?token=...`
- `POST /api/jobs/{id}/cancel?token=...`

## Public GCP deployment

The production prototype follows the same Compute Engine pattern as
`conservation_ai`: one dedicated VM, one FastAPI container, a static public IP,
and a persistent data disk. It exposes HTTP on port 8000; domain and TLS
termination are not included. It currently uses CPU inference because the
project-wide GPU quota is consumed by the conservation deployment; measured
embedding latency remains below the OpenAI request latency.

Prerequisites:

- An authenticated `gcloud` CLI with access to project `visual-history-lab`.
- Compute Engine capacity for `e2-standard-8` in `us-east1-c`. Set
  `ACCELERATOR=gpu` and select a G2 machine only after increasing the project's
  global GPU quota.
- A replacement OpenAI key stored only in `deploy/gcp/app.env`.
- Docker is installed automatically on the VM.

Configure and deploy from `NEHM_Stable`:

```bash
cp deploy/gcp/deploy.env.example deploy/gcp/deploy.env
cp deploy/gcp/app.env.example deploy/gcp/app.env
# Add OPENAI_API_KEY to deploy/gcp/app.env. Never commit this file.

deploy/gcp/setup.sh
deploy/gcp/transfer-artifacts.sh
deploy/gcp/deploy.sh
deploy/gcp/status.sh
```

Routine operations:

```bash
deploy/gcp/update.sh       # deploy code without retransferring model artifacts
deploy/gcp/stop.sh         # stop the VM and GPU billing
deploy/gcp/start.sh        # start the VM and app
deploy/gcp/status.sh       # URL, GPU status, containers, and health
deploy/gcp/teardown.sh     # remove compute resources; preserves data by default
```

The public prototype intentionally has no daily request cap. Queue size, upload
size, image pixels, OpenAI request timeout, and total job duration remain bounded.
An uncapped GPT-5.1 endpoint can incur unbounded API charges or be abused. Configure
a Google Cloud budget alert, configure OpenAI project budget notifications, monitor
usage, and use `deploy/gcp/stop.sh` as the emergency stop.

Uploaded images are normalized locally, sent to OpenAI for vision and synthesis,
then deleted by the worker. Results are temporary and are removed after their TTL.
The VM-side secret file is stored under `/opt/plm-agent/secrets/`; it is excluded
from Docker builds and code synchronization.
