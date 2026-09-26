FROM python:3.11-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PLM_PROJECT_ROOT=/app \
    PLM_WEB_DATA_DIR=/data \
    PLM_WEB_STATIC_DIR=/app/PLM_Agent/web/static \
    PLM_WEB_PUBLIC_PROTOTYPE=true \
    PLM_WEB_HOST=0.0.0.0 \
    PLM_WEB_PORT=8000

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY PLM_Agent/requirements-web.txt /tmp/requirements-web.txt
RUN python -c "from pathlib import Path; p=Path('/tmp/requirements-web.txt'); p.write_text(''.join(line for line in p.read_text().splitlines(keepends=True) if not line.startswith(('torch==', 'torchvision=='))))" \
    && python -m pip install --upgrade pip \
    && python -m pip install \
        --index-url https://download.pytorch.org/whl/cpu \
        torch==2.8.0 torchvision==0.23.0 \
    && python -m pip install -r /tmp/requirements-web.txt

COPY PLM_Agent /app/PLM_Agent
COPY nehm_pipeline /app/nehm_pipeline
COPY LongCLIP/model /app/LongCLIP/model

RUN useradd --create-home --uid 10001 plm \
    && mkdir -p \
        /data \
        /app/Database \
        /app/LongCLIP/checkpoints \
        /app/NEHM_RESULTS/student_inference_B/global_dictionary_interpretability \
    && chown -R 10001:10001 \
        /data \
        /app/PLM_Agent \
        /app/nehm_pipeline \
        /app/LongCLIP/model

USER 10001:10001

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=5 \
    CMD python3 -c "import json,urllib.request; assert json.load(urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=5))['status'] == 'ok'"

CMD ["python3", "-m", "uvicorn", "PLM_Agent.web.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--no-access-log", "--proxy-headers", "--forwarded-allow-ips", "*"]
