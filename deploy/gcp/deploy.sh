#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

require_command gcloud
require_command tar

instance_exists || {
  printf 'Instance %s does not exist; run setup.sh first.\n' "${INSTANCE_NAME}" >&2
  exit 1
}
wait_for_ssh
wait_for_host_setup

remote "sudo test -f $(printf '%q' "${PERSIST_ROOT}/artifacts/Database/Material_Database.xlsx") && sudo test -f $(printf '%q' "${PERSIST_ROOT}/artifacts/NEHM_RESULTS/student_inference_B/student_z_2304_B_current_images.npy") && sudo test -f $(printf '%q' "${PERSIST_ROOT}/artifacts/LongCLIP/checkpoints/longclip-B.pt")" || {
  printf 'Runtime artifacts are missing; run transfer-artifacts.sh first.\n' >&2
  exit 1
}
remote "sudo rm -rf $(printf '%q' "${REMOTE_ROOT}/app.upload") && sudo install -d -m 0755 $(printf '%q' "${REMOTE_ROOT}/app.upload")"

# Stream an explicit source allowlist. Secrets, generated outputs, corpora,
# notebooks, and unrelated project trees never enter the code archive.
tar \
  --exclude='*/.env' \
  --exclude='*/.env.*' \
  --exclude='*/.venv' \
  --exclude='*/venv' \
  --exclude='*/__pycache__' \
  --exclude='deploy/gcp/app.env' \
  --exclude='deploy/gcp/deploy.env' \
  -C "${PROJECT_ROOT}" -czf - \
  Dockerfile .dockerignore PLM_Agent nehm_pipeline LongCLIP/model deploy/gcp \
  | "${GCLOUD[@]}" "${COMPUTE[@]}" ssh "${SSH_TARGET}" \
      --zone "${GCP_ZONE}" \
      --command "sudo tar -xzf - -C $(printf '%q' "${REMOTE_ROOT}/app.upload")"

remote "sudo rm -rf $(printf '%q' "${REMOTE_ROOT}/app.previous"); if sudo test -d $(printf '%q' "${REMOTE_ROOT}/app"); then sudo mv $(printf '%q' "${REMOTE_ROOT}/app") $(printf '%q' "${REMOTE_ROOT}/app.previous"); fi; sudo mv $(printf '%q' "${REMOTE_ROOT}/app.upload") $(printf '%q' "${REMOTE_ROOT}/app")"
write_runtime_env
COMPOSE="$(compose_command)"
remote "${COMPOSE} config --quiet"
remote "${COMPOSE} up -d --build --remove-orphans"
remote "sudo rm -rf $(printf '%q' "${REMOTE_ROOT}/app.previous")"

remote "for attempt in \$(seq 1 180); do if curl --fail --silent http://127.0.0.1:$(printf '%q' "${APP_PORT}")/api/health >/dev/null; then exit 0; fi; sleep 10; done; echo 'Application health check timed out.' >&2; ${COMPOSE} ps; exit 1"

STATIC_IP="$("${GCLOUD[@]}" "${COMPUTE[@]}" addresses describe "${STATIC_IP_NAME}" \
  --region "${GCP_REGION}" --format='value(address)')"
printf 'Deployment is healthy: http://%s:%s/\n' "${STATIC_IP}" "${APP_PORT}"
