#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

require_command gcloud
if ! instance_exists; then
  printf 'Instance: absent\n'
  exit 1
fi

STATE="$("${GCLOUD[@]}" "${COMPUTE[@]}" instances describe "${INSTANCE_NAME}" \
  --zone "${GCP_ZONE}" --format='value(status)')"
printf 'Instance: %s\n' "${STATE}"
STATIC_IP="$("${GCLOUD[@]}" "${COMPUTE[@]}" addresses describe "${STATIC_IP_NAME}" \
  --region "${GCP_REGION}" --format='value(address)' 2>/dev/null || true)"
if [[ -n "${STATIC_IP}" ]]; then
  printf 'URL: http://%s:%s/\n' "${STATIC_IP}" "${APP_PORT}"
fi
if [[ "${STATE}" != "RUNNING" ]]; then
  exit 1
fi

wait_for_ssh
COMPOSE="$(compose_command)"
remote "if command -v nvidia-smi >/dev/null 2>&1; then nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader; else echo 'Accelerator: CPU'; fi; ${COMPOSE} ps; printf 'API health: '; curl --fail --silent http://127.0.0.1:$(printf '%q' "${APP_PORT}")/api/health; printf '\n'"
