#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

require_command gcloud
instance_exists || {
  printf 'Instance %s does not exist.\n' "${INSTANCE_NAME}" >&2
  exit 1
}

STATE="$("${GCLOUD[@]}" "${COMPUTE[@]}" instances describe "${INSTANCE_NAME}" \
  --zone "${GCP_ZONE}" --format='value(status)')"
if [[ "${STATE}" != "RUNNING" ]]; then
  "${GCLOUD[@]}" "${COMPUTE[@]}" instances start "${INSTANCE_NAME}" \
    --zone "${GCP_ZONE}"
fi

wait_for_ssh
wait_for_host_setup
COMPOSE="$(compose_command)"
remote "${COMPOSE} up -d"
remote "for attempt in \$(seq 1 180); do if curl --fail --silent http://127.0.0.1:$(printf '%q' "${APP_PORT}")/api/health >/dev/null; then exit 0; fi; sleep 10; done; echo 'Application health check timed out.' >&2; exit 1"
STATIC_IP="$("${GCLOUD[@]}" "${COMPUTE[@]}" addresses describe "${STATIC_IP_NAME}" \
  --region "${GCP_REGION}" --format='value(address)')"
printf 'Application is healthy: http://%s:%s/\n' "${STATIC_IP}" "${APP_PORT}"
