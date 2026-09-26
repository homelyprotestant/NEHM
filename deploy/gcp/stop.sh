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
if [[ "${STATE}" == "RUNNING" ]]; then
  wait_for_ssh
  COMPOSE="$(compose_command)"
  remote "${COMPOSE} stop"
  "${GCLOUD[@]}" "${COMPUTE[@]}" instances stop "${INSTANCE_NAME}" \
    --zone "${GCP_ZONE}"
fi

printf 'VM is stopped; persistent web data and artifacts are retained.\n'
