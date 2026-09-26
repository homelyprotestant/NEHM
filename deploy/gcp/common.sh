#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

if [[ -f "${SCRIPT_DIR}/deploy.env" ]]; then
  # shellcheck disable=SC1091
  source "${SCRIPT_DIR}/deploy.env"
fi

: "${GCP_PROJECT:=visual-history-lab}"
: "${GCP_REGION:=us-east1}"
: "${GCP_ZONE:=us-east1-c}"
: "${INSTANCE_NAME:=plm-agent-gpu}"
: "${STATIC_IP_NAME:=plm-agent-web-ip}"
: "${FIREWALL_RULE:=plm-agent-web}"
: "${NETWORK_TAG:=plm-agent-web}"
: "${MACHINE_TYPE:=e2-standard-8}"
: "${ACCELERATOR:=cpu}"
: "${BOOT_DISK_SIZE:=100GB}"
: "${DATA_DISK_SIZE:=100GB}"
: "${DATA_DISK_NAME:=plm-agent-data}"
: "${SOURCE_IMAGE_FAMILY:=ubuntu-2204-lts}"
: "${SOURCE_IMAGE_PROJECT:=ubuntu-os-cloud}"
: "${REMOTE_ROOT:=/opt/plm-agent}"
: "${PERSIST_ROOT:=/var/lib/plm-agent}"
: "${APP_BIND_ADDRESS:=0.0.0.0}"
: "${APP_PORT:=8000}"
: "${SSH_USER:=}"

GCLOUD=(gcloud --quiet)
COMPUTE=(compute --project "${GCP_PROJECT}")
SSH_TARGET="${INSTANCE_NAME}"
if [[ -n "${SSH_USER}" ]]; then
  SSH_TARGET="${SSH_USER}@${INSTANCE_NAME}"
fi

require_command() {
  command -v "$1" >/dev/null 2>&1 || {
    printf 'Required command not found: %s\n' "$1" >&2
    exit 1
  }
}

instance_exists() {
  "${GCLOUD[@]}" "${COMPUTE[@]}" instances describe "${INSTANCE_NAME}" \
    --zone "${GCP_ZONE}" >/dev/null 2>&1
}

disk_exists() {
  "${GCLOUD[@]}" "${COMPUTE[@]}" disks describe "${DATA_DISK_NAME}" \
    --zone "${GCP_ZONE}" >/dev/null 2>&1
}

remote() {
  "${GCLOUD[@]}" "${COMPUTE[@]}" ssh "${SSH_TARGET}" \
    --zone "${GCP_ZONE}" --command "$1"
}

copy_to_vm() {
  "${GCLOUD[@]}" "${COMPUTE[@]}" scp \
    --zone "${GCP_ZONE}" "$1" "${SSH_TARGET}:$2"
}

wait_for_ssh() {
  local attempt
  for attempt in $(seq 1 60); do
    if remote "true" >/dev/null 2>&1; then
      return 0
    fi
    sleep 10
  done
  printf 'Timed out waiting for SSH on %s\n' "${INSTANCE_NAME}" >&2
  return 1
}

wait_for_host_setup() {
  local attempt
  for attempt in $(seq 1 90); do
    if remote "sudo test -f $(printf '%q' "${PERSIST_ROOT}/.host-ready")" \
      >/dev/null 2>&1; then
      return 0
    fi
    sleep 10
  done
  printf 'Timed out waiting for Docker/GPU host setup.\n' >&2
  return 1
}

compose_command() {
  local files="-f deploy/gcp/docker-compose.yml"
  if [[ "${ACCELERATOR}" == "gpu" ]]; then
    files="${files} -f deploy/gcp/docker-compose.gpu.yml"
  fi
  printf 'cd %q && sudo docker compose --env-file %q %s' \
    "${REMOTE_ROOT}/app" "${REMOTE_ROOT}/runtime.env" "${files}"
}

write_runtime_env() {
  local content
  printf -v content \
    'PERSIST_ROOT=%s\nARTIFACT_ROOT=%s\nAPP_BIND_ADDRESS=%s\nAPP_PORT=%s\nSECRETS_FILE=%s\nACCELERATOR=%s\n' \
    "${PERSIST_ROOT}" "${PERSIST_ROOT}/artifacts" "${APP_BIND_ADDRESS}" \
    "${APP_PORT}" "${REMOTE_ROOT}/secrets/app.env" "${ACCELERATOR}"
  remote "printf %s $(printf '%q' "${content}") | sudo tee $(printf '%q' "${REMOTE_ROOT}/runtime.env") >/dev/null && sudo chmod 0644 $(printf '%q' "${REMOTE_ROOT}/runtime.env")"
}
