#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

require_command gcloud

if [[ "$("${GCLOUD[@]}" billing projects describe "${GCP_PROJECT}" \
  --format='value(billingEnabled)')" != "True" ]]; then
  printf 'Project %s does not have billing enabled.\n' "${GCP_PROJECT}" >&2
  exit 1
fi
"${GCLOUD[@]}" services enable compute.googleapis.com --project "${GCP_PROJECT}"

if ! "${GCLOUD[@]}" "${COMPUTE[@]}" addresses describe "${STATIC_IP_NAME}" \
  --region "${GCP_REGION}" >/dev/null 2>&1; then
  "${GCLOUD[@]}" "${COMPUTE[@]}" addresses create "${STATIC_IP_NAME}" \
    --region "${GCP_REGION}"
fi
STATIC_IP="$("${GCLOUD[@]}" "${COMPUTE[@]}" addresses describe "${STATIC_IP_NAME}" \
  --region "${GCP_REGION}" --format='value(address)')"

if ! "${GCLOUD[@]}" "${COMPUTE[@]}" firewall-rules describe "${FIREWALL_RULE}" \
  >/dev/null 2>&1; then
  "${GCLOUD[@]}" "${COMPUTE[@]}" firewall-rules create "${FIREWALL_RULE}" \
    --allow "tcp:${APP_PORT}" \
    --direction INGRESS \
    --source-ranges "0.0.0.0/0" \
    --target-tags "${NETWORK_TAG}"
fi

if ! disk_exists; then
  "${GCLOUD[@]}" "${COMPUTE[@]}" disks create "${DATA_DISK_NAME}" \
    --zone "${GCP_ZONE}" \
    --size "${DATA_DISK_SIZE}" \
    --type pd-balanced
fi

if instance_exists; then
  CURRENT_MACHINE="$("${GCLOUD[@]}" "${COMPUTE[@]}" instances describe \
    "${INSTANCE_NAME}" --zone "${GCP_ZONE}" \
    --format='value(machineType.basename())')"
  if [[ "${CURRENT_MACHINE}" != "${MACHINE_TYPE}" ]]; then
    printf 'Existing VM uses %s, expected %s.\n' \
      "${CURRENT_MACHINE}" "${MACHINE_TYPE}" >&2
    exit 1
  fi
else
  MAINTENANCE_ARGS=(--restart-on-failure)
  if [[ "${ACCELERATOR}" == "gpu" ]]; then
    MAINTENANCE_ARGS+=(--maintenance-policy TERMINATE)
  fi
  "${GCLOUD[@]}" "${COMPUTE[@]}" instances create "${INSTANCE_NAME}" \
    --zone "${GCP_ZONE}" \
    --machine-type "${MACHINE_TYPE}" \
    "${MAINTENANCE_ARGS[@]}" \
    --boot-disk-size "${BOOT_DISK_SIZE}" \
    --boot-disk-type pd-balanced \
    --image-family "${SOURCE_IMAGE_FAMILY}" \
    --image-project "${SOURCE_IMAGE_PROJECT}" \
    --address "${STATIC_IP}" \
    --tags "${NETWORK_TAG}" \
    --disk "name=${DATA_DISK_NAME},device-name=plm-agent-data,mode=rw,boot=no,auto-delete=no" \
    --metadata "plm-accelerator=${ACCELERATOR}" \
    --metadata-from-file "startup-script=${SCRIPT_DIR}/startup.sh"
fi

wait_for_ssh
wait_for_host_setup
remote "sudo install -d -m 0755 $(printf '%q' "${REMOTE_ROOT}") $(printf '%q' "${REMOTE_ROOT}/secrets")"

if remote "sudo test -s $(printf '%q' "${REMOTE_ROOT}/secrets/app.env")"; then
  printf 'Retained existing VM-side app.env.\n'
elif [[ -f "${SCRIPT_DIR}/app.env" ]]; then
  copy_to_vm "${SCRIPT_DIR}/app.env" "/tmp/plm-agent-app.env"
  remote "sudo install -o root -g root -m 0600 /tmp/plm-agent-app.env $(printf '%q' "${REMOTE_ROOT}/secrets/app.env") && rm -f /tmp/plm-agent-app.env"
  printf 'Installed app.env separately from application source.\n'
else
  remote "sudo install -o root -g root -m 0600 /dev/null $(printf '%q' "${REMOTE_ROOT}/secrets/app.env")"
  printf 'Created empty VM-side app.env; add OPENAI_API_KEY before starting.\n'
fi

write_runtime_env
printf 'VM is ready at http://%s:%s/.\n' "${STATIC_IP}" "${APP_PORT}"
printf 'Run transfer-artifacts.sh, then deploy.sh.\n'
