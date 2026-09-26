#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

require_command gcloud

DELETE_DATA=false
ASSUME_YES=false
for argument in "$@"; do
  case "${argument}" in
    --delete-data) DELETE_DATA=true ;;
    --yes) ASSUME_YES=true ;;
    *)
      printf 'Usage: %s [--delete-data] [--yes]\n' "$0" >&2
      exit 2
      ;;
  esac
done

if [[ "${ASSUME_YES}" != "true" ]]; then
  printf 'Delete VM %s in project %s? Type the instance name: ' \
    "${INSTANCE_NAME}" "${GCP_PROJECT}"
  read -r confirmation
  [[ "${confirmation}" == "${INSTANCE_NAME}" ]] || {
    printf 'Cancelled.\n'
    exit 1
  }
fi

if instance_exists; then
  "${GCLOUD[@]}" "${COMPUTE[@]}" instances delete "${INSTANCE_NAME}" \
    --zone "${GCP_ZONE}"
fi
if "${GCLOUD[@]}" "${COMPUTE[@]}" firewall-rules describe "${FIREWALL_RULE}" \
  >/dev/null 2>&1; then
  "${GCLOUD[@]}" "${COMPUTE[@]}" firewall-rules delete "${FIREWALL_RULE}"
fi
if "${GCLOUD[@]}" "${COMPUTE[@]}" addresses describe "${STATIC_IP_NAME}" \
  --region "${GCP_REGION}" >/dev/null 2>&1; then
  "${GCLOUD[@]}" "${COMPUTE[@]}" addresses delete "${STATIC_IP_NAME}" \
    --region "${GCP_REGION}"
fi

if [[ "${DELETE_DATA}" == "true" ]] && disk_exists; then
  "${GCLOUD[@]}" "${COMPUTE[@]}" disks delete "${DATA_DISK_NAME}" \
    --zone "${GCP_ZONE}"
  printf 'VM and persistent data disk deleted.\n'
else
  printf 'VM deleted. Persistent disk %s was retained.\n' "${DATA_DISK_NAME}"
fi
