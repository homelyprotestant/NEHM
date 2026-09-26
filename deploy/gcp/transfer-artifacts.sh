#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

require_command gcloud
require_command tar
require_command shasum

# Keep this list synchronized with UploadArtifacts.required_paths(). The
# current-images corpus is intentional; the older student_z_2304_B.npy index
# is not row-aligned with the current Material_Database.xlsx.
ARTIFACTS=(
  "Database/Material_Database.xlsx"
  "NEHM_RESULTS/student_inference_B/student_z_2304_B_current_images.npy"
  "NEHM_RESULTS/student_inference_B/global_dictionary_interpretability/global_sparse_codes.npy"
  "NEHM_RESULTS/student_inference_B/global_dictionary_interpretability/global_dictionary_atoms.npy"
  "NEHM_RESULTS/student_inference_B/global_dictionary_interpretability/global_embedding_mean.npy"
  "NEHM_RESULTS/student_inference_B/global_dictionary_interpretability/global_ksvd_config.json"
  "NEHM_RESULTS/student_inference_B/global_dictionary_interpretability/atom_microscopy_labels.json"
  "NEHM_RESULTS/vit_student_finetune_best_B_logitb.pth"
  "LongCLIP/checkpoints/longclip-B.pt"
)

for relative_path in "${ARTIFACTS[@]}"; do
  if [[ ! -f "${PROJECT_ROOT}/${relative_path}" ]]; then
    printf 'Missing required runtime artifact: %s\n' \
      "${PROJECT_ROOT}/${relative_path}" >&2
    exit 1
  fi
done
instance_exists || {
  printf 'Instance %s does not exist; run setup.sh first.\n' "${INSTANCE_NAME}" >&2
  exit 1
}

ARCHIVE="$(mktemp "${TMPDIR:-/tmp}/plm-artifacts.XXXXXX.tar.gz")"
MANIFEST="$(mktemp "${TMPDIR:-/tmp}/plm-artifacts.XXXXXX.sha256")"
trap 'rm -f "${ARCHIVE}" "${MANIFEST}"' EXIT
(cd "${PROJECT_ROOT}" && shasum -a 256 "${ARTIFACTS[@]}") > "${MANIFEST}"
tar -C "${PROJECT_ROOT}" -czf "${ARCHIVE}" "${ARTIFACTS[@]}"

wait_for_ssh
"${GCLOUD[@]}" "${COMPUTE[@]}" scp --zone "${GCP_ZONE}" \
  "${ARCHIVE}" "${SSH_TARGET}:/tmp/plm-artifacts.tar.gz"
"${GCLOUD[@]}" "${COMPUTE[@]}" scp --zone "${GCP_ZONE}" \
  "${MANIFEST}" "${SSH_TARGET}:/tmp/plm-artifacts.sha256"

COMPOSE="$(compose_command)"
HAS_APPLICATION=false
if remote "sudo test -f $(printf '%q' "${REMOTE_ROOT}/app/deploy/gcp/docker-compose.yml")"; then
  HAS_APPLICATION=true
  remote "${COMPOSE} stop app || true"
fi

remote "set -euo pipefail; upload=$(printf '%q' "${PERSIST_ROOT}/artifacts.upload"); current=$(printf '%q' "${PERSIST_ROOT}/artifacts"); previous=$(printf '%q' "${PERSIST_ROOT}/artifacts.previous"); sudo rm -rf \"\${upload}\" \"\${previous}\"; sudo install -d -m 0755 \"\${upload}\"; sudo tar -xzf /tmp/plm-artifacts.tar.gz -C \"\${upload}\"; (cd \"\${upload}\" && sudo sha256sum --check /tmp/plm-artifacts.sha256); sudo chmod -R a+rX \"\${upload}\"; sudo test -f \"\${upload}/Database/Material_Database.xlsx\"; sudo test -f \"\${upload}/NEHM_RESULTS/student_inference_B/student_z_2304_B_current_images.npy\"; if sudo test -d \"\${current}\"; then sudo mv \"\${current}\" \"\${previous}\"; fi; sudo mv \"\${upload}\" \"\${current}\"; sudo rm -rf \"\${previous}\"; rm -f /tmp/plm-artifacts.tar.gz /tmp/plm-artifacts.sha256"

if [[ "${HAS_APPLICATION}" == "true" ]]; then
  remote "${COMPOSE} up -d app"
  remote "for attempt in \$(seq 1 120); do if curl --fail --silent http://127.0.0.1:$(printf '%q' "${APP_PORT}")/api/health >/dev/null; then exit 0; fi; sleep 5; done; echo 'Health check failed after artifact transfer.' >&2; exit 1"
  printf 'Runtime artifacts transferred and application is healthy.\n'
else
  printf 'Runtime artifacts transferred. Run deploy.sh next.\n'
fi
