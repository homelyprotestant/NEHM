#!/usr/bin/env bash
set -euo pipefail

export DEBIAN_FRONTEND=noninteractive

install -d -m 0755 /etc/apt/keyrings
apt-get update
apt-get install --no-install-recommends -y ca-certificates curl gnupg

if ! command -v docker >/dev/null 2>&1; then
  curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
    | gpg --batch --yes --dearmor -o /etc/apt/keyrings/docker.gpg
  chmod a+r /etc/apt/keyrings/docker.gpg
  # shellcheck disable=SC1091
  source /etc/os-release
  printf 'deb [arch=%s signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu %s stable\n' \
    "$(dpkg --print-architecture)" "${VERSION_CODENAME}" \
    > /etc/apt/sources.list.d/docker.list
  apt-get update
  apt-get install --no-install-recommends -y \
    docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
fi

ACCELERATOR="$(curl -fsS -H 'Metadata-Flavor: Google' \
  http://metadata.google.internal/computeMetadata/v1/instance/attributes/plm-accelerator \
  || printf 'cpu')"
if [[ "${ACCELERATOR}" == "gpu" ]]; then
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    install -d -m 0755 /opt/google/cuda-installer
    curl -fSsL \
      https://storage.googleapis.com/compute-gpu-installation-us/installer/latest/cuda_installer.pyz \
      -o /opt/google/cuda-installer/cuda_installer.pyz
    python3 /opt/google/cuda-installer/cuda_installer.pyz install_driver
  fi

  if ! dpkg-query -W nvidia-container-toolkit >/dev/null 2>&1; then
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
      | gpg --batch --yes --dearmor \
        -o /etc/apt/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -fsSL \
      https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
      | sed 's#deb https://#deb [signed-by=/etc/apt/keyrings/nvidia-container-toolkit-keyring.gpg] https://#' \
      > /etc/apt/sources.list.d/nvidia-container-toolkit.list
    apt-get update
    apt-get install --no-install-recommends -y nvidia-container-toolkit
  fi
  nvidia-ctk runtime configure --runtime=docker
fi

systemctl enable --now docker
systemctl restart docker

DISK_DEVICE="/dev/disk/by-id/google-plm-agent-data"
if [[ ! -b "${DISK_DEVICE}" ]]; then
  printf 'Persistent disk %s is not attached.\n' "${DISK_DEVICE}" >&2
  exit 1
fi
if ! blkid "${DISK_DEVICE}" >/dev/null 2>&1; then
  mkfs.ext4 -F "${DISK_DEVICE}"
fi

install -d -m 0755 /var/lib/plm-agent
if ! grep -qF "${DISK_DEVICE} /var/lib/plm-agent " /etc/fstab; then
  printf '%s /var/lib/plm-agent ext4 defaults,nofail 0 2\n' "${DISK_DEVICE}" \
    >> /etc/fstab
fi
mountpoint -q /var/lib/plm-agent || mount /var/lib/plm-agent
install -d -m 0755 /var/lib/plm-agent/artifacts /var/lib/plm-agent/web-data
chown 10001:10001 /var/lib/plm-agent/web-data
install -d -m 0755 /opt/plm-agent/secrets

if [[ "${ACCELERATOR}" == "gpu" ]]; then
  nvidia-smi
  docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
fi
touch /var/lib/plm-agent/.host-ready
