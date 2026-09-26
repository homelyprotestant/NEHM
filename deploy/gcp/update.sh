#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source updates use the same secret-free sync, image build, and health gate as
# the initial deployment. VM-side app.env and persistent data remain untouched.
exec "${SCRIPT_DIR}/deploy.sh"
