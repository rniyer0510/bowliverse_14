#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <image-ref>" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONTEXT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
IMAGE_REF="$1"

docker buildx build \
  -f "${CONTEXT_ROOT}/app/deploy/Dockerfile" \
  -t "${IMAGE_REF}" \
  "${CONTEXT_ROOT}"
