#!/usr/bin/env bash
set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP="${APP_DIR}/build/deepstream_action_predict"
DEFAULT_VIDEO="/root/data_with_weight_file/dataset/val/clips/broken/broken_0000.mp4"

export LANG="C.utf8"
export LC_ALL="C.utf8"
export LC_CTYPE="C.utf8"

if [[ ! -x "${APP}" ]]; then
  "${APP_DIR}/build.sh"
fi

if [[ $# -lt 1 ]]; then
  if [[ ! -f "${DEFAULT_VIDEO}" ]]; then
    echo "Usage: $0 <video path|uri> [infer_config] [preprocess_config]" >&2
    echo "Default video not found: ${DEFAULT_VIDEO}" >&2
    exit 1
  fi
  echo "[안내] 입력 영상이 없어 기본 샘플을 사용합니다."
  echo "[안내] ${DEFAULT_VIDEO}"
  set -- "${DEFAULT_VIDEO}"
fi

cd "${APP_DIR}"
export GST_DEBUG="${GST_DEBUG:-1}"

set +e
set +o pipefail
"${APP}" "$@" 2>&1 | grep -v -E \
  '(^INFO:|[[:space:]]INFO[[:space:]]|INFO: CUSTOM_LIB|Implicit Engine Info|NvDsInferContext|notifyLoadModelStatus|sequence_image_process\.cpp|Opening in BLOCKING MODE|nvstreammux: Successfully handled EOS)'
status=${PIPESTATUS[0]}
set -e
set -o pipefail
exit "${status}"
