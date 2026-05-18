#!/usr/bin/env bash
set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CALIBRATOR_DIR="${APP_DIR}/tensorrt_int8_calibrator"
BUILD_DIR="${CALIBRATOR_DIR}/build-release"

python3 "${APP_DIR}/scripts/prepare_calib_tensors.py"

cmake --preset=release -S "${CALIBRATOR_DIR}"
cmake --build "${BUILD_DIR}" --parallel

"${BUILD_DIR}/trt_int8_calibrator" "$@"
