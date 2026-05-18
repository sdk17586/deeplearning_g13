#!/usr/bin/env bash
set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CALIBRATOR_DIR="${APP_DIR}/tensorrt_int8_calibrator"
BUILD_DIR="${CALIBRATOR_DIR}/build"

python3 "${APP_DIR}/scripts/prepare_calib_tensors.py"

cmake -S "${CALIBRATOR_DIR}" -B "${BUILD_DIR}"
cmake --build "${BUILD_DIR}" --parallel

"${BUILD_DIR}/trt_int8_calibrator" "$@"
