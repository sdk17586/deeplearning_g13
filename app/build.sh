#!/usr/bin/env bash
set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${APP_DIR}/build"

cmake -S "${APP_DIR}" -B "${BUILD_DIR}" -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build "${BUILD_DIR}" --parallel
ln -sfn build/compile_commands.json "${APP_DIR}/compile_commands.json"

echo "[OK] Built: ${APP_DIR}/build/deepstream_action_predict"
