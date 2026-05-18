#!/usr/bin/env bash
set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

rm -rf "${APP_DIR}/build"
rm -f "${APP_DIR}/compile_commands.json"

echo "[OK] Cleaned build artifacts"
