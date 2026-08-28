#!/usr/bin/env bash
# Starts the Corvus frontend (Expo dev server).
set -euo pipefail

FRONTEND_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$FRONTEND_DIR"

if [[ ! -d node_modules ]]; then
  npm install
fi

npm start
