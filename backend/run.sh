#!/usr/bin/env bash
# Starts the Corvus backend: installs deps if needed, then starts the API
# server. Requires GEMINI_API_KEY in .env (loaded via dotenv).
set -euo pipefail

BACKEND_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- API server ------------------------------------------------------------
cd "$BACKEND_DIR"
if [[ ! -d node_modules ]]; then
  npm install
fi

npm start
