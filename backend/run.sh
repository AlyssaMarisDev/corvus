#!/usr/bin/env bash
# Starts the Corvus backend: installs deps if needed, then starts the API
# server. Requires GEMINI_API_KEY in .env (loaded via dotenv).
set -euo pipefail

# Windows consoles default to a legacy code page that mangles UTF-8 output
# (an em dash shows up as three stray characters); switch to UTF-8.
# No-op off Windows.
chcp.com 65001 >/dev/null 2>&1 || true

BACKEND_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- API server ------------------------------------------------------------
cd "$BACKEND_DIR"
if [[ ! -d node_modules ]]; then
  npm install
fi

npm start
