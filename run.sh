#!/usr/bin/env bash
# Starts the whole Corvus stack: Postgres (Docker), backend, frontend.
# Logs: backend/backend.log and frontend/frontend.log
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Starting backend (db container + Ollama check + API)..."
(cd "$ROOT_DIR/backend" && ./run.sh > backend.log 2>&1 &)

echo "Starting frontend (Expo dev server)..."
(cd "$ROOT_DIR/frontend" && ./run.sh > frontend.log 2>&1 &)

echo ""
echo "Corvus is starting up."
echo "  Backend:  http://localhost:4000  (tail -f backend/backend.log)"
echo "  Frontend: Expo dev server        (tail -f frontend/frontend.log)"
