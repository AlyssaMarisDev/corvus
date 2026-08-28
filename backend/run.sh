#!/usr/bin/env bash
# Starts the Corvus backend: ensures the pgvector Postgres container and the
# Ollama model exist, installs deps if needed, then starts the API server.
set -euo pipefail

BACKEND_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DB_CONTAINER="corvus-db"
DB_PASSWORD="${POSTGRES_PASSWORD:-corvus}"
DB_NAME="${POSTGRES_DB:-corvus}"
# Host port 5432 is often taken by other local Postgres containers; 5433 avoids that.
DB_PORT="${DB_PORT:-5433}"

# --- Postgres + pgvector ---------------------------------------------------
if ! docker ps --format '{{.Names}}' | grep -qx "$DB_CONTAINER"; then
  if docker ps -a --format '{{.Names}}' | grep -qx "$DB_CONTAINER"; then
    echo "Starting existing $DB_CONTAINER container..."
    docker start "$DB_CONTAINER" >/dev/null
  else
    echo "Creating $DB_CONTAINER container (pgvector/pgvector:pg16)..."
    docker run -d --name "$DB_CONTAINER" \
      -e POSTGRES_PASSWORD="$DB_PASSWORD" \
      -e POSTGRES_DB="$DB_NAME" \
      -p "${DB_PORT:-5433}:5432" \
      pgvector/pgvector:pg16 >/dev/null
  fi
fi

echo "Waiting for Postgres to be ready..."
until docker exec "$DB_CONTAINER" pg_isready -U postgres >/dev/null 2>&1; do
  sleep 1
done

# --- Ollama model ----------------------------------------------------------
OLLAMA_MODEL="${OLLAMA_MODEL:-llama3.1}"
if command -v ollama >/dev/null 2>&1; then
  if ! ollama list | grep -q "^${OLLAMA_MODEL}"; then
    echo "Pulling Ollama model $OLLAMA_MODEL..."
    ollama pull "$OLLAMA_MODEL"
  fi
else
  echo "Warning: ollama CLI not found; assuming a server is reachable at ${OLLAMA_HOST:-http://localhost:11434}"
fi

# --- API server ------------------------------------------------------------
cd "$BACKEND_DIR"
if [[ ! -d node_modules ]]; then
  npm install
fi

npm start
