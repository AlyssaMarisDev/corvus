import pg from "pg";
import { logger } from "./logger.js";

const { Pool } = pg;

const pool = new Pool({
  connectionString:
    process.env.DATABASE_URL ?? "postgres://postgres:corvus@localhost:5433/corvus",
});

pool.on("error", (err) => {
  logger.error({ err }, "idle database client error");
});

async function query(text, params) {
  const start = performance.now();
  try {
    const result = await pool.query(text, params);
    logger.debug(
      { rowCount: result.rowCount, durationMs: Math.round(performance.now() - start) },
      "db query"
    );
    return result;
  } catch (err) {
    logger.error({ err, durationMs: Math.round(performance.now() - start) }, "db query failed");
    throw err;
  }
}

export async function initDb() {
  logger.info("initializing database");
  await query("CREATE EXTENSION IF NOT EXISTS vector;");
  await query(`
    CREATE TABLE IF NOT EXISTS conversations (
      id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
      created_at timestamptz NOT NULL DEFAULT now()
    );
  `);
  // embedding is reserved for future semantic memory over chat history;
  // /chat intentionally makes only a single LLM call, so it stays NULL for now.
  // vector(768) matches Gemini's embedding models at 768 output dimensions.
  await query(`
    CREATE TABLE IF NOT EXISTS messages (
      id bigserial PRIMARY KEY,
      conversation_id uuid NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
      role text NOT NULL CHECK (role IN ('user', 'assistant')),
      content text NOT NULL,
      embedding vector(768),
      created_at timestamptz NOT NULL DEFAULT now()
    );
  `);
  // Long-term memory: durable facts about the user, shared across all
  // conversations and retrieved by semantic similarity. Deletes are soft:
  // deleted_at IS NULL means the memory is active.
  await query(`
    CREATE TABLE IF NOT EXISTS memories (
      id bigserial PRIMARY KEY,
      content text NOT NULL,
      embedding vector(768) NOT NULL,
      created_at timestamptz NOT NULL DEFAULT now(),
      updated_at timestamptz NOT NULL DEFAULT now(),
      deleted_at timestamptz
    );
  `);
  // For databases created before soft delete existed.
  await query(
    "ALTER TABLE memories ADD COLUMN IF NOT EXISTS deleted_at timestamptz;"
  );
  await query(`
    CREATE INDEX IF NOT EXISTS memories_embedding_idx
      ON memories USING hnsw (embedding vector_cosine_ops);
  `);
  logger.info("database ready");
}

export async function createConversation() {
  const { rows } = await query(
    "INSERT INTO conversations DEFAULT VALUES RETURNING id;"
  );
  return rows[0].id;
}

export async function conversationExists(conversationId) {
  const { rows } = await query(
    "SELECT 1 FROM conversations WHERE id = $1;",
    [conversationId]
  );
  return rows.length > 0;
}

export async function loadHistory(conversationId) {
  const { rows } = await query(
    "SELECT role, content FROM messages WHERE conversation_id = $1 ORDER BY id ASC;",
    [conversationId]
  );
  return rows;
}

export async function saveMessage(conversationId, role, content) {
  await query(
    "INSERT INTO messages (conversation_id, role, content) VALUES ($1, $2, $3);",
    [conversationId, role, content]
  );
}

// Cosine distance cutoff for memory retrieval; loose on purpose so borderline
// memories still reach the model, which can ignore irrelevant ones.
const MEMORY_DISTANCE_CUTOFF = 0.6;

// pgvector accepts vectors as string literals like '[0.1,0.2,...]'.
function toVectorLiteral(embedding) {
  return `[${embedding.join(",")}]`;
}

export async function searchMemories(
  queryEmbedding,
  { limit = 5, maxDistance = MEMORY_DISTANCE_CUTOFF } = {}
) {
  const { rows } = await query(
    `SELECT id, content, embedding <=> $1 AS distance
     FROM memories
     WHERE deleted_at IS NULL
       AND embedding <=> $1 < $2
     ORDER BY distance
     LIMIT $3;`,
    [toVectorLiteral(queryEmbedding), maxDistance, limit]
  );
  return rows;
}

export async function saveMemory(content, embedding) {
  await query("INSERT INTO memories (content, embedding) VALUES ($1, $2);", [
    content,
    toVectorLiteral(embedding),
  ]);
}

export async function updateMemory(id, content, embedding) {
  await query(
    "UPDATE memories SET content = $2, embedding = $3, updated_at = now() WHERE id = $1;",
    [id, content, toVectorLiteral(embedding)]
  );
}

// Soft delete: the row stays in the table but is excluded from retrieval.
export async function deleteMemory(id) {
  await query(
    "UPDATE memories SET deleted_at = now() WHERE id = $1 AND deleted_at IS NULL;",
    [id]
  );
}
