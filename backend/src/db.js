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
  // embedding powers semantic search over past conversations (the deep-think
  // subgraph's fetch_past_conversations tool); it is filled in the background
  // after each message is saved. vector(768) matches Gemini's embedding
  // models at 768 output dimensions.
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
  await query(`
    CREATE INDEX IF NOT EXISTS messages_embedding_idx
      ON messages USING hnsw (embedding vector_cosine_ops);
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
  // Core profile memories: stable identity facts and interaction preferences.
  // Every active row is injected into the system prompt, so retrieval never
  // uses the embedding; it exists only so soft-deleted rows stay searchable
  // by the deep-think subgraph.
  await query(`
    CREATE TABLE IF NOT EXISTS core_memories (
      id bigserial PRIMARY KEY,
      content text NOT NULL,
      embedding vector(768) NOT NULL,
      created_at timestamptz NOT NULL DEFAULT now(),
      updated_at timestamptz NOT NULL DEFAULT now(),
      deleted_at timestamptz
    );
  `);
  await query(`
    CREATE INDEX IF NOT EXISTS core_memories_embedding_idx
      ON core_memories USING hnsw (embedding vector_cosine_ops);
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
  const { rows } = await query(
    "INSERT INTO messages (conversation_id, role, content) VALUES ($1, $2, $3) RETURNING id;",
    [conversationId, role, content]
  );
  return rows[0].id;
}

// Cosine distance cutoff for memory retrieval; loose on purpose so borderline
// memories still reach the model, which can ignore irrelevant ones.
const MEMORY_DISTANCE_CUTOFF = 0.6;

// pgvector accepts vectors as string literals like '[0.1,0.2,...]'.
function toVectorLiteral(embedding) {
  return `[${embedding.join(",")}]`;
}

export async function saveMessageEmbedding(id, embedding) {
  await query("UPDATE messages SET embedding = $2 WHERE id = $1;", [
    id,
    toVectorLiteral(embedding),
  ]);
}

// Rows without an embedding yet: the backfill script fills these, which also
// repairs rows whose background embed failed after saving.
export async function getMessagesWithoutEmbedding({ limit = 100 } = {}) {
  const { rows } = await query(
    "SELECT id, content FROM messages WHERE embedding IS NULL ORDER BY id LIMIT $1;",
    [limit]
  );
  return rows;
}

// Semantic search over past conversation messages. Unembedded rows are
// excluded; excludeConversationId keeps the current conversation (already in
// the model's context) out of the results.
export async function searchMessages(
  queryEmbedding,
  { excludeConversationId, limit = 8, maxDistance = MEMORY_DISTANCE_CUTOFF } = {}
) {
  const conditions = ["embedding IS NOT NULL", "embedding <=> $1 < $2"];
  const params = [toVectorLiteral(queryEmbedding), maxDistance];
  if (excludeConversationId) {
    params.push(excludeConversationId);
    conditions.push(`conversation_id != $${params.length}`);
  }
  params.push(limit);
  const { rows } = await query(
    `SELECT id, conversation_id, role, content, created_at, embedding <=> $1 AS distance
     FROM messages
     WHERE ${conditions.join(" AND ")}
     ORDER BY distance
     LIMIT $${params.length};`,
    params
  );
  return rows;
}

export async function searchMemories(
  queryEmbedding,
  { limit = 5, maxDistance = MEMORY_DISTANCE_CUTOFF } = {}
) {
  const { rows } = await query(
    `SELECT id, content, updated_at, embedding <=> $1 AS distance
     FROM memories
     WHERE deleted_at IS NULL
       AND embedding <=> $1 < $2
     ORDER BY distance
     LIMIT $3;`,
    [toVectorLiteral(queryEmbedding), maxDistance, limit]
  );
  return rows;
}

// Deleted memories are soft-deleted rows; the deep-think subgraph searches
// them to answer questions about facts that changed or were forgotten.
export async function searchDeletedMemories(
  queryEmbedding,
  { limit = 5, maxDistance = MEMORY_DISTANCE_CUTOFF } = {}
) {
  const { rows } = await query(
    `SELECT id, content, updated_at, deleted_at, embedding <=> $1 AS distance
     FROM memories
     WHERE deleted_at IS NOT NULL
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

// Active core memories are all injected into the system prompt, so this
// fetches every row; the limit is a sanity cap, not a relevance cutoff.
export async function getCoreMemories({ limit = 25 } = {}) {
  const { rows } = await query(
    `SELECT id, content, updated_at
     FROM core_memories
     WHERE deleted_at IS NULL
     ORDER BY id
     LIMIT $1;`,
    [limit]
  );
  return rows;
}

export async function searchDeletedCoreMemories(
  queryEmbedding,
  { limit = 5, maxDistance = MEMORY_DISTANCE_CUTOFF } = {}
) {
  const { rows } = await query(
    `SELECT id, content, updated_at, deleted_at, embedding <=> $1 AS distance
     FROM core_memories
     WHERE deleted_at IS NOT NULL
       AND embedding <=> $1 < $2
     ORDER BY distance
     LIMIT $3;`,
    [toVectorLiteral(queryEmbedding), maxDistance, limit]
  );
  return rows;
}

export async function saveCoreMemory(content, embedding) {
  await query("INSERT INTO core_memories (content, embedding) VALUES ($1, $2);", [
    content,
    toVectorLiteral(embedding),
  ]);
}

export async function updateCoreMemory(id, content, embedding) {
  await query(
    "UPDATE core_memories SET content = $2, embedding = $3, updated_at = now() WHERE id = $1;",
    [id, content, toVectorLiteral(embedding)]
  );
}

// Soft delete: the row stays in the table but is excluded from retrieval.
export async function deleteCoreMemory(id) {
  await query(
    "UPDATE core_memories SET deleted_at = now() WHERE id = $1 AND deleted_at IS NULL;",
    [id]
  );
}

// Uniform random pick across both memory tiers, for the thought loop's
// working-memory seeding. Returns { content, source } or null when both
// tables are empty.
export async function getRandomMemory() {
  const { rows } = await query(
    `SELECT content, source FROM (
       SELECT content, 'memory' AS source FROM memories WHERE deleted_at IS NULL
       UNION ALL
       SELECT content, 'core memory' AS source FROM core_memories WHERE deleted_at IS NULL
     ) AS all_memories
     ORDER BY random()
     LIMIT 1;`
  );
  return rows[0] ?? null;
}
