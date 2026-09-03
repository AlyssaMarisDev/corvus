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

// The tags a memory can carry; "core" marks profile facts injected into
// every system prompt. Source of truth for the memory_tag Postgres enum —
// initDb creates the type and syncs values on boot.
export const MEMORY_TAGS = ["core"];

export async function initDb() {
  logger.info("initializing database");
  await query("CREATE EXTENSION IF NOT EXISTS vector;");
  // CREATE TYPE has no IF NOT EXISTS, hence the DO block; the ALTER loop
  // adds any values an existing database is missing when MEMORY_TAGS grows.
  // Each ADD VALUE is its own statement: a new enum value cannot be used in
  // the transaction that creates it.
  await query(`
    DO $$ BEGIN
      CREATE TYPE memory_tag AS ENUM (${MEMORY_TAGS.map((t) => `'${t}'`).join(", ")});
    EXCEPTION WHEN duplicate_object THEN NULL;
    END $$;
  `);
  for (const tag of MEMORY_TAGS) {
    await query(`ALTER TYPE memory_tag ADD VALUE IF NOT EXISTS '${tag}';`);
  }
  // Corvus has a single conversation, so messages carry no conversation id.
  // embedding powers semantic search over past messages (the search_memory
  // tool, tools/searchMemory.js); it is filled in the background after each message is
  // saved. vector(768) matches Gemini's embedding models at 768 output
  // dimensions.
  await query(`
    CREATE TABLE IF NOT EXISTS messages (
      id bigserial PRIMARY KEY,
      role text NOT NULL CHECK (role IN ('user', 'assistant')),
      content text NOT NULL,
      embedding vector(768),
      created_at timestamptz NOT NULL DEFAULT now()
    );
  `);
  // For databases from the multi-conversation era: folding every thread into
  // the single conversation is just dropping the column (the FK goes with
  // it), after which the conversations table is empty of meaning.
  await query("ALTER TABLE messages DROP COLUMN IF EXISTS conversation_id;");
  await query("DROP TABLE IF EXISTS conversations;");
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
      deleted_at timestamptz,
      tag memory_tag
    );
  `);
  // For databases created before soft delete existed.
  await query(
    "ALTER TABLE memories ADD COLUMN IF NOT EXISTS deleted_at timestamptz;"
  );
  // For databases created before tags existed.
  await query("ALTER TABLE memories ADD COLUMN IF NOT EXISTS tag memory_tag;");
  await query(`
    CREATE INDEX IF NOT EXISTS memories_embedding_idx
      ON memories USING hnsw (embedding vector_cosine_ops);
  `);
  // One-time migration folding the old core_memories table into memories
  // tagged 'core'; skips once the table is gone.
  await query(`
    DO $$ BEGIN
      IF EXISTS (
        SELECT 1 FROM pg_tables WHERE schemaname = 'public' AND tablename = 'core_memories'
      ) THEN
        INSERT INTO memories (content, embedding, created_at, updated_at, deleted_at, tag)
          SELECT content, embedding, created_at, updated_at, deleted_at, 'core'
          FROM core_memories;
        DROP TABLE core_memories;
      END IF;
    END $$;
  `);
  // Self-scheduled reminders: set_reminder (tools/setReminder.js) lets Corvus schedule
  // something to resurface into working memory at a future time. Recurring
  // reminders (recurrence_interval NOT NULL) roll due_at forward on each
  // fire instead of being retired; one-off ones are cancelled once fired.
  await query(`
    CREATE TABLE IF NOT EXISTS reminders (
      id bigserial PRIMARY KEY,
      content text NOT NULL,
      due_at timestamptz NOT NULL,
      recurrence_interval interval,
      created_at timestamptz NOT NULL DEFAULT now(),
      last_fired_at timestamptz,
      cancelled_at timestamptz
    );
  `);
  await query(`
    CREATE INDEX IF NOT EXISTS reminders_due_idx
      ON reminders (due_at) WHERE cancelled_at IS NULL;
  `);
  logger.info("database ready");
}

// The full conversation, oldest first — for the frontend's launch load.
export async function loadHistory() {
  const { rows } = await query(
    "SELECT role, content FROM messages ORDER BY id ASC;"
  );
  return rows;
}

// Only the most recent rows feed the model's context (the chat route caps
// the conversation at the last 10 exchanges). Fetched newest-first, then
// reversed back into chronological order.
export async function loadRecentMessages({ limit = 20 } = {}) {
  const { rows } = await query(
    "SELECT role, content FROM messages ORDER BY id DESC LIMIT $1;",
    [limit]
  );
  return rows.reverse();
}

export async function saveMessage(role, content) {
  const { rows } = await query(
    "INSERT INTO messages (role, content) VALUES ($1, $2) RETURNING id;",
    [role, content]
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

// Semantic search over past messages. Unembedded rows are excluded.
export async function searchMessages(
  queryEmbedding,
  { limit = 8, maxDistance = MEMORY_DISTANCE_CUTOFF } = {}
) {
  const { rows } = await query(
    `SELECT id, role, content, created_at, embedding <=> $1 AS distance
     FROM messages
     WHERE embedding IS NOT NULL AND embedding <=> $1 < $2
     ORDER BY distance
     LIMIT $3;`,
    [toVectorLiteral(queryEmbedding), maxDistance, limit]
  );
  return rows;
}

// Core-tagged rows are always excluded: they are already injected into
// every system prompt and always given to the thought generator directly
// (brain.js), so they must never surface here too — regular similarity
// search only ever searches the non-core tier.
export async function searchMemories(
  queryEmbedding,
  { limit = 5, maxDistance = MEMORY_DISTANCE_CUTOFF } = {}
) {
  const { rows } = await query(
    `SELECT id, content, tag, updated_at, embedding <=> $1 AS distance
     FROM memories
     WHERE deleted_at IS NULL AND tag IS DISTINCT FROM 'core' AND embedding <=> $1 < $2
     ORDER BY distance
     LIMIT $3;`,
    [toVectorLiteral(queryEmbedding), maxDistance, limit]
  );
  return rows;
}

export async function saveMemory(content, embedding, tag = null) {
  await query("INSERT INTO memories (content, embedding, tag) VALUES ($1, $2, $3);", [
    content,
    toVectorLiteral(embedding),
    tag,
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

// Tagged memories (e.g. core) are all injected into the system prompt, so
// this fetches every active row with the tag rather than
// similarity-searching; the limit is a sanity cap, not a relevance cutoff.
export async function getMemoriesByTag(tag, { limit = 25 } = {}) {
  const { rows } = await query(
    `SELECT id, content, updated_at
     FROM memories
     WHERE deleted_at IS NULL AND tag = $1
     ORDER BY id
     LIMIT $2;`,
    [tag, limit]
  );
  return rows;
}

// A memory's selection weight decays exponentially with its age (by
// updated_at), halving every MEMORY_PULL_HALFLIFE_SECONDS. Old memories can
// still surface, just less often.
const MEMORY_PULL_HALFLIFE_SECONDS = 7 * 24 * 60 * 60;

// Recency-weighted random pick across active, non-core memories, for the
// thought loop's working-memory seeding. Core memories are excluded: they
// are already always given to the thought generator directly (brain.js),
// so they must never surface here as well. Uses the exponential-race
// method: -ln(1-random())/w is an Exp(w) variate, so ordering by it
// ascending picks each row with probability proportional to its weight w.
// Returns { content, updated_at }, or null when there are no eligible rows.
export async function getRandomMemory() {
  const { rows } = await query(
    `SELECT content, updated_at
     FROM memories
     WHERE deleted_at IS NULL AND tag IS DISTINCT FROM 'core'
     ORDER BY -ln(1 - random()) / exp(-EXTRACT(EPOCH FROM (now() - updated_at)) * ln(2) / $1)
     LIMIT 1;`,
    [MEMORY_PULL_HALFLIFE_SECONDS]
  );
  return rows[0] ?? null;
}

// recurrenceInterval is a Postgres interval literal (e.g. "1 day") or null
// for a one-time reminder. dueAt is a Date.
export async function saveReminder(content, dueAt, recurrenceInterval) {
  const { rows } = await query(
    `INSERT INTO reminders (content, due_at, recurrence_interval)
     VALUES ($1, $2, $3) RETURNING id;`,
    [content, dueAt, recurrenceInterval]
  );
  return rows[0].id;
}

// Due, not-yet-cancelled reminders, earliest first — polled by the brain's
// reminder check on a short interval independent of thought cadence.
export async function getDueReminders() {
  const { rows } = await query(
    `SELECT id, content, due_at, recurrence_interval
     FROM reminders
     WHERE cancelled_at IS NULL AND due_at <= now()
     ORDER BY due_at ASC;`
  );
  return rows;
}

// Recurring reminders roll due_at forward by their own interval (anchoring
// to the schedule rather than to now, so a fixed daily time doesn't drift);
// one-off reminders are retired via cancelled_at.
export async function fireReminder(id, recurrenceInterval) {
  if (recurrenceInterval) {
    await query(
      `UPDATE reminders
       SET due_at = due_at + $2, last_fired_at = now()
       WHERE id = $1;`,
      [id, recurrenceInterval]
    );
  } else {
    await query(
      `UPDATE reminders SET cancelled_at = now(), last_fired_at = now() WHERE id = $1;`,
      [id]
    );
  }
}
