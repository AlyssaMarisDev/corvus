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

// The subject to use whenever a chat turn's origin doesn't resolve one of
// its own (the web frontend has no identity at all; Discord messages from a
// sender with no row in `subjects` fall back to this too) — this is the
// single primary user of a personal Corvus instance. Configurable in case
// that name ever needs to change without touching code.
export const DEFAULT_SUBJECT = process.env.PRIMARY_SUBJECT ?? "bree";

// Fixed subject for stable facts about Corvus itself — its own setup,
// capabilities, or how it operates (e.g. "communicates with the user over
// Discord and a web app") — rather than about whoever it's currently
// speaking with. Core memories on this subject describe Corvus, not the
// speaker, so they always apply no matter who's talking; getMemoriesByTag
// surfaces them alongside the current speaker's own core memories.
export const CORVUS_SUBJECT = "corvus";

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
  // Long-term memory: durable facts learned in conversation, shared across
  // all conversations and retrieved by semantic similarity. Deletes are
  // soft: deleted_at IS NULL means the memory is active. subject is who or
  // what the fact is about — a name (matching a subjects.name row for a
  // known speaker, e.g. "bree", or any other entity mentioned, e.g.
  // "Quentin") or NULL for a general fact tied to no one in particular
  // (e.g. "Corvus and the user communicate over Discord"). It's plain text
  // rather than a foreign key: memories can be about entities (pets,
  // places, other people) that never message Corvus and so never get a
  // subjects row of their own.
  await query(`
    CREATE TABLE IF NOT EXISTS memories (
      id bigserial PRIMARY KEY,
      content text NOT NULL,
      embedding vector(768) NOT NULL,
      created_at timestamptz NOT NULL DEFAULT now(),
      updated_at timestamptz NOT NULL DEFAULT now(),
      deleted_at timestamptz,
      tag memory_tag,
      subject text
    );
  `);
  // For databases created before soft delete existed.
  await query(
    "ALTER TABLE memories ADD COLUMN IF NOT EXISTS deleted_at timestamptz;"
  );
  // For databases created before tags existed.
  await query("ALTER TABLE memories ADD COLUMN IF NOT EXISTS tag memory_tag;");
  // For databases created before subjects existed.
  await query("ALTER TABLE memories ADD COLUMN IF NOT EXISTS subject text;");
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
  // Identity mapping: which name (a memory subject, e.g. "bree") a given
  // channel-specific sender id corresponds to, so an inbound message can be
  // attributed to the right person. Only Discord is wired up today
  // (discord_user_id), but the shape leaves room for other channels later.
  // A name with no discord_user_id is fine (nothing maps to it yet); every
  // column besides id/name/created_at is nullable for that reason.
  await query(`
    CREATE TABLE IF NOT EXISTS subjects (
      id bigserial PRIMARY KEY,
      name text NOT NULL UNIQUE,
      discord_user_id text UNIQUE,
      created_at timestamptz NOT NULL DEFAULT now()
    );
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
// search only ever searches the non-core tier. Deliberately NOT filtered by
// subject: a memory about "Quentin" or a general fact should surface
// regardless of who's currently chatting — only the always-injected core
// tier is scoped to the current speaker (see getMemoriesByTag).
export async function searchMemories(
  queryEmbedding,
  { limit = 5, maxDistance = MEMORY_DISTANCE_CUTOFF } = {}
) {
  const { rows } = await query(
    `SELECT id, content, tag, subject, updated_at, embedding <=> $1 AS distance
     FROM memories
     WHERE deleted_at IS NULL AND tag IS DISTINCT FROM 'core' AND embedding <=> $1 < $2
     ORDER BY distance
     LIMIT $3;`,
    [toVectorLiteral(queryEmbedding), maxDistance, limit]
  );
  return rows;
}

// subject is who/what the memory is about (see the memories table comment
// in initDb) — a name, or null for a general fact tied to no one entity.
export async function saveMemory(content, embedding, tag = null, subject = null) {
  await query(
    "INSERT INTO memories (content, embedding, tag, subject) VALUES ($1, $2, $3, $4);",
    [content, toVectorLiteral(embedding), tag, subject]
  );
}

export async function updateMemory(id, content, embedding, subject = null) {
  await query(
    "UPDATE memories SET content = $2, embedding = $3, subject = $4, updated_at = now() WHERE id = $1;",
    [id, content, toVectorLiteral(embedding), subject]
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
// Scoped to whoever is currently speaking: rows whose subject matches
// `subject` (their own identity/interaction-preference facts), rows tagged
// CORVUS_SUBJECT (stable facts about Corvus itself — always relevant no
// matter who's talking), or rows with no subject at all (general core
// facts) — but never another named subject's core facts. Pass
// DEFAULT_SUBJECT for callers with no specific speaker (the thought loop,
// reminder delivery). The CORVUS_SUBJECT comparison is case-insensitive
// since it's a literal an LLM extraction writes freeform, not an enforced
// enum value.
export async function getMemoriesByTag(tag, { subject = null, limit = 25 } = {}) {
  const { rows } = await query(
    `SELECT id, content, subject, updated_at
     FROM memories
     WHERE deleted_at IS NULL AND tag = $1
       AND (subject IS NULL OR subject = $2 OR lower(subject) = lower($3))
     ORDER BY id
     LIMIT $4;`,
    [tag, subject, CORVUS_SUBJECT, limit]
  );
  return rows;
}

// One-off migration support (see backfill-memory-subjects.js): active
// memories at or below `maxId` still missing a subject — i.e. saved before
// the subject column existed. Bounded by maxId (the highest memories.id
// present before that migration shipped) so a later rerun can never touch
// a genuinely general (subject-less) memory saved afterward by the
// broadened extractor.
export async function getMemoriesWithoutSubject({ maxId, limit = 100 } = {}) {
  const { rows } = await query(
    `SELECT id, content
     FROM memories
     WHERE deleted_at IS NULL AND subject IS NULL AND id <= $1
     ORDER BY id
     LIMIT $2;`,
    [maxId, limit]
  );
  return rows;
}

// Upserts the (name, discord_user_id) mapping used to attribute inbound
// Discord messages to a memory subject (see discord.js). Safe to call every
// boot: re-affirms the mapping if DISCORD_ALLOWED_USER_ID changed.
export async function ensureSubject(name, discordUserId = null) {
  await query(
    `INSERT INTO subjects (name, discord_user_id) VALUES ($1, $2)
     ON CONFLICT (name) DO UPDATE SET discord_user_id = EXCLUDED.discord_user_id;`,
    [name, discordUserId]
  );
}

// Resolves an inbound Discord sender to their memory subject name, or null
// if nobody's been mapped to that id yet (the caller falls back to
// DEFAULT_SUBJECT).
export async function getSubjectNameByDiscordId(discordUserId) {
  const { rows } = await query("SELECT name FROM subjects WHERE discord_user_id = $1;", [
    discordUserId,
  ]);
  return rows[0]?.name ?? null;
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
    `SELECT content, subject, updated_at
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
