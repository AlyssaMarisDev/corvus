import pg from "pg";

const { Pool } = pg;

const pool = new Pool({
  connectionString:
    process.env.DATABASE_URL ?? "postgres://postgres:corvus@localhost:5433/corvus",
});

export async function initDb() {
  await pool.query("CREATE EXTENSION IF NOT EXISTS vector;");
  await pool.query(`
    CREATE TABLE IF NOT EXISTS conversations (
      id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
      created_at timestamptz NOT NULL DEFAULT now()
    );
  `);
  // embedding is reserved for future semantic memory over chat history;
  // /chat intentionally makes only a single LLM call, so it stays NULL for now.
  // vector(768) matches Ollama's nomic-embed-text for when that lands.
  await pool.query(`
    CREATE TABLE IF NOT EXISTS messages (
      id bigserial PRIMARY KEY,
      conversation_id uuid NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
      role text NOT NULL CHECK (role IN ('user', 'assistant')),
      content text NOT NULL,
      embedding vector(768),
      created_at timestamptz NOT NULL DEFAULT now()
    );
  `);
}

export async function createConversation() {
  const { rows } = await pool.query(
    "INSERT INTO conversations DEFAULT VALUES RETURNING id;"
  );
  return rows[0].id;
}

export async function conversationExists(conversationId) {
  const { rows } = await pool.query(
    "SELECT 1 FROM conversations WHERE id = $1;",
    [conversationId]
  );
  return rows.length > 0;
}

export async function loadHistory(conversationId) {
  const { rows } = await pool.query(
    "SELECT role, content FROM messages WHERE conversation_id = $1 ORDER BY id ASC;",
    [conversationId]
  );
  return rows;
}

export async function saveMessage(conversationId, role, content) {
  await pool.query(
    "INSERT INTO messages (conversation_id, role, content) VALUES ($1, $2, $3);",
    [conversationId, role, content]
  );
}
