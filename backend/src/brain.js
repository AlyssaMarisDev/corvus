import { createClient } from "redis";
import { getRandomMemory } from "./db.js";
import { THOUGHT_PROMPT } from "./prompt.js";
import { logger } from "./logger.js";

const REDIS_URL = process.env.REDIS_URL ?? "redis://localhost:4203";
const DEEPSEEK_MODEL = process.env.DEEPSEEK_MODEL ?? "deepseek-v4-flash";
const THOUGHT_INTERVAL_MS = 5_000;
const MEMORY_INTERVAL_MS = 20_000;
const THOUGHT_TTL_SECONDS = 60;

// Each working-memory entry (thought or surfaced memory) is its own key so
// Redis enforces the 60s TTL; the timestamp prefix keeps lexicographic order
// chronological when enumerating, and the random suffix avoids collisions
// when the thought and memory loops write in the same millisecond.
const KEY_PREFIX = "corvus:thought:";

function workingMemoryKey() {
  return `${KEY_PREFIX}${Date.now()}:${crypto.randomUUID().slice(0, 8)}`;
}

const redis = createClient({ url: REDIS_URL });
redis.on("error", (err) => logger.error({ err }, "redis client error"));

let connecting = false;
async function ensureConnected() {
  if (redis.isReady) return true;
  if (connecting) return false;
  connecting = true;
  try {
    await redis.connect();
    logger.info("thought loop connected to redis");
    return true;
  } catch (err) {
    logger.error({ err }, "redis connect failed; skipping thought tick");
    return false;
  } finally {
    connecting = false;
  }
}

// node-redis v6 scanIterator yields batches of keys, not individual keys.
async function currentThoughts() {
  const keys = [];
  for await (const batch of redis.scanIterator({
    MATCH: `${KEY_PREFIX}*`,
    COUNT: 100,
  })) {
    keys.push(...batch);
  }
  keys.sort();
  if (!keys.length) return [];
  return (await redis.mGet(keys)).filter(Boolean);
}

async function generateThought(existingThoughts) {
  const workingMemory = existingThoughts.length
    ? existingThoughts.map((t, i) => `${i + 1}. ${t}`).join("\n")
    : "(working memory is empty)";
  const response = await fetch("https://api.deepseek.com/chat/completions", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${process.env.DEEPSEEK_API_KEY}`,
    },
    body: JSON.stringify({
      model: DEEPSEEK_MODEL,
      messages: [
        { role: "system", content: THOUGHT_PROMPT },
        { role: "user", content: `Current working memory:\n${workingMemory}` },
      ],
      // Thinking mode adds latency the 5-second cadence cannot afford.
      thinking: { type: "disabled" },
      max_tokens: 128,
    }),
  });
  if (!response.ok) {
    throw new Error(`DeepSeek chat failed: ${response.status} ${await response.text()}`);
  }
  const data = await response.json();
  return data.choices[0].message.content.trim();
}

// A slow generation must not stack up overlapping ticks.
let ticking = false;
async function tick() {
  if (ticking) return;
  ticking = true;
  try {
    if (!(await ensureConnected())) return;
    const thought = await generateThought(await currentThoughts());
    await redis.set(workingMemoryKey(), thought, {
      EX: THOUGHT_TTL_SECONDS,
    });
    // Fetched after the write so the log shows the true current state,
    // new thought included.
    const workingMemory = await currentThoughts();
    logger.info(
      { thought, workingMemory, workingMemorySize: workingMemory.length },
      "new thought"
    );
  } catch (err) {
    logger.error({ err }, "thought tick failed");
  } finally {
    ticking = false;
  }
}

// Surfaces a random long-term memory (regular or core) into working memory,
// tagged so logs and the thought generator can tell it apart from thoughts.
let pulling = false;
async function pullRandomMemory() {
  if (pulling) return;
  pulling = true;
  try {
    if (!(await ensureConnected())) return;
    const memory = await getRandomMemory();
    if (!memory) {
      logger.debug("no long-term memories to pull into working memory");
      return;
    }
    const entry = `[${memory.source}] ${memory.content}`;
    await redis.set(workingMemoryKey(), entry, { EX: THOUGHT_TTL_SECONDS });
    logger.info({ memory: entry }, "memory pulled into working memory");
  } catch (err) {
    logger.error({ err }, "memory pull failed");
  } finally {
    pulling = false;
  }
}

export function startThoughtLoop() {
  if (!process.env.DEEPSEEK_API_KEY) {
    logger.warn("DEEPSEEK_API_KEY not set; thought loop disabled");
    return;
  }
  logger.info(
    {
      intervalMs: THOUGHT_INTERVAL_MS,
      memoryIntervalMs: MEMORY_INTERVAL_MS,
      ttlSeconds: THOUGHT_TTL_SECONDS,
      model: DEEPSEEK_MODEL,
    },
    "thought loop starting"
  );
  // Seed working memory with a random memory before the first thought so the
  // brain doesn't start blank; the first tick then sees the seeded memory.
  void pullRandomMemory().then(() => tick());
  setInterval(() => void tick(), THOUGHT_INTERVAL_MS).unref();
  setInterval(() => void pullRandomMemory(), MEMORY_INTERVAL_MS).unref();
}
