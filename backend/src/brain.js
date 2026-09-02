import { createClient } from "redis";
import { getRandomMemory, searchMemories, searchMessages } from "./db.js";
import { embed } from "./memory.js";
import { INTERACTION_SYNTHESIS_PROMPT, THOUGHT_PROMPT, formatFullTimestamp } from "./prompt.js";
import { logger } from "./logger.js";

const REDIS_URL = process.env.REDIS_URL ?? "redis://localhost:4203";
const DEEPSEEK_MODEL = process.env.DEEPSEEK_MODEL ?? "deepseek-v4-flash";
// Idle brains think once a minute to keep spend down; while chat requests
// are arriving the cadence rises to every 5 seconds (see notifyChatRequest).
const IDLE_THOUGHT_INTERVAL_MS = 45_000;
const ACTIVE_THOUGHT_INTERVAL_MS = 5_000;
// Active mode is a sliding window: each chat request extends it to a minute
// from that request, so sustained conversation keeps the brisk cadence.
const ACTIVE_WINDOW_MS = 60_000;
const MEMORY_INTERVAL_MS = 45_000;
const THOUGHT_TTL_SECONDS = 60;
// Interaction summaries anchor the brain to recent reality, so they outlive
// the 60-second thought churn.
const INTERACTION_TTL_SECONDS = 30 * 60;

// Each working-memory entry (thought, surfaced memory/message, or
// interaction summary) is its own key so Redis enforces per-entry TTLs; the
// timestamp prefix keeps lexicographic order chronological when enumerating,
// and the random suffix avoids collisions when the thought and memory loops
// write in the same millisecond.
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
async function currentWorkingMemory() {
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

// Read-only access for the chat path, which injects these entries into the
// system prompt alongside the retrieved long-term memories. Never throws —
// a chat reply must not fail on a working-memory read.
export async function getWorkingMemory() {
  try {
    if (!(await ensureConnected())) return [];
    return await currentWorkingMemory();
  } catch (err) {
    logger.error({ err }, "working memory read failed");
    return [];
  }
}

// Every working-memory entry carries the timestamp of when it occurred:
// thoughts are stamped at generation, memories at last update, messages at
// send time.
function thoughtEntry(content) {
  return `[thought · ${formatFullTimestamp(new Date())}] ${content}`;
}

function memoryEntry({ content, tag, updated_at }) {
  return `[${tag === "core" ? "core memory" : "memory"} · ${formatFullTimestamp(updated_at)}] ${content}`;
}

function messageEntry({ role, created_at, content }) {
  return `[message · ${formatFullTimestamp(created_at)}] ${role === "user" ? "User" : "Butler"}: ${content}`;
}

function interactionEntry(summary) {
  return `[interaction · ${formatFullTimestamp(new Date())}] ${summary}`;
}

// Identity for a working-memory entry is its text: re-adding an entry that
// is already present deletes its old key first, so a re-surfaced entry gets
// a fresh TTL instead of a duplicate.
async function addWorkingMemoryEntry(entry, ttlSeconds = THOUGHT_TTL_SECONDS) {
  const keys = [];
  for await (const batch of redis.scanIterator({
    MATCH: `${KEY_PREFIX}*`,
    COUNT: 100,
  })) {
    keys.push(...batch);
  }
  if (keys.length) {
    const values = await redis.mGet(keys);
    const stale = keys.filter((_, i) => values[i] === entry);
    if (stale.length) await redis.del(stale);
  }
  await redis.set(workingMemoryKey(), entry, {
    expiration: { type: "EX", value: ttlSeconds },
  });
}

// Shared DeepSeek call for the brain's writing (thoughts, interaction
// synthesis). Thinking mode stays disabled: both callers are
// latency-sensitive background work.
async function deepseekChat(systemPrompt, userContent, maxTokens) {
  const response = await fetch("https://api.deepseek.com/chat/completions", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${process.env.DEEPSEEK_API_KEY}`,
    },
    body: JSON.stringify({
      model: DEEPSEEK_MODEL,
      messages: [
        { role: "system", content: systemPrompt },
        { role: "user", content: userContent },
      ],
      thinking: { type: "disabled" },
      max_tokens: maxTokens,
    }),
  });
  if (!response.ok) {
    throw new Error(`DeepSeek chat failed: ${response.status} ${await response.text()}`);
  }
  const data = await response.json();
  return data.choices[0].message.content.trim();
}

async function generateThought(existingThoughts) {
  const workingMemory = existingThoughts.length
    ? existingThoughts.map((t, i) => `${i + 1}. ${t}`).join("\n")
    : "(working memory is empty)";
  return deepseekChat(
    THOUGHT_PROMPT,
    `Current date and time: ${formatFullTimestamp(new Date())}.\n\nCurrent working memory:\n${workingMemory}`,
    128
  );
}

// Synthesizes a completed chat exchange into a short anchor summary and
// stores it in working memory with the long interaction TTL, so the brain
// stays grounded in what actually just happened between user and Corvus.
// Never throws — synthesis is post-reply background work.
export async function synthesizeInteraction(userText, replyText) {
  try {
    if (!(await ensureConnected())) return;
    const summary = await deepseekChat(
      INTERACTION_SYNTHESIS_PROMPT,
      `User: ${userText}\nButler: ${replyText}`,
      160
    );
    const entry = interactionEntry(summary);
    await addWorkingMemoryEntry(entry, INTERACTION_TTL_SECONDS);
    logger.info({ interaction: entry }, "interaction synthesized into working memory");
  } catch (err) {
    logger.error({ err }, "interaction synthesis failed");
  }
}

// Semantic counterpart to the random pull: embed the new thought once and
// surface both the closest memory (either tier) and the closest
// past-conversation message into working memory. The distance cutoffs in
// searchMemories/searchMessages mean an unrelated thought surfaces nothing.
// Never throws — a failure here must not cost the thought's log line.
async function surfaceRelatedContext(thought) {
  try {
    const embedding = await embed(thought);
    const [[memoryMatch], [messageMatch]] = await Promise.all([
      searchMemories(embedding, { limit: 1, includeCore: true }),
      searchMessages(embedding, { limit: 1 }),
    ]);
    const surfacedMemory = memoryMatch ? memoryEntry(memoryMatch) : null;
    const surfacedMessage = messageMatch ? messageEntry(messageMatch) : null;
    if (surfacedMemory) await addWorkingMemoryEntry(surfacedMemory);
    if (surfacedMessage) await addWorkingMemoryEntry(surfacedMessage);
    return { surfacedMemory, surfacedMessage };
  } catch (err) {
    logger.error({ err }, "related context surfacing failed");
    return { surfacedMemory: null, surfacedMessage: null };
  }
}

// A slow generation must not stack up overlapping ticks.
let ticking = false;
async function tick() {
  if (ticking) return;
  ticking = true;
  try {
    if (!(await ensureConnected())) return;
    const generationStart = performance.now();
    const thought = await generateThought(await currentWorkingMemory());
    const generationMs = Math.round(performance.now() - generationStart);
    await redis.set(workingMemoryKey(), thoughtEntry(thought), {
      expiration: { type: "EX", value: THOUGHT_TTL_SECONDS },
    });
    const { surfacedMemory, surfacedMessage } = await surfaceRelatedContext(thought);
    // Fetched after the writes so the log shows the true current state,
    // new thought and surfaced entries included.
    const workingMemory = await currentWorkingMemory();
    logger.info(
      { thought, surfacedMemory, surfacedMessage, workingMemory, workingMemorySize: workingMemory.length, generationMs },
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
    const entry = memoryEntry(memory);
    await addWorkingMemoryEntry(entry);
    logger.info({ memory: entry }, "memory pulled into working memory");
  } catch (err) {
    logger.error({ err }, "memory pull failed");
  } finally {
    pulling = false;
  }
}

// Epoch ms until which active mode lasts; 0 (or any past time) means idle.
let activeUntil = 0;
let thoughtTimer = null;

function thoughtMode() {
  return Date.now() < activeUntil ? "active" : "idle";
}

function currentThoughtInterval() {
  return thoughtMode() === "active" ? ACTIVE_THOUGHT_INTERVAL_MS : IDLE_THOUGHT_INTERVAL_MS;
}

// Self-scheduling chain rather than setInterval, so each tick picks the
// interval for the brain's current mode. The next tick is scheduled only
// after the current one settles; tick() itself never throws.
function scheduleThought(delayMs) {
  if (thoughtTimer) clearTimeout(thoughtTimer);
  thoughtTimer = setTimeout(() => {
    const modeBefore = thoughtMode();
    void tick().then(() => {
      if (modeBefore === "active" && thoughtMode() === "idle") {
        logger.info("brain returning to idle mode");
      }
      scheduleThought(currentThoughtInterval());
    });
  }, delayMs);
  thoughtTimer.unref();
}

// Called by the chat route when a user message comes in: switches the brain
// to active mode (or extends the sliding window), so thinking is brisk while
// conversation is flowing and cheap while nobody is talking to Corvus.
export function notifyChatRequest() {
  const wasIdle = thoughtMode() === "idle";
  activeUntil = Date.now() + ACTIVE_WINDOW_MS;
  // Already active: the pending tick is at most one active interval out, so
  // only the window needs extending. No timer: the loop isn't running.
  if (!wasIdle || !thoughtTimer) return;
  logger.info(
    { activeIntervalMs: ACTIVE_THOUGHT_INTERVAL_MS, activeWindowMs: ACTIVE_WINDOW_MS },
    "brain entering active mode"
  );
  // Don't wait out the remainder of a just-scheduled idle interval.
  scheduleThought(ACTIVE_THOUGHT_INTERVAL_MS);
}

export function startThoughtLoop() {
  if (!process.env.DEEPSEEK_API_KEY) {
    logger.warn("DEEPSEEK_API_KEY not set; thought loop disabled");
    return;
  }
  logger.info(
    {
      idleIntervalMs: IDLE_THOUGHT_INTERVAL_MS,
      activeIntervalMs: ACTIVE_THOUGHT_INTERVAL_MS,
      activeWindowMs: ACTIVE_WINDOW_MS,
      memoryIntervalMs: MEMORY_INTERVAL_MS,
      ttlSeconds: THOUGHT_TTL_SECONDS,
      model: DEEPSEEK_MODEL,
    },
    "thought loop starting"
  );
  // Seed working memory with a random memory before the first thought so the
  // brain doesn't start blank; the first tick then sees the seeded memory.
  // The self-scheduling chain takes over from there.
  void pullRandomMemory()
    .then(() => tick())
    .then(() => scheduleThought(currentThoughtInterval()));
  setInterval(() => void pullRandomMemory(), MEMORY_INTERVAL_MS).unref();
}
