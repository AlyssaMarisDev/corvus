import { createClient } from "redis";
import {
  fireReminder,
  getDueReminders,
  getRandomMemory,
  saveMessage,
  saveMessageEmbedding,
  searchMemories,
  searchMessages,
} from "./db.js";
import { embed, retrieveCoreMemories } from "./memory.js";
import {
  INTERACTION_SYNTHESIS_PROMPT,
  PROACTIVE_PROMPT,
  REMINDER_DELIVERY_PROMPT,
  THOUGHT_JUDGE_PROMPT,
  THOUGHT_PROMPT,
  buildSystemPrompt,
  formatFullTimestamp,
} from "./prompt.js";
import { DEEPSEEK_MODEL, streamDeepSeekReply, toUsageDetails } from "./deepseek.js";
import { broadcast } from "./events.js";
import { sendDiscordMessage, discordEnabled } from "./discord.js";
import { endError, endOk, startChild, startTrace } from "./tracing.js";
import { logger } from "./logger.js";

const REDIS_URL = process.env.REDIS_URL ?? "redis://localhost:4203";
// Idle brains think once a minute to keep spend down; while chat requests
// are arriving the cadence rises to every 5 seconds (see notifyChatRequest).
const IDLE_THOUGHT_INTERVAL_MS = 45_000;
const ACTIVE_THOUGHT_INTERVAL_MS = 5_000;
// Active mode is a sliding window: each chat request extends it to a minute
// from that request, so sustained conversation keeps the brisk cadence.
const ACTIVE_WINDOW_MS = 60_000;
const MEMORY_INTERVAL_MS = 45_000;
// Reminders are time-sensitive (the user picked, or Corvus picked, a
// specific moment), so they're polled on their own short cadence rather
// than riding the thought tick's idle/active interval.
const REMINDER_CHECK_INTERVAL_MS = 15_000;
const THOUGHT_TTL_SECONDS = 60;
// Long enough that a fired reminder survives well past the next few thought
// ticks even in idle mode, without lingering forever if never noticed.
const REMINDER_TTL_SECONDS = 60 * 60;
// Interaction summaries anchor the brain to recent reality, so they outlive
// the 60-second thought churn.
const INTERACTION_TTL_SECONDS = 30 * 60;
// Proactive pipeline: each generated thought is judged by a small model,
// and judge-approved thoughts go to the main response model, which decides
// whether to actually message the user. The cooldown starts on a judge
// approval, so the expensive main-model call (and thus proactive messages)
// happen at most once per interval. Proactive work is also suppressed while
// a chat reply is streaming, so it never interrupts a live answer.
const PROACTIVE_ENABLED = process.env.PROACTIVE_ENABLED !== "false";
const PROACTIVE_COOLDOWN_MS = Number(process.env.PROACTIVE_COOLDOWN_MS ?? 300_000);
// The main model replies with exactly this when it chooses to stay silent.
const SILENT_SENTINEL = "[SILENT]";

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

// Core memories never reach working memory (they're always given to the
// thought generator directly, see generateThought/judgeThought below), so
// every entry built here is necessarily a regular memory.
// subject labels who/what a regular memory is about (see db.js's memories
// table and prompt.js's formatMemoryLine, which does the same for the
// system prompt's memory list); omitted when null (a general fact).
function memoryEntry({ content, updated_at, subject }) {
  const label = subject ? `[${subject}] ` : "";
  return `[memory · ${formatFullTimestamp(updated_at)}] ${label}${content}`;
}

function messageEntry({ role, created_at, content }) {
  return `[message · ${formatFullTimestamp(created_at)}] ${role === "user" ? "User" : "Butler"}: ${content}`;
}

function interactionEntry(summary) {
  return `[interaction · ${formatFullTimestamp(new Date())}] ${summary}`;
}

// Reminders are self-scheduled via the set_reminder tool (tools/setReminder.js) and are
// always tagged very important — the prompt (prompt.js) tells Corvus to
// treat this tag as a priority anchor in working memory.
function reminderEntry(content) {
  return `[reminder · very important · ${formatFullTimestamp(new Date())}] ${content}`;
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
// synthesis, thought judging). Thinking mode stays disabled: all callers
// are latency-sensitive background work. json enables DeepSeek's JSON
// output mode, which requires the prompt to ask for JSON. parent/name
// wrap the call in a Langfuse generation observation — every caller passes
// its own trace (a thought tick, or the chat trace's extract span) plus a
// name identifying which of the brain's writing tasks this is.
async function deepseekChat(
  systemPrompt,
  userContent,
  maxTokens,
  { json = false, parent, name = "deepseek-chat" } = {}
) {
  const messages = [
    { role: "system", content: systemPrompt },
    { role: "user", content: userContent },
  ];
  const generation = startChild(parent, name, { model: DEEPSEEK_MODEL, input: messages }, "generation");
  try {
    const response = await fetch("https://api.deepseek.com/chat/completions", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${process.env.DEEPSEEK_API_KEY}`,
      },
      body: JSON.stringify({
        model: DEEPSEEK_MODEL,
        messages,
        thinking: { type: "disabled" },
        max_tokens: maxTokens,
        ...(json ? { response_format: { type: "json_object" } } : {}),
      }),
    });
    if (!response.ok) {
      throw new Error(`DeepSeek chat failed: ${response.status} ${await response.text()}`);
    }
    const data = await response.json();
    const content = data.choices[0].message.content.trim();
    endOk(generation, { output: content, usageDetails: toUsageDetails(data.usage) });
    return content;
  } catch (err) {
    endError(generation, err);
    throw err;
  }
}

// Core memories are always-apply facts about the user (identity, interaction
// preferences), so they're passed directly here rather than left to
// randomly surface into working memory like regular memories do.
function formatCoreMemories(coreMemories) {
  return coreMemories.length ? coreMemories.map((m) => `- ${m.content}`).join("\n") : "(none)";
}

async function generateThought(existingThoughts, coreMemories, trace) {
  const workingMemory = existingThoughts.length
    ? existingThoughts.map((t, i) => `${i + 1}. ${t}`).join("\n")
    : "(working memory is empty)";
  return deepseekChat(
    THOUGHT_PROMPT,
    `Current date and time: ${formatFullTimestamp(new Date())}.\n\nCore profile (always apply):\n${formatCoreMemories(coreMemories)}\n\nCurrent working memory:\n${workingMemory}`,
    128,
    { parent: trace, name: "generate-thought" }
  );
}

// Shared by every interaction-synthesis call site (a chat exchange, or a
// proactive/reminder message Corvus sent unprompted): asks the small model
// to condense `content` into a short third-person anchor summary and stores
// it in working memory with the long interaction TTL, so the brain stays
// grounded in what actually just happened. Never throws — synthesis is
// always post-send background work.
async function storeInteractionSummary(content, parent) {
  const summary = await deepseekChat(INTERACTION_SYNTHESIS_PROMPT, content, 160, {
    parent,
    name: "synthesize-interaction",
  });
  const entry = interactionEntry(summary);
  await addWorkingMemoryEntry(entry, INTERACTION_TTL_SECONDS);
  logger.info({ interaction: entry }, "interaction synthesized into working memory");
}

// Synthesizes a completed chat exchange into working memory. See
// storeInteractionSummary.
export async function synthesizeInteraction(userText, replyText, parent) {
  try {
    if (!(await ensureConnected())) return;
    await storeInteractionSummary(`User: ${userText}\nButler: ${replyText}`, parent);
  } catch (err) {
    logger.error({ err }, "interaction synthesis failed");
  }
}

// Same as synthesizeInteraction, but for a message Corvus sent unprompted
// (a voiced thought or a delivered reminder — see sendProactiveMessage).
// There's no user turn to summarize, so `context` describes what prompted
// Corvus to speak (the thought, or the reminder it fired on) and the model
// is asked to summarize that it acted on it, not just what it said — this
// is what lets the thought loop recognize a reminder as already delivered
// instead of re-surfacing it as unaddressed (see THOUGHT_PROMPT/
// THOUGHT_JUDGE_PROMPT).
async function synthesizeProactiveInteraction(context, message, parent) {
  try {
    if (!(await ensureConnected())) return;
    await storeInteractionSummary(`${context}\nButler said: ${message}`, parent);
  } catch (err) {
    logger.error({ err }, "proactive interaction synthesis failed");
  }
}

// Semantic counterpart to the random pull: embed the new thought once and
// surface both the closest regular memory and the closest past-conversation
// message into working memory. Core memories are never surfaced here — see
// searchMemories — since they're always given to the thought generator
// directly instead. The distance cutoffs in searchMemories/searchMessages
// mean an unrelated thought surfaces nothing. Never throws — a failure here
// must not cost the thought's log line.
async function surfaceRelatedContext(thought, parent) {
  try {
    const embedding = await embed(thought, parent);
    const [[memoryMatch], [messageMatch]] = await Promise.all([
      searchMemories(embedding, { limit: 1 }),
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

// First proactive gate: the judge sees the whole working memory — the
// candidate thought is its newest [thought] entry, judged together with the
// memories and messages it just surfaced — plus the always-apply core
// profile, and answers JSON {"surface": bool, "reason": str}. Any doubt or
// parse failure means no.
async function judgeThought(workingMemory, coreMemories, trace) {
  const entries = workingMemory.length
    ? workingMemory.map((e, i) => `${i + 1}. ${e}`).join("\n")
    : "(working memory is empty)";
  const raw = await deepseekChat(
    THOUGHT_JUDGE_PROMPT,
    `Current date and time: ${formatFullTimestamp(new Date())}.\n\nCore profile (always apply):\n${formatCoreMemories(coreMemories)}\n\nCurrent working memory (oldest first; the final [thought] entry is the candidate):\n${entries}`,
    128,
    { json: true, parent: trace, name: "judge-thought" }
  );
  try {
    const parsed = JSON.parse(raw);
    return { surface: parsed.surface === true, reason: String(parsed.reason ?? "") };
  } catch {
    return { surface: false, reason: "unparseable judge response" };
  }
}

// Shared by both proactive callers (a judge-approved thought, or a fired
// reminder): the main response model (same persona prompt and reasoning
// effort as chat) composes the message itself, using promptAddendum to
// decide its own framing (PROACTIVE_PROMPT offers a silence option;
// REMINDER_DELIVERY_PROMPT does not). Tokens are held back until they can
// no longer be the silence sentinel's prefix, then stream out via onToken.
// Returns the message, or null when the model declines (either via the
// sentinel, or — for callers that don't offer it — an empty reply).
async function generateProactiveMessage({
  workingMemory,
  coreMemories,
  trigger,
  promptAddendum,
  onToken,
  parent,
  name,
}) {
  const systemPrompt = `${buildSystemPrompt({
    coreMemories,
    workingMemory,
  })}\n\n${promptAddendum}`;
  const userMessage = { role: "user", content: trigger };
  const generation = startChild(
    parent,
    name,
    { model: DEEPSEEK_MODEL, input: [{ role: "system", content: systemPrompt }, userMessage] },
    "generation"
  );
  let heldBack = "";
  let streaming = false;
  try {
    const { content, usage } = await streamDeepSeekReply(systemPrompt, [userMessage], {
      emitEvents: false,
      onToken: (text) => {
        if (streaming) {
          onToken(text);
          return;
        }
        heldBack += text;
        // While the trimmed accumulation is still a prefix of the sentinel,
        // the model may be about to decline — keep holding tokens back.
        if (SILENT_SENTINEL.startsWith(heldBack.trimStart())) return;
        streaming = true;
        onToken(heldBack);
      },
    });
    const message = content.trim();
    const declined = !message || message === SILENT_SENTINEL;
    endOk(generation, {
      output: declined ? SILENT_SENTINEL : message,
      usageDetails: toUsageDetails(usage),
      metadata: { declined },
    });
    return declined ? null : message;
  } catch (err) {
    endError(generation, err);
    throw err;
  }
}

// Cooldown state for the proactive pipeline, plus the count of chat replies
// currently streaming — a proactive message must never interrupt one.
let lastVoiceAttemptAt = 0;
let activeChatStreams = 0;

// Called by the chat route when a reply stream starts and ends (normal
// completion or client disconnect alike).
export function notifyChatStreamStart() {
  activeChatStreams += 1;
}

export function notifyChatStreamEnd() {
  activeChatStreams = Math.max(0, activeChatStreams - 1);
  // A reminder that fired mid-reply is waiting on exactly this; flush it the
  // instant the stream clears rather than waiting for the next poll.
  if (activeChatStreams === 0) void flushPendingReminders();
}

// Shared tail end of every proactive send, once a message has actually been
// composed (by either maybeVoiceThought or deliverReminder): persist it to
// the conversation, push it to connected clients, and echo it back into
// working memory so the brain knows it just spoke. Streaming (proactive_
// start/token) already happened via generateProactiveMessage's onToken;
// this only sends the final "done" broadcast. `interactionContext` describes
// what prompted the message (the voiced thought, or the reminder that
// fired) and is synthesized into an [interaction] anchor alongside the
// literal [message] entry — without it, working memory only ever recorded
// the raw text Corvus sent, never an explicit "I acted on this" anchor, so
// the brain could see a still-present [reminder] entry and think it hadn't
// been handled yet even after delivery.
async function sendProactiveMessage(message, interactionContext, parent) {
  const messageId = await saveMessage("assistant", message);
  // Same background-embed pattern as the chat route; a failure leaves
  // embedding NULL, which the backfill script repairs on its next run.
  void embed(message, parent)
    .then((e) => saveMessageEmbedding(messageId, e))
    .catch((err) => logger.error({ err }, "proactive message embedding failed"));
  // Ground the brain in the fact that it just reached out, so subsequent
  // thoughts build on the outreach instead of repeating it.
  await addWorkingMemoryEntry(
    messageEntry({ role: "assistant", created_at: new Date(), content: message }),
    INTERACTION_TTL_SECONDS
  );
  await synthesizeProactiveInteraction(interactionContext, message, parent);
  broadcast("proactive_done", { message });
  // Point-in-time marker of the exact moment this message went out over
  // the /events SSE channel — this is what answers "when was this pushed"
  // in Langfuse, distinct from the generation that composed it.
  startChild(parent, "push-to-frontend", { input: { message }, metadata: { channel: "sse:/events" } }, "event");
  // Same message, over Discord — sendDiscordMessage never throws (see
  // discord.js), so this can never break the SSE push above.
  if (discordEnabled) {
    void sendDiscordMessage(message);
    startChild(parent, "push-to-discord", { input: { message }, metadata: { channel: "discord" } }, "event");
  }
}

// The proactive pipeline: judge the just-generated thought, and when
// approved, let the main model decide whether to turn it into an actual
// message. coreMemories comes from the caller (tick), which already fetched
// it for generateThought — reused here rather than fetched again. Never
// throws — it runs inside the tick.
async function maybeVoiceThought(thought, workingMemory, coreMemories, trace) {
  if (!PROACTIVE_ENABLED || activeChatStreams > 0) return;
  if (Date.now() - lastVoiceAttemptAt < PROACTIVE_COOLDOWN_MS) return;
  try {
    const verdict = await judgeThought(workingMemory, coreMemories, trace);
    logger.info({ thought, ...verdict }, "thought judged");
    if (!verdict.surface) return;
    lastVoiceAttemptAt = Date.now();

    let started = false;
    const message = await generateProactiveMessage({
      workingMemory,
      coreMemories,
      trigger:
        `A thought from your subconscious passed an initial relevance check:\n\n"${thought}"\n\n` +
        `Decide whether to actually message the user. Reply with only your message, or exactly ${SILENT_SENTINEL} to stay silent.`,
      promptAddendum: PROACTIVE_PROMPT,
      parent: trace,
      name: "voice-thought",
      onToken: (text) => {
        if (!started) {
          started = true;
          broadcast("proactive_start", {});
        }
        broadcast("proactive_token", { text });
      },
    });
    if (!message) {
      logger.info({ thought }, "corvus chose silence over a surfaced thought");
      return;
    }

    await sendProactiveMessage(
      message,
      `Corvus spoke unprompted, following up on its own thought: "${thought}"`,
      trace
    );
    logger.info(
      { thought, messageLength: message.length },
      "proactive message sent"
    );
  } catch (err) {
    logger.error({ err }, "proactive pipeline failed");
  }
}

// Reminders skip the judge (and the silence option) entirely: firing one is
// Corvus's own past decision that this moment was worth interrupting for
// (see REMINDER_TOOL_PROMPT), so there's nothing left to weigh — only how
// to say it. Unlike maybeVoiceThought, this ignores PROACTIVE_COOLDOWN_MS —
// a reminder is a specific commitment, not a whim — but still won't
// interrupt a live chat reply (see flushPendingReminders). This is the one
// place where a fired reminder becomes user-facing action; the plan is for
// it to eventually grow into more than a composed message (an agent loop),
// without needing to touch the thought/judge pipeline above.
async function deliverReminder(reminder) {
  if (!PROACTIVE_ENABLED) return;
  // Reminders get their own trace (distinct from the thought-tick trace
  // that resurfaced them into working memory, see checkReminders): the
  // "push llm call" that composes the delivery message is a generation
  // span inside it, and sendProactiveMessage logs the actual SSE push as
  // an event in the same trace.
  const trace = startTrace("reminder-delivery", {
    input: { reminderId: reminder.id, content: reminder.content, dueAt: reminder.due_at },
  });
  try {
    const workingMemory = await currentWorkingMemory();
    const coreMemories = await retrieveCoreMemories();
    let started = false;
    const message = await generateProactiveMessage({
      workingMemory,
      coreMemories,
      trigger: `Reminder: ${reminder.content}`,
      promptAddendum: REMINDER_DELIVERY_PROMPT,
      parent: trace,
      name: "deliver-reminder",
      onToken: (text) => {
        if (!started) {
          started = true;
          broadcast("proactive_start", {});
        }
        broadcast("proactive_token", { text });
      },
    });
    if (!message) {
      // REMINDER_DELIVERY_PROMPT forbids silence; a model that ignores that
      // is worth knowing about rather than silently swallowing the miss.
      logger.warn({ reminder: reminder.content }, "reminder delivery produced no message");
      endOk(trace, { output: { delivered: false } });
      return;
    }

    await sendProactiveMessage(
      message,
      `A reminder Corvus had set for itself just fired and was delivered: "${reminder.content}"`,
      trace
    );
    // Delivering a reminder is still "the brain speaking up," so it feeds
    // the same cooldown as an ordinary proactive message — otherwise a
    // reminder firing right before an organic thought would let both go out
    // back to back.
    lastVoiceAttemptAt = Date.now();
    logger.info(
      { reminder: reminder.content, messageLength: message.length },
      "reminder delivered"
    );
    endOk(trace, { output: { delivered: true, message } });
  } catch (err) {
    logger.error({ err, reminder: reminder.content }, "reminder delivery failed");
    endError(trace, err);
  }
}

// Reminders that fired while a chat reply was streaming: delivery is
// deferred (never queued behind a live reply) rather than dropped, and
// flushed as soon as the stream ends or the next reminder check runs.
let pendingReminderDeliveries = [];

async function flushPendingReminders() {
  if (!pendingReminderDeliveries.length || activeChatStreams > 0) return;
  const due = pendingReminderDeliveries;
  pendingReminderDeliveries = [];
  for (const reminder of due) {
    await deliverReminder(reminder);
  }
}

// A slow generation must not stack up overlapping ticks.
let ticking = false;
async function tick() {
  if (ticking) return;
  ticking = true;
  // One trace per thought-loop tick: the thought generation, the judge's
  // verdict, and (if approved) the proactive message and its push event
  // all nest under this single trace (see maybeVoiceThought).
  const trace = startTrace("thought-tick", {});
  try {
    if (!(await ensureConnected())) {
      endOk(trace, { output: { skipped: "redis not connected" } });
      return;
    }
    const generationStart = performance.now();
    // Core memories are always given to the thought generator (and later
    // the judge) directly, rather than left to randomly surface into
    // working memory like regular memories do; fetched once and reused for
    // both so it's only ever one extra query per tick.
    const [coreMemories, existingThoughts] = await Promise.all([
      retrieveCoreMemories(),
      currentWorkingMemory(),
    ]);
    const thought = await generateThought(existingThoughts, coreMemories, trace);
    const generationMs = Math.round(performance.now() - generationStart);
    await redis.set(workingMemoryKey(), thoughtEntry(thought), {
      expiration: { type: "EX", value: THOUGHT_TTL_SECONDS },
    });
    const { surfacedMemory, surfacedMessage } = await surfaceRelatedContext(thought, trace);
    // Fetched after the writes so the log shows the true current state,
    // new thought and surfaced entries included.
    const workingMemory = await currentWorkingMemory();
    logger.info(
      { thought, surfacedMemory, surfacedMessage, workingMemorySize: workingMemory.length, generationMs },
      "new thought"
    );
    // The thought and its surfaced context are in working memory, so the
    // judge sees the full picture when deciding whether to speak up.
    await maybeVoiceThought(thought, workingMemory, coreMemories, trace);
    endOk(trace, { output: { thought, surfacedMemory, surfacedMessage } });
  } catch (err) {
    logger.error({ err }, "thought tick failed");
    endError(trace, err);
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

// Polls Postgres for due reminders (set via the set_reminder tool, see
// tools/setReminder.js/db.js) and resurfaces each into working memory tagged very
// important. Firing also wakes the brain into active mode, since a
// reminder is by definition a moment Corvus decided was worth attention.
// Recurring reminders are rescheduled rather than retired (db.js#fireReminder).
let checkingReminders = false;
async function checkReminders() {
  if (checkingReminders) return;
  checkingReminders = true;
  try {
    if (!(await ensureConnected())) return;
    const due = await getDueReminders();
    for (const reminder of due) {
      await addWorkingMemoryEntry(reminderEntry(reminder.content), REMINDER_TTL_SECONDS);
      await fireReminder(reminder.id, reminder.recurrence_interval);
      pendingReminderDeliveries.push(reminder);
      logger.info(
        { id: reminder.id, content: reminder.content, recurring: !!reminder.recurrence_interval },
        "reminder fired into working memory"
      );
    }
    if (due.length) activateBrain("reminder fired");
    await flushPendingReminders();
  } catch (err) {
    logger.error({ err }, "reminder check failed");
  } finally {
    checkingReminders = false;
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

// Switches the brain to active mode (or extends the sliding window) for any
// reason worth thinking briskly about — a chat request or a reminder firing
// today. Already active: the pending tick is at most one active interval
// out, so only the window needs extending. No timer: the loop isn't running.
function activateBrain(reason) {
  const wasIdle = thoughtMode() === "idle";
  activeUntil = Date.now() + ACTIVE_WINDOW_MS;
  if (!wasIdle || !thoughtTimer) return;
  logger.info(
    { reason, activeIntervalMs: ACTIVE_THOUGHT_INTERVAL_MS, activeWindowMs: ACTIVE_WINDOW_MS },
    "brain entering active mode"
  );
  // Don't wait out the remainder of a just-scheduled idle interval.
  scheduleThought(ACTIVE_THOUGHT_INTERVAL_MS);
}

// Called by the chat route when a user message comes in.
export function notifyChatRequest() {
  activateBrain("chat request");
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
      proactiveEnabled: PROACTIVE_ENABLED,
      proactiveCooldownMs: PROACTIVE_COOLDOWN_MS,
      model: DEEPSEEK_MODEL,
    },
    "thought loop starting"
  );
  // Seed working memory with a random memory before the first thought so the
  // brain doesn't start blank; the first tick then sees the seeded memory.
  // The self-scheduling chain takes over from there.
  void pullRandomMemory()
    .then(() => checkReminders())
    .then(() => tick())
    .then(() => scheduleThought(currentThoughtInterval()));
  setInterval(() => void pullRandomMemory(), MEMORY_INTERVAL_MS).unref();
  setInterval(() => void checkReminders(), REMINDER_CHECK_INTERVAL_MS).unref();
}
