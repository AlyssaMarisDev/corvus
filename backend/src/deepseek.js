import { dispatchCustomEvent } from "@langchain/core/callbacks/dispatch";
import { logger } from "./logger.js";

export const DEEPSEEK_MODEL = process.env.DEEPSEEK_MODEL ?? "deepseek-v4-flash";
// Response generation thinks harder than the brain's background loops (which
// run with thinking disabled). One of low | high | max.
export const DEEPSEEK_REASONING_EFFORT = process.env.DEEPSEEK_REASONING_EFFORT ?? "high";
// Reasoning tokens share the completion budget with the reply, so the cap
// sits far above what an answer needs — a too-small budget can end the turn
// mid-reasoning and return no content at all.
const MAX_REPLY_TOKENS = 8192;

// Streams a chat completion from DeepSeek with thinking enabled. With
// emitEvents (the chat path, inside a graph run), answer deltas surface as
// corvus_token custom events for streamEvents consumers; reasoning deltas
// are accumulated and returned but never dispatched as events — raw
// chain-of-thought stays internal (see the reasoning_content branch below).
// The proactive path runs outside any graph run, so it disables events and
// receives answer deltas via onToken instead. tools takes OpenAI-format
// function specs ({type: "function", function: {name, description,
// parameters}}); when omitted no tools are
// offered and toolCalls is always empty. Per DeepSeek's thinking-mode +
// tool-calling contract, any assistant message being resent that carries
// tool_calls must also carry its original reasoning_content (messages
// without tool_calls need none) — callers that replay a tool-calling turn
// are responsible for attaching it (see agent.js).
// DeepSeek's OpenAI-shaped usage object -> Langfuse's usageDetails shape.
// Returns undefined when no usage arrived (e.g. stream_options wasn't
// honored), so callers can spread it in without emitting an empty object.
export function toUsageDetails(usage) {
  if (!usage) return undefined;
  return {
    input: usage.prompt_tokens,
    output: usage.completion_tokens,
    total: usage.total_tokens,
    ...(usage.prompt_cache_hit_tokens != null
      ? { cache_read_input_tokens: usage.prompt_cache_hit_tokens }
      : {}),
  };
}

export async function streamDeepSeekReply(
  systemPrompt,
  messages,
  { emitEvents = true, onToken, tools, toolChoice = "auto" } = {}
) {
  const response = await fetch("https://api.deepseek.com/chat/completions", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${process.env.DEEPSEEK_API_KEY}`,
    },
    body: JSON.stringify({
      model: DEEPSEEK_MODEL,
      messages: [{ role: "system", content: systemPrompt }, ...messages],
      thinking: { type: "enabled" },
      reasoning_effort: DEEPSEEK_REASONING_EFFORT,
      max_tokens: MAX_REPLY_TOKENS,
      stream: true,
      // Asks the API to emit a final chunk carrying token usage (see the
      // usage capture below) — used to populate Langfuse's usageDetails
      // (see agent.js/brain.js callers).
      stream_options: { include_usage: true },
      // tools stays bound (with tool_choice "none") rather than omitted once
      // a tool has already been used this turn: history may already contain
      // a tool-calling assistant message, and dropping tools entirely risks
      // the API no longer recognizing the tool name it called earlier.
      ...(tools?.length ? { tools, tool_choice: toolChoice } : {}),
    }),
    // High-effort reasoning can think for a while before the first token.
    signal: AbortSignal.timeout(180_000),
  });
  if (!response.ok) {
    throw new Error(`DeepSeek chat failed: ${response.status} ${await response.text()}`);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let content = "";
  let reasoning = "";
  // Populated once the stream_options: include_usage chunk arrives
  // (typically last, with an empty choices array) — passed back to the
  // caller for Langfuse's usageDetails.
  let usage = null;
  // Sparse, indexed by the delta's tool_call index; name/id arrive once,
  // argument fragments accumulate across chunks.
  const toolCallDeltas = [];
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    // SSE events are separated by a blank line; only data: lines carry payloads.
    let boundary;
    while ((boundary = buffer.indexOf("\n\n")) !== -1) {
      const rawEvent = buffer.slice(0, boundary);
      buffer = buffer.slice(boundary + 2);
      for (const line of rawEvent.split("\n")) {
        if (!line.startsWith("data:")) continue;
        const data = line.slice(5).trim();
        if (data === "[DONE]") continue;
        const parsed = JSON.parse(data);
        if (parsed.usage) usage = parsed.usage;
        const delta = parsed.choices?.[0]?.delta ?? {};
        if (delta.reasoning_content) {
          // Kept for logging and for the reasoning_content replay contract
          // (see agent.js's toApiMessage), but never surfaced as a
          // corvus_status event — raw chain-of-thought is internal
          // scratchpad, not a user-facing status line. Discrete tool-status
          // announcements (webSearch, searchMemory) dispatch corvus_status
          // themselves instead.
          reasoning += delta.reasoning_content;
        }
        if (delta.content) {
          content += delta.content;
          if (emitEvents) {
            await dispatchCustomEvent("corvus_token", { text: delta.content });
          }
          onToken?.(delta.content);
        }
        for (const tc of delta.tool_calls ?? []) {
          const i = tc.index ?? 0;
          const acc = (toolCallDeltas[i] ??= { id: "", name: "", args: "" });
          if (tc.id) acc.id = tc.id;
          if (tc.function?.name) acc.name = tc.function.name;
          if (tc.function?.arguments) acc.args += tc.function.arguments;
        }
      }
    }
  }
  const toolCalls = toolCallDeltas.filter(Boolean).map((tc) => {
    let args = {};
    try {
      args = tc.args ? JSON.parse(tc.args) : {};
    } catch (err) {
      logger.warn({ err, raw: tc.args }, "tool call arguments failed to parse");
    }
    return { id: tc.id, name: tc.name, args };
  });
  return { content, reasoning, toolCalls, usage };
}
