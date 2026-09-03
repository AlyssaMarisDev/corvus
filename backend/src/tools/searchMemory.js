// The search_memory tool: searches both long-term memories and past
// conversation messages by semantic similarity to a single query, so Corvus
// can look something up without waiting for it to surface on its own (see
// retrieveMemories/retrievePastConversations in memory.js). Wired into the
// live DeepSeek tool-calling path alongside set_reminder and web_search
// (see tools/index.js and agent.js).
import { z } from "zod";
import { ToolMessage } from "@langchain/core/messages";
import { dispatchCustomEvent } from "@langchain/core/callbacks/dispatch";
import { retrieveMemories, retrievePastConversations } from "../memory.js";
import { formatMemoryTimestamp } from "../prompt.js";
import { logger } from "../logger.js";

export const searchMemoryTool = {
  name: "search_memory",
  description:
    "Search your long-term memories about the user and their past conversation messages by semantic similarity to a short query. Returns whatever matches from either. Use it when the conversation and the memories already provided to you don't cover what's being asked.",
  schema: z.object({
    query: z.string().describe("short search phrase to match memories and past messages against"),
  }),
};

function formatMemoryResults(memories) {
  if (!memories.length) return "No matching long-term memories found.";
  return memories
    .map((m) => `- ${m.content} (last updated: ${formatMemoryTimestamp(m.updated_at)})`)
    .join("\n");
}

// Long messages are truncated so a single hit cannot flood the model's
// context.
function formatConversationResults(messages) {
  if (!messages.length) return "No matching past conversation messages found.";
  return messages
    .map((m) => {
      const content = m.content.length > 500 ? `${m.content.slice(0, 500)}…` : m.content;
      return `- [${m.role}] ${formatMemoryTimestamp(m.created_at)}: ${content}`;
    })
    .join("\n");
}

// Executes a single search_memory tool call: runs the query against both
// long-term memories and past conversation messages (retrieveMemories/
// retrievePastConversations, memory.js — neither ever throws, degrading to
// no results instead) and answers with both sets of findings so the next
// corvus call sees valid tool-call/tool-response history. Dispatches a
// corvus_status announcement, same pattern as executeWebSearch.
export async function executeSearchMemory(tc) {
  const parsed = searchMemoryTool.schema.safeParse(tc.args);
  if (!parsed.success) {
    logger.warn({ args: tc.args, error: parsed.error.message }, "invalid search_memory call");
    return new ToolMessage({
      content: `Could not search: ${parsed.error.issues[0]?.message ?? "invalid arguments"}.`,
      tool_call_id: tc.id,
    });
  }
  const { query } = parsed.data;
  await dispatchCustomEvent("corvus_status", { text: `Searching memory for "${query}"…` });
  const [memories, pastMessages] = await Promise.all([
    retrieveMemories(query),
    retrievePastConversations(query),
  ]);
  logger.info(
    { query, memories: memories.length, pastMessages: pastMessages.length },
    "memory search completed"
  );
  const content = `Memories:\n${formatMemoryResults(memories)}\n\nPast conversation messages:\n${formatConversationResults(pastMessages)}`;
  return new ToolMessage({ content, tool_call_id: tc.id });
}
