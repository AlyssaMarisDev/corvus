// The save_memory tool: lets Corvus write a long-term memory for itself on
// demand, independent of the automatic memory extractor (memory.js's
// extractMemories, which still runs after every exchange regardless). Use
// this for something worth remembering that the extractor might not catch
// from the exchange alone, or that's worth saving before the turn even
// ends. Wired into the live DeepSeek tool-calling path alongside
// set_reminder/web_search/search_memory (see tools/index.js and agent.js).
import { z } from "zod";
import { ToolMessage } from "@langchain/core/messages";
import { CORVUS_SUBJECT, saveMemory } from "../db.js";
import { embed } from "../memory.js";
import { logger } from "../logger.js";

// Mirrors memory.js's subjectField (the memory extractor's own schema):
// same semantics, so a memory this tool saves is indistinguishable from one
// the extractor would have saved. Uses "" rather than null for "no
// subject" for the same reason as memory.js's subjectField: some models'
// structured-output/function-calling schema conversion (e.g. Gemini's
// response_schema) can't handle the "type": ["string", "null"] that Zod's
// .nullable() produces.
const subjectField = z
  .string()
  .describe(
    "who/what this fact is about: the current speaker's name, another named entity " +
      `(e.g. a pet or another person), the literal "${CORVUS_SUBJECT}" for a stable fact ` +
      'about Corvus itself, or "" for a general fact'
  );

export const saveMemoryTool = {
  name: "save_memory",
  description:
    "Save a long-term memory right now, in your own words. Independent of your automatic memory extraction, which still runs after every exchange — use this when something is worth remembering immediately, or that automatic extraction might miss (something you noticed or concluded rather than something stated outright, or a detail from earlier in the conversation).",
  schema: z.object({
    content: z
      .string()
      .describe(
        "the complete, self-contained memory text, naming its subject explicitly (e.g. " +
          '"Bree is a software engineer and works on AI agents", not "Works on AI agents")'
      ),
    subject: subjectField,
    core: z
      .boolean()
      .describe(
        "true to save this as a core profile memory — always shown in the system prompt no " +
          "matter who's speaking. Reserve for stable identity facts or explicit interaction " +
          `preferences about the current speaker, or stable facts about Corvus itself (subject ` +
          `"${CORVUS_SUBJECT}"). false for everything else, which is most memories.`
      ),
  }),
};

// Executes a single save_memory tool call: embeds the content and inserts
// it via db.js's saveMemory (the same insert path the extractor uses), then
// answers with a confirmation so the next corvus call sees valid
// tool-call/tool-response history. A malformed call gets an error
// ToolMessage back instead of throwing the call out of the turn.
export async function executeSaveMemory(tc, parent) {
  const parsed = saveMemoryTool.schema.safeParse(tc.args);
  if (!parsed.success) {
    logger.warn({ args: tc.args, error: parsed.error.message }, "invalid save_memory call");
    return new ToolMessage({
      content: `Could not save that memory: ${parsed.error.issues[0]?.message ?? "invalid arguments"}.`,
      tool_call_id: tc.id,
    });
  }
  const { content, subject, core } = parsed.data;
  const embedding = await embed(content, parent);
  await saveMemory(content, embedding, core ? "core" : null, subject || null);
  logger.info({ content, subject, core }, "memory saved via save_memory tool");
  return new ToolMessage({
    content: `Saved${core ? " as a core memory" : ""}${subject ? ` (subject: ${subject})` : ""}.`,
    tool_call_id: tc.id,
  });
}
