import { z } from "zod";
import { HumanMessage, SystemMessage } from "@langchain/core/messages";
import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import {
  CORVUS_SUBJECT,
  DEFAULT_SUBJECT,
  deleteMemory,
  getMemoriesByTag,
  saveMemory,
  searchMemories,
  searchMessages,
  updateMemory,
} from "./db.js";
import { MEMORY_EXTRACTOR_PROMPT, formatMemoryTimestamp } from "./prompt.js";
import { endError, endOk, startChild } from "./tracing.js";
import { logger } from "./logger.js";

const EMBEDDING_MODEL = process.env.GEMINI_EMBEDDING_MODEL ?? "gemini-embedding-001";
// Matches the vector(768) columns in the database.
const EMBEDDING_DIMENSIONS = 768;

// LangChain's GoogleGenerativeAIEmbeddings doesn't expose
// outputDimensionality (gemini-embedding-001 defaults to 3072 dims), so call
// the embedContent REST API directly to get 768-dim vectors.
export async function embed(text, parent) {
  const generation = startChild(
    parent,
    "gemini-embed",
    { model: EMBEDDING_MODEL, input: text },
    "embedding"
  );
  try {
    const response = await fetch(
      `https://generativelanguage.googleapis.com/v1beta/models/${EMBEDDING_MODEL}:embedContent`,
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "x-goog-api-key": process.env.GEMINI_API_KEY,
        },
        body: JSON.stringify({
          model: `models/${EMBEDDING_MODEL}`,
          content: { parts: [{ text }] },
          outputDimensionality: EMBEDDING_DIMENSIONS,
        }),
      }
    );
    if (!response.ok) {
      throw new Error(`embedContent failed: ${response.status} ${await response.text()}`);
    }
    const data = await response.json();
    endOk(generation, { output: { dimensions: data.embedding.values.length } });
    return data.embedding.values;
  } catch (err) {
    endError(generation, err);
    throw err;
  }
}

// subject is who/what a fact is about: the current speaker's name (see
// memory_extractor's "Current speaker" line), another specific entity named
// in the conversation (e.g. "Quentin"), or "" for a general fact tied to no
// one in particular. Required (not nullable/optional): Gemini's structured
// output tends to silently drop optional fields, and — worse — its
// response_schema proto rejects the "type": ["string", "null"] that Zod's
// .nullable() produces for this field (400 Bad Request: "Proto field is not
// repeating, cannot start list"). Empty string is the required-field-safe
// stand-in for null; normalizeSubject() below converts it back.
const subjectField = z
  .string()
  .describe(
    "who/what this fact is about: the current speaker's name, another named entity " +
      `(e.g. a pet or another person), the literal "${CORVUS_SUBJECT}" for a stable fact ` +
      'about Corvus itself, or "" for a general fact'
  );

// Maps the schema's "" sentinel (see subjectField above) back to null for
// storage/lookup, which is what db.js's subject columns actually use.
function normalizeSubject(subject) {
  return subject ? subject : null;
}

// Three separate lists with all-required fields: Gemini's structured output
// tends to silently drop optional fields, which previously produced update
// operations without their revised content.
const extractionSchema = z.object({
  save: z
    .array(z.object({ content: z.string().describe("the new memory text"), subject: subjectField }))
    .max(3)
    .describe("new facts to store"),
  update: z
    .array(
      z.object({
        id: z.number().describe("id of the existing memory to revise"),
        content: z.string().describe("the complete revised memory text"),
        subject: subjectField,
      })
    )
    .max(3)
    .describe("existing memories that are stale or imprecise"),
  delete: z
    .array(z.object({ id: z.number().describe("id of the existing memory to remove") }))
    .max(3)
    .describe("existing memories that are contradicted or no longer true"),
  saveCore: z
    .array(z.object({ content: z.string().describe("the new core memory text"), subject: subjectField }))
    .max(2)
    .describe("new core profile facts to store"),
  updateCore: z
    .array(
      z.object({
        id: z.number().describe("id of the existing core memory to revise"),
        content: z.string().describe("the complete revised core memory text"),
        subject: subjectField,
      })
    )
    .max(2)
    .describe("existing core memories that are stale or imprecise"),
  deleteCore: z
    .array(z.object({ id: z.number().describe("id of the existing core memory to remove") }))
    .max(2)
    .describe("existing core memories that are contradicted or no longer true"),
});

const extractor = new ChatGoogleGenerativeAI({
  model: process.env.GEMINI_MODEL,
  apiKey: process.env.GEMINI_API_KEY,
  // Thinking models burn reasoning tokens against this budget; 1024 truncated
  // the JSON mid-string (OutputParserException) whenever extraction thought hard.
  maxOutputTokens: 8192,
}).withStructuredOutput(extractionSchema);

// Retrieval must never break chat: any failure degrades to no memories.
export async function retrieveMemories(text, parent) {
  try {
    const queryEmbedding = await embed(text, parent);
    return await searchMemories(queryEmbedding);
  } catch (err) {
    logger.error({ err }, "memory retrieval failed; continuing without memories");
    return [];
  }
}

// Core memories are always injected into the system prompt, so this fetches
// every active row rather than similarity-searching. Scoped to `subject`
// (the current speaker) plus subject-less general core facts — see
// getMemoriesByTag.
export async function retrieveCoreMemories(subject = DEFAULT_SUBJECT) {
  try {
    return await getMemoriesByTag("core", { subject });
  } catch (err) {
    logger.error({ err }, "core memory retrieval failed; continuing without them");
    return [];
  }
}

// Used by the search_memory tool to search past conversation messages.
// Same never-throws contract as retrieveMemories.
export async function retrievePastConversations(text, parent) {
  try {
    const queryEmbedding = await embed(text, parent);
    return await searchMessages(queryEmbedding);
  } catch (err) {
    logger.error({ err }, "past conversation retrieval failed; continuing without it");
    return [];
  }
}

// Ids are validated against the related set so a hallucinated id can never
// touch an unrelated memory.
async function applyOperations(
  { save, update, delete: remove, saveCore, updateCore, deleteCore },
  relatedIds,
  coreIds
) {
  for (const { content, subject } of save) {
    await saveMemory(content, await embed(content), null, normalizeSubject(subject));
  }
  for (const { id, content, subject } of update) {
    if (relatedIds.has(id)) {
      await updateMemory(id, content, await embed(content), normalizeSubject(subject));
    } else {
      logger.warn({ id }, "skipping update for unknown memory id");
    }
  }
  for (const { id } of remove) {
    if (relatedIds.has(id)) {
      await deleteMemory(id);
    } else {
      logger.warn({ id }, "skipping delete for unknown memory id");
    }
  }
  for (const { content, subject } of saveCore) {
    await saveMemory(content, await embed(content), "core", normalizeSubject(subject));
  }
  for (const { id, content, subject } of updateCore) {
    if (coreIds.has(id)) {
      await updateMemory(id, content, await embed(content), normalizeSubject(subject));
    } else {
      logger.warn({ id }, "skipping update for unknown core memory id");
    }
  }
  for (const { id } of deleteCore) {
    if (coreIds.has(id)) {
      await deleteMemory(id);
    } else {
      logger.warn({ id }, "skipping delete for unknown core memory id");
    }
  }
}

function messageText(message) {
  const content = message?.content;
  if (typeof content === "string") return content;
  if (Array.isArray(content)) {
    return content
      .map((part) => (part?.type === "text" ? part.text : ""))
      .join("");
  }
  return "";
}

function formatTranscript(messages) {
  return messages
    .filter((m) => ["human", "ai"].includes(m._getType()))
    .map((m) => `${m._getType() === "human" ? "User" : "Butler"}: ${messageText(m)}`)
    .join("\n");
}

// Renders an existing memory row for the extractor's prompt, including its
// subject so the model can tell who/what each candidate for update/delete is
// currently about.
function formatExistingMemory(m) {
  const subjectLabel = m.subject ? `subject: ${m.subject}` : "subject: none (general)";
  return `- [id: ${m.id}] (${subjectLabel}) (last updated: ${formatMemoryTimestamp(m.updated_at)}) ${m.content}`;
}

// Runs after a reply completes with the full conversation. The extractor sees
// the whole transcript for context, but the related-memory search stays keyed
// on the latest exchange: that is where new facts appear, and embedding the
// entire history would dilute the similarity match. `subject` identifies who
// is currently speaking with Corvus (see db.js's DEFAULT_SUBJECT/subjects),
// so the extractor can label first-person facts with the right name instead
// of a generic "the user".
export async function extractMemories(messages, subject = DEFAULT_SUBJECT, parent) {
  const start = performance.now();
  let generation;
  try {
    const reply = messages[messages.length - 1];
    const userMessage = messages.findLast((m) => m._getType() === "human");
    if (!userMessage || reply?._getType() !== "ai" || !messageText(reply)) return;

    const exchangeEmbedding = await embed(
      `User: ${messageText(userMessage)}\nButler: ${messageText(reply)}`,
      parent
    );
    const [related, coreMemories] = await Promise.all([
      searchMemories(exchangeEmbedding),
      getMemoriesByTag("core", { subject }),
    ]);
    // bigserial ids come back from node-pg as strings; normalize to numbers.
    const relatedIds = new Set(related.map((m) => Number(m.id)));
    const coreIds = new Set(coreMemories.map((m) => Number(m.id)));

    const memoryList = related.length
      ? related.map(formatExistingMemory).join("\n")
      : "(none)";

    const coreList = coreMemories.length
      ? coreMemories.map(formatExistingMemory).join("\n")
      : "(none)";

    const humanContent = `Current speaker: ${subject}\nReserved subject for facts about Corvus itself: ${CORVUS_SUBJECT}\n\nExisting core profile memories (for the current speaker, plus general and Corvus-subject ones):\n${coreList}\n\nExisting related memories (any subject):\n${memoryList}\n\nConversation:\n${formatTranscript(messages)}`;
    generation = startChild(
      parent,
      "extract-memories",
      {
        model: process.env.GEMINI_MODEL,
        input: [
          { role: "system", content: MEMORY_EXTRACTOR_PROMPT },
          { role: "user", content: humanContent },
        ],
      },
      "generation"
    );
    const operations = await extractor.invoke([
      new SystemMessage(MEMORY_EXTRACTOR_PROMPT),
      new HumanMessage(humanContent),
    ]);
    endOk(generation, { output: operations });

    await applyOperations(operations, relatedIds, coreIds);
    logger.info(
      {
        operations:
          operations.save.length +
          operations.update.length +
          operations.delete.length +
          operations.saveCore.length +
          operations.updateCore.length +
          operations.deleteCore.length,
        relatedMemories: related.length,
        coreMemories: coreMemories.length,
        durationMs: Math.round(performance.now() - start),
      },
      "memory extraction completed"
    );
  } catch (err) {
    if (generation) endError(generation, err);
    logger.error({ err }, "memory extraction failed");
  }
}
