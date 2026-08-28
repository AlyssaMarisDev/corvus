import { z } from "zod";
import { HumanMessage, SystemMessage } from "@langchain/core/messages";
import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import {
  deleteMemory,
  saveMemory,
  searchMemories,
  updateMemory,
} from "./db.js";
import { MEMORY_EXTRACTOR_PROMPT } from "./prompt.js";
import { logger } from "./logger.js";

const EMBEDDING_MODEL = process.env.GEMINI_EMBEDDING_MODEL ?? "gemini-embedding-001";
// Matches the vector(768) columns in the database.
const EMBEDDING_DIMENSIONS = 768;

// LangChain's GoogleGenerativeAIEmbeddings doesn't expose
// outputDimensionality (gemini-embedding-001 defaults to 3072 dims), so call
// the embedContent REST API directly to get 768-dim vectors.
async function embed(text) {
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
  return data.embedding.values;
}

// Three separate lists with all-required fields: Gemini's structured output
// tends to silently drop optional fields, which previously produced update
// operations without their revised content.
const extractionSchema = z.object({
  save: z
    .array(z.object({ content: z.string().describe("the new memory text") }))
    .max(3)
    .describe("new facts about the user to store"),
  update: z
    .array(
      z.object({
        id: z.number().describe("id of the existing memory to revise"),
        content: z.string().describe("the complete revised memory text"),
      })
    )
    .max(3)
    .describe("existing memories that are stale or imprecise"),
  delete: z
    .array(z.object({ id: z.number().describe("id of the existing memory to remove") }))
    .max(3)
    .describe("existing memories that are contradicted or no longer true"),
});

const extractor = new ChatGoogleGenerativeAI({
  model: process.env.GEMINI_MODEL,
  apiKey: process.env.GEMINI_API_KEY,
  maxOutputTokens: 1024,
}).withStructuredOutput(extractionSchema);

// Retrieval must never break chat: any failure degrades to no memories.
export async function retrieveMemories(text) {
  try {
    const queryEmbedding = await embed(text);
    return await searchMemories(queryEmbedding);
  } catch (err) {
    logger.error({ err }, "memory retrieval failed; continuing without memories");
    return [];
  }
}

// Ids are validated against the related set so a hallucinated id can never
// touch an unrelated memory.
async function applyOperations({ save, update, delete: remove }, relatedIds) {
  for (const { content } of save) {
    await saveMemory(content, await embed(content));
  }
  for (const { id, content } of update) {
    if (relatedIds.has(id)) {
      await updateMemory(id, content, await embed(content));
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
}

// Runs after a reply completes. The extractor sees the most similar existing
// memories (with ids) so it can revise or remove contradicted ones instead of
// duplicating them.
export async function extractMemories(userMessage, assistantReply) {
  const start = performance.now();
  try {
    const exchangeEmbedding = await embed(
      `User: ${userMessage}\nAssistant: ${assistantReply}`
    );
    const related = await searchMemories(exchangeEmbedding);
    // bigserial ids come back from node-pg as strings; normalize to numbers.
    const relatedIds = new Set(related.map((m) => Number(m.id)));

    const memoryList = related.length
      ? related.map((m) => `- [id: ${m.id}] ${m.content}`).join("\n")
      : "(none)";

    const operations = await extractor.invoke([
      new SystemMessage(MEMORY_EXTRACTOR_PROMPT),
      new HumanMessage(
        `Existing related memories:\n${memoryList}\n\nLatest exchange:\nUser: ${userMessage}\nAssistant: ${assistantReply}`
      ),
    ]);

    await applyOperations(operations, relatedIds);
    logger.info(
      {
        operations:
          operations.save.length + operations.update.length + operations.delete.length,
        relatedMemories: related.length,
        durationMs: Math.round(performance.now() - start),
      },
      "memory extraction completed"
    );
  } catch (err) {
    logger.error({ err }, "memory extraction failed");
  }
}
