import {
  Annotation,
  END,
  MessagesAnnotation,
  START,
  StateGraph,
} from "@langchain/langgraph";
import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import { SystemMessage } from "@langchain/core/messages";
import { buildSystemPrompt } from "./prompt.js";
import { extractMemories, retrieveMemories } from "./memory.js";
import { logger } from "./logger.js";

const model = new ChatGoogleGenerativeAI({
  model: process.env.GEMINI_MODEL,
  apiKey: process.env.GEMINI_API_KEY,
  maxOutputTokens: 1024,
});

// Graph state: the conversation messages plus the long-term memories
// retrieved for the latest user message. conversationId rides along so node
// logs can be correlated with the request that triggered them.
const CorvusAnnotation = Annotation.Root({
  ...MessagesAnnotation.spec,
  memories: Annotation({
    reducer: (_current, next) => next,
    default: () => [],
  }),
  conversationId: Annotation({
    reducer: (_current, next) => next,
    default: () => undefined,
  }),
});

// Streaming the model explicitly (rather than invoke) so token-level
// on_chat_model_stream events reach streamEvents consumers.
async function callModel(state) {
  const start = performance.now();
  try {
    const chunks = await model.stream([
      new SystemMessage(buildSystemPrompt(state.memories)),
      ...state.messages,
    ]);
    let response;
    for await (const chunk of chunks) {
      response = response ? response.concat(chunk) : chunk;
    }
    if (!response) {
      throw new Error("model returned no content");
    }
    logger.info(
      {
        model: process.env.GEMINI_MODEL,
        inputMessages: state.messages.length,
        memories: state.memories.length,
        durationMs: Math.round(performance.now() - start),
      },
      "llm call completed"
    );
    return { messages: [response] };
  } catch (err) {
    logger.error(
      {
        err,
        model: process.env.GEMINI_MODEL,
        durationMs: Math.round(performance.now() - start),
      },
      "llm call failed"
    );
    throw err;
  }
}

function chunkText(content) {
  if (typeof content === "string") return content;
  if (Array.isArray(content)) {
    return content
      .map((part) => (part?.type === "text" ? part.text : ""))
      .join("");
  }
  return "";
}

// Populates state.memories for the corvus node. Retrieval failures degrade to
// no memories inside retrieveMemories, so this node never fails the run.
async function retrieve(state) {
  const lastMessage = state.messages[state.messages.length - 1];
  const memories = await retrieveMemories(chunkText(lastMessage?.content));
  logger.info(
    { conversationId: state.conversationId, memoryCount: memories.length },
    "memories retrieved"
  );
  return { memories };
}

// Revises long-term memory from the completed exchange. extractMemories logs
// and swallows its own errors, so this node never fails the run.
async function extract(state) {
  const reply = chunkText(state.messages[state.messages.length - 1]?.content);
  const userMessage = state.messages.findLast((m) => m._getType() === "human");
  if (!userMessage || !reply) return;
  await extractMemories(chunkText(userMessage.content), reply);
}

// Only the corvus node calls the chat model for the reply; retrieve/extract
// handle long-term memory around it.
export const graph = new StateGraph(CorvusAnnotation)
  .addNode("retrieve", retrieve)
  .addNode("corvus", callModel)
  .addNode("extract", extract)
  .addEdge(START, "retrieve")
  .addEdge("retrieve", "corvus")
  .addEdge("corvus", "extract")
  .addEdge("extract", END)
  .compile();

// Yields reply tokens as the model generates them; callers concatenate the
// chunks to reconstruct the full reply. Once the corvus node ends, the rest
// of the run (memory extraction) is drained in the background so it never
// delays the response.
export async function* streamCorvus(messages, conversationId) {
  const events = graph.streamEvents({ messages, conversationId }, { version: "v2" });
  const iterator = events[Symbol.asyncIterator]();
  let replyComplete = false;
  try {
    while (true) {
      const { value: event, done } = await iterator.next();
      if (done) return;
      if (event.event === "on_chat_model_stream") {
        const text = chunkText(event.data?.chunk?.content);
        if (text) yield text;
      } else if (event.event === "on_chain_end" && event.name === "corvus") {
        replyComplete = true;
        return;
      }
    }
  } finally {
    if (replyComplete) {
      // Keep consuming so the extract node runs to completion, detached from
      // the caller; this is the only place extraction errors would surface.
      void (async () => {
        try {
          while (!(await iterator.next()).done) {
            // discard non-token events
          }
        } catch (err) {
          logger.error({ err }, "graph run failed after reply completed");
        }
      })();
    } else {
      // Caller aborted mid-reply (e.g. client disconnect): cancel the run.
      await iterator.return?.();
    }
  }
}
