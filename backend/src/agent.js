import {
  Annotation,
  END,
  MessagesAnnotation,
  START,
  StateGraph,
} from "@langchain/langgraph";
import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import { HumanMessage, SystemMessage, ToolMessage } from "@langchain/core/messages";
import { dispatchCustomEvent } from "@langchain/core/callbacks/dispatch";
import { z } from "zod";
import { buildSystemPrompt } from "./prompt.js";
import { deepThinkGraph } from "./deepthink.js";
import { getWorkingMemory, synthesizeInteraction } from "./brain.js";
import { extractMemories, retrieveCoreMemories, retrieveMemories } from "./memory.js";
import { logger } from "./logger.js";

// Deep recall is disabled while the working-memory integration is exercised
// in isolation; flip to true to re-enable the think_deeper tool and the
// deep-think subgraph.
const DEEP_THINK_ENABLED = false;

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
  coreMemories: Annotation({
    reducer: (_current, next) => next,
    default: () => [],
  }),
  // The thought loop's Redis working memory, injected into the system prompt
  // alongside the retrieved memories.
  workingMemory: Annotation({
    reducer: (_current, next) => next,
    default: () => [],
  }),
  conversationId: Annotation({
    reducer: (_current, next) => next,
    default: () => undefined,
  }),
  // Set once the deep-think subgraph has run this turn; gates both the
  // routing and whether the corvus node offers the think_deeper tool.
  deepThinkUsed: Annotation({
    reducer: (_current, next) => next,
    default: () => false,
  }),
});

// Offered to the corvus node only until the subgraph has been used, so deep
// recall can fire at most once per turn.
const thinkDeeper = {
  name: "think_deeper",
  description:
    "Deep-recall search over the user's long-term memories, including deleted ones. Findings tag each memory as [active] or [deleted] (outdated or contradicted). Call only when the conversation and the provided memories do not suffice to answer.",
  schema: z.object({
    directive: z
      .string()
      .describe("self-contained instruction for the deep-recall planner"),
    status: z
      .string()
      .describe("short thinking-out-loud line shown to the user while searching"),
  }),
};

// Streaming the model explicitly (rather than invoke) so token-level
// on_chat_model_stream events reach streamEvents consumers.
async function callModel(state) {
  const start = performance.now();
  try {
    // After the subgraph has run, the model must answer from the findings, so
    // no tools are offered on the second call.
    const activeModel =
      DEEP_THINK_ENABLED && !state.deepThinkUsed ? model.bindTools([thinkDeeper]) : model;
    const chunks = await activeModel.stream([
      new SystemMessage(
        buildSystemPrompt({
          memories: state.memories,
          coreMemories: state.coreMemories,
          workingMemory: state.workingMemory,
          deepThinkEnabled: DEEP_THINK_ENABLED,
        })
      ),
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
        coreMemories: state.coreMemories.length,
        workingMemory: state.workingMemory.length,
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

// Populates state.memories, state.coreMemories, and state.workingMemory for
// the corvus node. Retrieval failures degrade to empty lists inside
// retrieveMemories, retrieveCoreMemories, and getWorkingMemory, so this node
// never fails the run.
async function retrieve(state) {
  const lastMessage = state.messages[state.messages.length - 1];
  const [memories, coreMemories, workingMemory] = await Promise.all([
    retrieveMemories(chunkText(lastMessage?.content)),
    retrieveCoreMemories(),
    getWorkingMemory(),
  ]);
  logger.info(
    {
      conversationId: state.conversationId,
      memoryCount: memories.length,
      coreMemoryCount: coreMemories.length,
      workingMemoryCount: workingMemory.length,
    },
    "context retrieved"
  );
  return { memories, coreMemories, workingMemory };
}

// Runs the deep-recall subgraph for the think_deeper tool call and answers
// the tool call with the consolidated findings, so the next corvus call sees
// valid tool-call/tool-response history.
async function deepThink(state) {
  const lastMessage = state.messages[state.messages.length - 1];
  const toolCalls = lastMessage?.tool_calls ?? [];
  const call = toolCalls.find((tc) => tc.name === "think_deeper") ?? toolCalls[0];
  const { directive, status } = call.args;

  logger.info({ conversationId: state.conversationId, directive }, "deep-think subgraph invoked");
  await dispatchCustomEvent("corvus_status", { text: status });

  // conversationId travels via configurable so the fetch_past_conversations
  // tool can exclude the current conversation from its search.
  const result = await deepThinkGraph.invoke(
    {
      directive,
      messages: [new HumanMessage(directive)],
    },
    { configurable: { conversationId: state.conversationId } }
  );
  const summary =
    chunkText(result.messages[result.messages.length - 1]?.content) ||
    "Nothing relevant was found in long-term memory.";

  // Gemini requires a ToolMessage for every tool call; the subgraph runs
  // once, so any extra parallel calls get a note instead of a second run.
  const responses = toolCalls.map(
    (tc) =>
      new ToolMessage({
        content:
          tc.id === call.id
            ? summary
            : "Deep recall was already invoked for this turn.",
        tool_call_id: tc.id,
      })
  );
  return { messages: responses, deepThinkUsed: true };
}

function routeAfterCorvus(state) {
  if (!DEEP_THINK_ENABLED || state.deepThinkUsed) return "extract";
  const lastMessage = state.messages[state.messages.length - 1];
  const wantsDeepThink = lastMessage?.tool_calls?.some(
    (tc) => tc.name === "think_deeper"
  );
  return wantsDeepThink ? "deepThink" : "extract";
}

// Revises long-term memory from the completed exchange and synthesizes it
// into working memory as an interaction anchor. Both steps log and swallow
// their own errors, so this node never fails the run.
async function extract(state) {
  const reply = state.messages[state.messages.length - 1];
  const userMessage = state.messages.findLast((m) => m._getType() === "human");
  const work = [extractMemories(state.messages)];
  if (userMessage && reply?._getType() === "ai" && chunkText(reply.content)) {
    work.push(synthesizeInteraction(chunkText(userMessage.content), chunkText(reply.content)));
  }
  await Promise.all(work);
}

// Only the corvus node calls the chat model for the reply; retrieve/extract
// handle long-term memory around it, and deepThink runs the recall subgraph
// when corvus chooses think_deeper.
export const graph = new StateGraph(CorvusAnnotation)
  .addNode("retrieve", retrieve)
  .addNode("corvus", callModel)
  .addNode("deepThink", deepThink)
  .addNode("extract", extract)
  .addEdge(START, "retrieve")
  .addEdge("retrieve", "corvus")
  .addConditionalEdges("corvus", routeAfterCorvus, {
    deepThink: "deepThink",
    extract: "extract",
  })
  .addEdge("deepThink", "corvus")
  .addEdge("extract", END)
  .compile();

// Yields { type: "token" | "status", text } events: reply tokens as the model
// generates them, plus the thinking-out-loud status when corvus invokes deep
// recall. Callers concatenate token text to reconstruct the full reply. Once
// the final corvus node ends, the rest of the run (memory extraction) is
// drained in the background so it never delays the response.
export async function* streamCorvus(messages, conversationId) {
  const events = graph.streamEvents({ messages, conversationId }, { version: "v2" });
  const iterator = events[Symbol.asyncIterator]();
  let replyComplete = false;
  try {
    while (true) {
      const { value: event, done } = await iterator.next();
      if (done) return;
      if (
        event.event === "on_chat_model_stream" &&
        event.metadata?.langgraph_node === "corvus"
      ) {
        const text = chunkText(event.data?.chunk?.content);
        if (text) yield { type: "token", text };
      } else if (
        event.event === "on_custom_event" &&
        event.name === "corvus_status"
      ) {
        const text = event.data?.text;
        if (text) yield { type: "status", text };
      } else if (event.event === "on_chain_end" && event.name === "corvus") {
        // The corvus node runs twice on a deep-think turn; the first run ends
        // in a tool call, so only the run without tool calls is the reply.
        const outMessages = event.data?.output?.messages ?? [];
        const lastOut = outMessages[outMessages.length - 1];
        const outToolCalls = lastOut?.tool_calls ?? lastOut?.kwargs?.tool_calls ?? [];
        if (outToolCalls.length) continue;
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
