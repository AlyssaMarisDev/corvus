import {
  Annotation,
  END,
  MessagesAnnotation,
  START,
  StateGraph,
} from "@langchain/langgraph";
import { AIMessage, ToolMessage } from "@langchain/core/messages";
import { z } from "zod";
import { buildSystemPrompt } from "./prompt.js";
import { getWorkingMemory, synthesizeInteraction } from "./brain.js";
import { extractMemories, retrieveCoreMemories, retrieveMemories } from "./memory.js";
import { DEEPSEEK_MODEL, DEEPSEEK_REASONING_EFFORT, streamDeepSeekReply } from "./deepseek.js";
import { TOOLS } from "./tools/index.js";
import { logger } from "./logger.js";

// Hard cap on set_reminder/web_search/search_memory tool rounds per turn,
// independent of what the model emits. Corvus can call tools repeatedly
// (e.g. search, read the results, search again) until it's satisfied or
// this limit forces it to finalize as plain text.
const MAX_TOOL_ROUNDS = 6;

// Graph state: the conversation messages plus the long-term memories
// retrieved for the latest user message.
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
  // Counts completed set_reminder/web_search/search_memory tool rounds this
  // turn; gates whether the corvus node keeps offering tools on the next
  // pass. Corvus loops through as many rounds as it wants, up to
  // MAX_TOOL_ROUNDS, before it must finalize a plain-text reply.
  toolRounds: Annotation({
    reducer: (_current, next) => next,
    default: () => 0,
  }),
});

// zod -> OpenAI/DeepSeek function-calling spec. $schema is metadata the API
// doesn't want in "parameters".
function toFunctionSpec({ name, description, schema }) {
  const parameters = z.toJSONSchema(schema);
  delete parameters.$schema;
  return { type: "function", function: { name, description, parameters } };
}

// Raw SSE stream rather than a LangChain chat model: the thinking parameters
// and reasoning_content deltas need first-class control, and custom events
// still surface tokens and reasoning to streamEvents consumers.
async function callModel(state) {
  const start = performance.now();
  // Corvus can loop through multiple tool rounds per turn: tools stay bound
  // (see streamDeepSeekReply) and tool_choice stays "auto" so it may call
  // again after seeing a result. Once MAX_TOOL_ROUNDS is reached, tool_choice
  // flips to "none" so the model must finalize its reply as plain text.
  const tools = TOOLS.map((t) => toFunctionSpec(t.definition));
  const toolChoice = state.toolRounds >= MAX_TOOL_ROUNDS ? "none" : "auto";
  try {
    const { content, reasoning, toolCalls } = await streamDeepSeekReply(
      buildSystemPrompt({
        memories: state.memories,
        coreMemories: state.coreMemories,
        workingMemory: state.workingMemory,
        reminderToolEnabled: toolChoice === "auto",
        webSearchToolEnabled: toolChoice === "auto",
        searchMemoryToolEnabled: toolChoice === "auto",
      }),
      state.messages.map(toApiMessage),
      { tools, toolChoice }
    );
    if (!content && !toolCalls.length) {
      throw new Error("model returned no content");
    }
    logger.info(
      {
        model: DEEPSEEK_MODEL,
        reasoningEffort: DEEPSEEK_REASONING_EFFORT,
        inputMessages: state.messages.length,
        memories: state.memories.length,
        coreMemories: state.coreMemories.length,
        workingMemory: state.workingMemory.length,
        replyChars: content.length,
        reasoningChars: reasoning.length,
        toolCalls: toolCalls.length,
        durationMs: Math.round(performance.now() - start),
      },
      "llm call completed"
    );
    if (toolCalls.length) {
      // DeepSeek's thinking-mode contract: an assistant message carrying
      // tool_calls must have its reasoning_content replayed on the next
      // request in this turn (toApiMessage reads it back off here).
      return {
        messages: [
          new AIMessage({
            content: content || "",
            tool_calls: toolCalls,
            additional_kwargs: { reasoning_content: reasoning },
          }),
        ],
      };
    }
    return { messages: [new AIMessage(content)] };
  } catch (err) {
    logger.error(
      {
        err,
        model: DEEPSEEK_MODEL,
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

// LangChain messages -> OpenAI chat format. tool_calls only appear on an AI
// message within the current turn (history persisted to Postgres is always
// flattened to plain text, see index.js), so replaying reasoning_content
// only ever matters for the in-flight turn.
function toApiMessage(message) {
  const content = chunkText(message.content);
  switch (message._getType()) {
    case "human":
      return { role: "user", content };
    case "ai": {
      const toolCalls = message.tool_calls ?? [];
      if (!toolCalls.length) return { role: "assistant", content };
      const apiMessage = {
        role: "assistant",
        content: content || null,
        tool_calls: toolCalls.map((tc) => ({
          id: tc.id,
          type: "function",
          function: { name: tc.name, arguments: JSON.stringify(tc.args ?? {}) },
        })),
      };
      // Required by DeepSeek whenever a tool-calling assistant message is
      // resent (see streamDeepSeekReply's doc comment).
      const reasoningContent = message.additional_kwargs?.reasoning_content;
      if (reasoningContent) apiMessage.reasoning_content = reasoningContent;
      return apiMessage;
    }
    case "tool":
      return { role: "tool", tool_call_id: message.tool_call_id, content };
    case "system":
      return { role: "system", content };
    default:
      return { role: "user", content };
  }
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
      memoryCount: memories.length,
      coreMemoryCount: coreMemories.length,
      workingMemoryCount: workingMemory.length,
    },
    "context retrieved"
  );
  return { memories, coreMemories, workingMemory };
}

// Executes every tool call from the latest corvus pass — however many the
// model requested together, dispatched by name to the matching TOOLS entry
// (see tools/index.js) — in one round, so no tool call is ever left
// unanswered. Loops back to corvus afterward, which may call tools again
// (up to MAX_TOOL_ROUNDS) if it isn't done yet.
async function runTools(state) {
  const lastMessage = state.messages[state.messages.length - 1];
  const toolCalls = lastMessage?.tool_calls ?? [];
  const responses = await Promise.all(
    toolCalls.map((tc) => {
      const tool = TOOLS.find((t) => t.definition.name === tc.name);
      if (tool) return tool.execute(tc);
      // Any other unrecognized name still needs a response to keep the
      // tool-call/tool-response history valid for the next corvus call.
      logger.warn({ name: tc.name }, "unhandled tool call");
      return new ToolMessage({ content: `Unknown tool "${tc.name}".`, tool_call_id: tc.id });
    })
  );
  return { messages: responses, toolRounds: state.toolRounds + 1 };
}

function routeAfterCorvus(state) {
  const lastMessage = state.messages[state.messages.length - 1];
  const toolCalls = lastMessage?.tool_calls ?? [];
  // Any known tool call (see tools/index.js), however many were requested
  // together, goes to runTools; callModel's MAX_TOOL_ROUNDS cap is what
  // eventually forces this to fall through to extract.
  if (toolCalls.some((tc) => TOOLS.some((t) => t.definition.name === tc.name))) {
    return "tools";
  }
  return "extract";
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
// handle long-term memory around it, and tools executes any set_reminder/
// web_search/search_memory calls. Every tool round loops back to corvus,
// which decides whether to call again, up to MAX_TOOL_ROUNDS, or finalize
// its reply.
export const graph = new StateGraph(CorvusAnnotation)
  .addNode("retrieve", retrieve)
  .addNode("corvus", callModel)
  .addNode("tools", runTools)
  .addNode("extract", extract)
  .addEdge(START, "retrieve")
  .addEdge("retrieve", "corvus")
  .addConditionalEdges("corvus", routeAfterCorvus, {
    tools: "tools",
    extract: "extract",
  })
  .addEdge("tools", "corvus")
  .addEdge("extract", END)
  .compile();

// Yields { type: "token" | "status", text } events: reply tokens as DeepSeek
// generates them, plus discrete tool-activity announcements as status events
// (e.g. webSearch's "Searching the web…" line, or search_memory's
// "Searching memory…" line) — never raw reasoning, which deepseek.js keeps
// internal. Callers concatenate token text to reconstruct the full reply.
// Once the final corvus node ends, the rest of the run (memory extraction)
// is drained in the background so it never delays the response.
export async function* streamCorvus(messages) {
  const events = graph.streamEvents({ messages }, { version: "v2" });
  const iterator = events[Symbol.asyncIterator]();
  let replyComplete = false;
  try {
    while (true) {
      const { value: event, done } = await iterator.next();
      if (done) return;
      // The corvus node streams over raw SSE, so tokens and reasoning arrive
      // as custom events rather than on_chat_model_stream.
      if (event.event === "on_custom_event" && event.name === "corvus_token") {
        const text = event.data?.text;
        if (text) yield { type: "token", text };
      } else if (
        event.event === "on_custom_event" &&
        event.name === "corvus_status"
      ) {
        const text = event.data?.text;
        if (text) yield { type: "status", text };
      } else if (event.event === "on_chain_end" && event.name === "corvus") {
        // The corvus node runs once per tool round plus a final pass on a
        // turn with tool calls; every run but the last ends in a tool call,
        // so only the run without tool calls is the reply.
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
