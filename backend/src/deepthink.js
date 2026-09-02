import {
  Annotation,
  END,
  MessagesAnnotation,
  START,
  StateGraph,
} from "@langchain/langgraph";
import { ToolNode } from "@langchain/langgraph/prebuilt";
import { tool } from "@langchain/core/tools";
import { HumanMessage, SystemMessage } from "@langchain/core/messages";
import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import { z } from "zod";
import {
  retrieveDeletedMemories,
  retrieveMemories,
  retrievePastConversations,
} from "./memory.js";
import { PLANNER_PROMPT, formatMemoryTimestamp } from "./prompt.js";
import { logger } from "./logger.js";

// Tool-execution rounds are capped; afterwards the planner is invoked without
// tools so it must consolidate whatever it has gathered.
const MAX_TOOL_ROUNDS = 3;

const model = new ChatGoogleGenerativeAI({
  model: process.env.GEMINI_MODEL,
  apiKey: process.env.GEMINI_API_KEY,
  maxOutputTokens: 1024,
});

function formatActive(memories) {
  if (!memories.length) return "No active memories found.";
  return memories
    .map((m) => `- [active] ${m.content} (last updated: ${formatMemoryTimestamp(m.updated_at)})`)
    .join("\n");
}

function formatDeleted(memories) {
  const lines = memories.map(
    (m) =>
      `- [deleted] ${m.content} (deleted: ${formatMemoryTimestamp(m.deleted_at)})${m.tag === "core" ? " [core profile fact]" : ""}`
  );
  if (!lines.length) return "No deleted memories found.";
  return lines.join("\n");
}

// Long messages are truncated so a single hit cannot flood the planner's
// context; the conversation id prefix lets the planner tell sources apart.
function formatConversations(messages) {
  if (!messages.length) return "No past conversation messages found.";
  return messages
    .map((m) => {
      const content =
        m.content.length > 500 ? `${m.content.slice(0, 500)}…` : m.content;
      return `- [${m.role}] ${formatMemoryTimestamp(m.created_at)} (conversation ${String(m.conversation_id).slice(0, 8)}): ${content}`;
    })
    .join("\n");
}

const fetchMemories = tool(
  async ({ query }) => formatActive(await retrieveMemories(query)),
  {
    name: "fetch_memories",
    description:
      "Search the user's active long-term memories by semantic similarity to a short query phrase.",
    schema: z.object({
      query: z.string().describe("short search phrase to match memories against"),
    }),
  }
);

const fetchDeletedMemories = tool(
  async ({ query }) => formatDeleted(await retrieveDeletedMemories(query)),
  {
    name: "fetch_deleted_memories",
    description:
      "Search the user's deleted (outdated or contradicted) long-term memories by semantic similarity, including deleted core profile facts. Useful when the directive concerns something that may have changed.",
    schema: z.object({
      query: z.string().describe("short search phrase to match deleted memories against"),
    }),
  }
);

// config.configurable.conversationId is set by the deepThink node so the
// current conversation (already in the model's context) is excluded.
const fetchPastConversations = tool(
  async ({ query }, config) =>
    formatConversations(
      await retrievePastConversations(query, {
        excludeConversationId: config?.configurable?.conversationId,
      })
    ),
  {
    name: "fetch_past_conversations",
    description:
      "Search messages from the user's past conversations (excluding the current one) by semantic similarity to a short query phrase. Useful when the directive concerns something said earlier that may not be captured in long-term memories.",
    schema: z.object({
      query: z
        .string()
        .describe("short search phrase to match past conversation messages against"),
    }),
  }
);

const tools = [fetchMemories, fetchDeletedMemories, fetchPastConversations];

const DeepThinkAnnotation = Annotation.Root({
  ...MessagesAnnotation.spec,
  directive: Annotation({
    reducer: (_current, next) => next,
    default: () => "",
  }),
  iterations: Annotation({
    reducer: (_current, next) => next,
    default: () => 0,
  }),
});

// The planner decides which memory tools to call for the directive. Once it
// stops calling tools, its text response is the consolidated answer. On the
// final allowed pass, function calling is disabled at the API level: Gemini
// otherwise pattern-matches the earlier rounds and emits tool calls even
// when no tools are bound.
async function planner(state) {
  const toolsAvailable = state.iterations < MAX_TOOL_ROUNDS;
  const plannerModel = toolsAvailable
    ? model.bindTools(tools)
    : model.bindTools(tools, { tool_choice: "none" });
  const messages = [new SystemMessage(PLANNER_PROMPT), ...state.messages];
  if (!toolsAvailable) {
    messages.push(
      new HumanMessage(
        "The search limit has been reached. Do not call any tools — write your consolidated summary now based on the results above."
      )
    );
  }
  const response = await plannerModel.invoke(messages);
  logger.info(
    {
      pass: state.iterations + 1,
      toolsAvailable,
      toolCalls: response.tool_calls?.length ?? 0,
    },
    "deep-think planner pass"
  );
  return { messages: [response], iterations: state.iterations + 1 };
}

function routeAfterPlanner(state) {
  // Hard cap on tool rounds, independent of what the model emits.
  if (state.iterations > MAX_TOOL_ROUNDS) return END;
  const lastMessage = state.messages[state.messages.length - 1];
  return lastMessage?.tool_calls?.length ? "executor" : END;
}

// The consolidated findings are the planner's final AI message, so the
// subgraph needs no separate output node.
export const deepThinkGraph = new StateGraph(DeepThinkAnnotation)
  .addNode("planner", planner)
  .addNode("executor", new ToolNode(tools))
  .addEdge(START, "planner")
  .addConditionalEdges("planner", routeAfterPlanner, {
    executor: "executor",
    [END]: END,
  })
  .addEdge("executor", "planner")
  .compile();
