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
import { retrieveDeletedMemories, retrieveMemories } from "./memory.js";
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
    .map((m) => `- ${m.content} (last updated: ${formatMemoryTimestamp(m.updated_at)})`)
    .join("\n");
}

function formatDeleted(memories) {
  if (!memories.length) return "No deleted memories found.";
  return memories
    .map((m) => `- ${m.content} (deleted: ${formatMemoryTimestamp(m.deleted_at)})`)
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
      "Search the user's deleted (outdated or contradicted) long-term memories by semantic similarity. Useful when the directive concerns something that may have changed.",
    schema: z.object({
      query: z.string().describe("short search phrase to match deleted memories against"),
    }),
  }
);

const tools = [fetchMemories, fetchDeletedMemories];

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
