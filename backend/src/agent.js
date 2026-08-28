import { END, MessagesAnnotation, START, StateGraph } from "@langchain/langgraph";
import { ChatOllama } from "@langchain/ollama";
import { SystemMessage } from "@langchain/core/messages";
import { CORVUS_PROMPT } from "./prompt.js";

const model = new ChatOllama({
  model: process.env.OLLAMA_MODEL ?? "llama3.1",
  baseUrl: process.env.OLLAMA_HOST ?? "http://localhost:11434",
});

async function callModel(state) {
  const response = await model.invoke([
    new SystemMessage(CORVUS_PROMPT),
    ...state.messages,
  ]);
  return { messages: [response] };
}

// Single node => exactly one LLM call per /chat request.
export const graph = new StateGraph(MessagesAnnotation)
  .addNode("corvus", callModel)
  .addEdge(START, "corvus")
  .addEdge("corvus", END)
  .compile();

export async function runCorvus(messages) {
  const result = await graph.invoke({ messages });
  const reply = result.messages[result.messages.length - 1];
  return typeof reply.content === "string"
    ? reply.content
    : JSON.stringify(reply.content);
}
