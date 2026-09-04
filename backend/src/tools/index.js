// Registry of every tool wired into the live DeepSeek tool-calling path:
// agent.js binds all of them (via toFunctionSpec) on every corvus pass, up
// to MAX_TOOL_ROUNDS, and dispatches a model's tool call to the matching
// entry's execute() here. Each tool lives in its own self-contained file —
// definition (name/description/zod schema) plus its executor — pairing
// them here keeps agent.js from needing to know each tool's name in more
// than one place.
import { setReminderTool, executeSetReminder } from "./setReminder.js";
import { webSearchTool, executeWebSearch } from "./webSearch.js";
import { searchMemoryTool, executeSearchMemory } from "./searchMemory.js";
import { saveMemoryTool, executeSaveMemory } from "./saveMemory.js";

export const TOOLS = [
  { definition: setReminderTool, execute: executeSetReminder },
  { definition: webSearchTool, execute: executeWebSearch },
  { definition: searchMemoryTool, execute: executeSearchMemory },
  { definition: saveMemoryTool, execute: executeSaveMemory },
];
