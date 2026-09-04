import "dotenv/config";
import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import { HumanMessage, SystemMessage } from "@langchain/core/messages";
import { DEFAULT_SUBJECT, getMemoriesWithoutSubject, updateMemory } from "./db.js";
import { embed } from "./memory.js";
import { logger } from "./logger.js";

// One-time migration for memories saved before memories.subject existed —
// back when the memory extractor only ever wrote facts about "the user"
// (see prompt.js's MEMORY_EXTRACTOR_PROMPT, now broadened to cover any
// subject). Every one of those old rows is known to be about DEFAULT_SUBJECT
// ("bree" unless PRIMARY_SUBJECT overrides it); this rewords their content
// to drop the now-redundant "User"/"the user" reference and tags them with
// that subject.
//
// MIGRATION_CUTOFF_ID is the highest memories.id that existed at the time
// this script was written (checked directly against the database — see the
// PR/commit introducing it). Bounding the query to ids at or below it means
// a rerun after the broadened extractor has already saved new, genuinely
// general (subject-less) memories can never mistake one of those for a
// pre-migration row — getMemoriesWithoutSubject only ever looks at ids
// under the cutoff. Idempotent beyond that: once a row's subject is set,
// it no longer matches the WHERE subject IS NULL clause.
//
// Run once, right after deploying the subject-tagging change and before
// relying on new general memories being distinguishable from these old
// ones:
//   npm run backfill-memory-subjects
const MIGRATION_CUTOFF_ID = 132;
const BATCH_SIZE = 20;

const REWORD_PROMPT = `You are migrating Corvus's long-term memory format. Memories used to be
written as statements about "the user" (for example: "User is 29 years
old."). Memories are now tagged with a separate "subject" field instead, so
the content itself should no longer name or refer to the subject by name.

Rewrite the given memory's content:
- Remove any "User"/"the user"/"User's" reference that is the sentence's
  leading grammatical subject, and start directly with the predicate.
- If a reference to the user is still needed elsewhere in the sentence for
  it to read naturally (for example, as the object of an instruction),
  replace it with a neutral pronoun ("they"/"them"/"their") instead of
  dropping it.
- Change nothing else: keep every other name, detail, and fact exactly as
  written, and keep imperative instructions imperative.

Examples:
"User is 29 years old." -> "Is 29 years old."
"User's favorite snack is chocolate." -> "Favorite snack is chocolate."
"Address the user as Bree." -> "Address as Bree."
"Push back plainly when confident the user is wrong, and do not be sycophantic." -> "Push back plainly when confident they are wrong, and do not be sycophantic."
"User has a dog named Quentin, currently being cared for in Seattle by her ex-boyfriend Brandon." -> "Has a dog named Quentin, currently being cared for in Seattle by her ex-boyfriend Brandon."

Output only the reworded sentence — no preamble, no quotation marks.`;

const rewriter = new ChatGoogleGenerativeAI({
  model: process.env.GEMINI_MODEL,
  apiKey: process.env.GEMINI_API_KEY,
  // Same reasoning-tokens-eat-the-budget trap as memory.js's extractor
  // (see its comment): a low maxOutputTokens truncated several rows
  // mid-sentence on this migration's first run (e.g. "Has lived in the
  // Netherlands for a year and is moving" — "back to Seattle on October
  // 17" got cut). Passing thinkingConfig: { thinkingBudget: 0 } to turn
  // thinking off outright made the API reject the request instead
  // (400 Bad Request), so this just gives reasoning a generous budget.
  maxOutputTokens: 4096,
});

function responseText(response) {
  const content = response?.content;
  if (typeof content === "string") return content;
  if (Array.isArray(content)) {
    return content.map((part) => (part?.type === "text" ? part.text : "")).join("");
  }
  return "";
}

async function reword(content) {
  const response = await rewriter.invoke([
    new SystemMessage(REWORD_PROMPT),
    new HumanMessage(content),
  ]);
  return responseText(response).trim().replace(/^"|"$/g, "");
}

async function main() {
  let total = 0;
  while (true) {
    const rows = await getMemoriesWithoutSubject({ maxId: MIGRATION_CUTOFF_ID, limit: BATCH_SIZE });
    if (!rows.length) break;
    for (const row of rows) {
      const revised = await reword(row.content);
      if (!revised) {
        logger.warn({ id: row.id }, "reword produced empty content; skipping row");
        continue;
      }
      await updateMemory(row.id, revised, await embed(revised), DEFAULT_SUBJECT);
      logger.info({ id: row.id, before: row.content, after: revised }, "memory subject backfilled");
    }
    total += rows.length;
  }
  logger.info({ total, subject: DEFAULT_SUBJECT }, "memory subject backfill complete");
}

main()
  .then(() => process.exit(0))
  .catch((err) => {
    logger.fatal({ err }, "memory subject backfill failed");
    process.exit(1);
  });
