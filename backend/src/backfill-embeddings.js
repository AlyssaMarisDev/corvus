import "dotenv/config";
import { getMessagesWithoutEmbedding, saveMessageEmbedding } from "./db.js";
import { embed } from "./memory.js";
import { logger } from "./logger.js";

const BATCH_SIZE = 100;

// Fills messages.embedding for rows saved before embeddings existed (or whose
// background embed after saving failed). Idempotent: safe to re-run until no
// rows remain. Requests are sequential to stay within the embedding API's
// rate limit.
async function main() {
  let total = 0;
  while (true) {
    const rows = await getMessagesWithoutEmbedding({ limit: BATCH_SIZE });
    if (!rows.length) break;
    for (const row of rows) {
      await saveMessageEmbedding(row.id, await embed(row.content));
    }
    total += rows.length;
    logger.info({ batch: rows.length, total }, "backfill batch completed");
  }
  logger.info({ total }, "backfill complete");
}

main()
  .then(() => process.exit(0))
  .catch((err) => {
    logger.fatal({ err }, "backfill failed");
    process.exit(1);
  });
