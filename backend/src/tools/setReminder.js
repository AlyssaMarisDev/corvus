// The set_reminder tool: schedules a reminder that resurfaces into Corvus's
// own working memory at a future date/time, tagged as very important.
// Wired into the live DeepSeek tool-calling path alongside web_search and
// search_memory (see tools/index.js and agent.js). See db.js's reminders
// table and brain.js's checkReminders for the other half.
import { z } from "zod";
import { ToolMessage } from "@langchain/core/messages";
import { saveReminder } from "../db.js";
import { formatFullTimestamp } from "../prompt.js";
import { logger } from "../logger.js";

export const setReminderTool = {
  name: "set_reminder",
  description:
    "Schedule a reminder that resurfaces into your own working memory at a future date/time, tagged as very important. Use for anything worth revisiting later: a promise to follow up, a deadline, a recurring check-in (e.g. \"check email every morning\"), etc. This does not message the user by itself — it only resurfaces into your own working memory when due.",
  schema: z.object({
    content: z
      .string()
      .describe("the reminder text, written as a self-contained note to your future self"),
    due_at: z
      .string()
      .describe(
        "ISO 8601 date-time, with timezone offset, for when the reminder should first fire, e.g. 2026-09-03T09:00:00+02:00"
      ),
    recurrence: z
      .enum(["none", "hourly", "daily", "weekly", "monthly"])
      .describe('repeat cadence after it first fires; "none" for a one-time reminder'),
  }),
};

// recurrence -> Postgres interval literal; null means one-time (see
// db.js#saveReminder/fireReminder).
const RECURRENCE_INTERVALS = {
  none: null,
  hourly: "1 hour",
  daily: "1 day",
  weekly: "7 days",
  monthly: "1 month",
};

// Executes a single set_reminder tool call, persisting it to Postgres
// (db.js), and answers with a confirmation so the next corvus call sees
// valid tool-call/tool-response history. Never throws a bad call out of the
// turn — a malformed call gets an error ToolMessage back so the model can
// acknowledge or retry in its reply.
export async function executeSetReminder(tc) {
  const parsed = setReminderTool.schema.safeParse(tc.args);
  if (!parsed.success) {
    logger.warn({ args: tc.args, error: parsed.error.message }, "invalid set_reminder call");
    return new ToolMessage({
      content: `Could not schedule that reminder: ${parsed.error.issues[0]?.message ?? "invalid arguments"}.`,
      tool_call_id: tc.id,
    });
  }
  const { content, due_at, recurrence } = parsed.data;
  const dueDate = new Date(due_at);
  if (Number.isNaN(dueDate.getTime())) {
    logger.warn({ due_at }, "invalid set_reminder due_at");
    return new ToolMessage({
      content: `Could not schedule that reminder: "${due_at}" is not a valid date-time.`,
      tool_call_id: tc.id,
    });
  }
  const recurrenceInterval = RECURRENCE_INTERVALS[recurrence];
  const id = await saveReminder(content, dueDate, recurrenceInterval);
  logger.info({ id, content, dueAt: dueDate, recurrence }, "reminder scheduled");
  return new ToolMessage({
    content: `Reminder scheduled for ${formatFullTimestamp(dueDate)}${
      recurrenceInterval ? `, repeating ${recurrence}` : ""
    }.`,
    tool_call_id: tc.id,
  });
}
