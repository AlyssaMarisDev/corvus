import { AIMessage, HumanMessage } from "@langchain/core/messages";
import { DEFAULT_SUBJECT, loadRecentMessages, saveMessage, saveMessageEmbedding } from "./db.js";
import { streamCorvus } from "./agent.js";
import { embed } from "./memory.js";
import { notifyChatRequest, notifyChatStreamStart, notifyChatStreamEnd } from "./brain.js";
import { endError, endOk, startTrace } from "./tracing.js";
import { logger } from "./logger.js";

// The model's context is capped at the last 10 exchanges (a user message
// plus the assistant's reply); older messages live on in long-term memory
// and past-message search instead.
const CONTEXT_TURNS = 10;

// Runs one full Corvus turn for `message` — shared by every inbound
// channel (the /chat SSE route and the WhatsApp webhook, see index.js).
// This is exactly what the original /chat handler did inline: load recent
// history, stream the graph, persist the exchange, embed it in the
// background. It's parameterized over how status/token events reach the
// caller so each channel can render them however makes sense for it
// (SSE events for the frontend, individual WhatsApp messages for Twilio).
//
// onStatus(text) / onToken(text) are invoked as streamCorvus yields events.
// isAborted(), if provided, is polled between events so a caller tied to a
// live connection (the SSE route, on client disconnect) can bail out early
// without persisting a partial reply; channels with no such notion (the
// WhatsApp webhook, which already ack'd Twilio) simply omit it. `subject`
// identifies who's actually talking (a memory subject, e.g. "bree") — the
// web UI has no identity of its own and always defaults to it; Discord
// resolves its own sender via the subjects table (see discord.js) and
// passes it in explicitly. Returns { reply, aborted, trace }.
export async function runChatTurn(
  message,
  { subject = DEFAULT_SUBJECT, onStatus, onToken, isAborted } = {}
) {
  let trace;
  try {
    // A validated chat message wakes the brain into active mode, whichever
    // channel it arrived on.
    notifyChatRequest();

    const history = await loadRecentMessages({ limit: CONTEXT_TURNS * 2 });
    // Defensive: merge any consecutive same-role rows so the model always
    // sees strictly alternating roles, regardless of how they got saved.
    const mergedHistory = [];
    for (const m of history) {
      const last = mergedHistory[mergedHistory.length - 1];
      if (last && last.role === m.role) {
        last.content += `\n${m.content}`;
      } else {
        mergedHistory.push({ ...m });
      }
    }
    const messages = mergedHistory.map((m) =>
      m.role === "user" ? new HumanMessage(m.content) : new AIMessage(m.content)
    );
    messages.push(new HumanMessage(message));

    // One Langfuse trace per turn: every LLM call the graph makes
    // (retrieve's memory embed, each corvus pass, tool calls, extract's
    // memory extraction + interaction synthesis) nests under this as a
    // child observation (see agent.js).
    trace = startTrace("chat", { input: { message } });

    // While a reply is streaming, the brain's proactive pipeline stays
    // quiet so an unprompted message never interrupts a live answer.
    notifyChatStreamStart();
    try {
      const start = performance.now();
      let reply = "";
      for await (const event of streamCorvus(messages, trace, subject)) {
        if (isAborted?.()) {
          endOk(trace, { output: { reply, aborted: true } });
          return { reply, aborted: true, trace };
        }
        if (event.type === "status") {
          // Discrete tool-activity announcements only (e.g. "Searching the
          // web…") — forwarded live but never persisted, so they never leak
          // into /history or get fed back to the model as its own reply.
          onStatus?.(event.text);
        } else {
          reply += event.text;
          onToken?.(event.text);
        }
      }

      logger.info(
        {
          historyLength: history.length,
          replyLength: reply.length,
          durationMs: Math.round(performance.now() - start),
        },
        "chat reply generated"
      );
      // The graph's background extract node (memory extraction + interaction
      // synthesis, see agent.js) still nests its own spans onto this trace
      // after this point — Langfuse correlates by trace id, not by whether
      // the root observation has already ended.
      endOk(trace, { output: { reply } });

      const userMessageId = await saveMessage("user", message);
      const replyId = await saveMessage("assistant", reply);

      // Embed in the background so the caller is not delayed; a failure
      // leaves embedding NULL, which the backfill script repairs on its
      // next run.
      for (const [id, text] of [
        [userMessageId, message],
        [replyId, reply],
      ]) {
        void embed(text, trace)
          .then((e) => saveMessageEmbedding(id, e))
          .catch((err) => logger.error({ err }, "message embedding failed"));
      }

      return { reply, aborted: false, trace };
    } finally {
      notifyChatStreamEnd();
    }
  } catch (err) {
    endError(trace, err);
    throw err;
  }
}
