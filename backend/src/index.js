import "dotenv/config";
import express from "express";
import cors from "cors";
import pinoHttp from "pino-http";
import { AIMessage, HumanMessage } from "@langchain/core/messages";
import {
  initDb,
  loadHistory,
  loadRecentMessages,
  saveMessage,
  saveMessageEmbedding,
} from "./db.js";
import { streamCorvus } from "./agent.js";
import { embed } from "./memory.js";
import {
  notifyChatRequest,
  notifyChatStreamStart,
  notifyChatStreamEnd,
  startThoughtLoop,
} from "./brain.js";
import { registerClient } from "./events.js";
import { logger } from "./logger.js";

const app = express();
app.use(cors());
app.use(express.json());
// Trim pino-http's default req/res logging (which includes full headers,
// remote address, etc.) down to just the fields worth seeing per request.
app.use(
  pinoHttp({
    logger,
    serializers: {
      req: (req) => ({ method: req.method, url: req.url }),
      res: (res) => ({ statusCode: res.statusCode }),
    },
  })
);

const PORT = process.env.PORT;

app.get("/health", (_req, res) => {
  res.json({ status: "ok", assistant: "Corvus" });
});

// The model's context is capped at the last 10 exchanges (a user message
// plus the assistant's reply); older messages live on in long-term memory
// and past-message search instead.
const CONTEXT_TURNS = 10;

app.post("/chat", async (req, res) => {
  try {
    const { message } = req.body ?? {};

    if (!message || typeof message !== "string") {
      req.log.warn("chat rejected: missing or invalid message");
      return res.status(400).json({ error: "message is required" });
    }

    // Log lengths rather than content so private conversations stay out of logs.
    req.log.info({ messageLength: message.length }, "chat message received");
    // A validated chat message wakes the brain into active mode.
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

    res.writeHead(200, {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache",
      Connection: "keep-alive",
    });

    // While a reply is streaming, the brain's proactive pipeline stays
    // quiet so an unprompted message never interrupts a live answer.
    // res "close" fires on normal completion and client disconnect alike.
    notifyChatStreamStart();
    res.on("close", notifyChatStreamEnd);

    let clientConnected = true;
    req.on("close", () => {
      clientConnected = false;
    });

    // Payloads are JSON-encoded so token text can safely contain newlines.
    const sendEvent = (event, data) => {
      res.write(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`);
    };

    const start = performance.now();
    let reply = "";
    for await (const event of streamCorvus(messages)) {
      if (!clientConnected) {
        req.log.info("chat stream abandoned by client");
        return res.end();
      }
      if (event.type === "status") {
        // Discrete tool-activity announcements only (e.g. "Searching the
        // web…") — forwarded live but never persisted, so they never leak
        // into /history or get fed back to the model as its own reply.
        sendEvent("status", { text: event.text });
      } else {
        reply += event.text;
        sendEvent("token", { text: event.text });
      }
    }

    req.log.info(
      {
        historyLength: history.length,
        replyLength: reply.length,
        durationMs: Math.round(performance.now() - start),
      },
      "chat reply generated"
    );

    const userMessageId = await saveMessage("user", message);
    const replyId = await saveMessage("assistant", reply);

    // Embed in the background so the done event is not delayed; a failure
    // leaves embedding NULL, which the backfill script repairs on its next run.
    for (const [id, text] of [
      [userMessageId, message],
      [replyId, reply],
    ]) {
      void embed(text)
        .then((e) => saveMessageEmbedding(id, e))
        .catch((err) => logger.error({ err }, "message embedding failed"));
    }

    sendEvent("done", {});
    res.end();
  } catch (err) {
    req.log.error({ err }, "chat error");
    if (res.headersSent) {
      // Headers already went out as SSE, so report failures as an event.
      res.write(
        `event: error\ndata: ${JSON.stringify({
          error: "failed to get a response from Corvus",
        })}\n\n`
      );
      res.end();
    } else {
      res.status(500).json({ error: "failed to get a response from Corvus" });
    }
  }
});

// The single conversation's full history — the frontend loads this on
// launch so messages sent while the app was closed (proactive ones) are
// not missed.
app.get("/history", async (req, res) => {
  try {
    const messages = await loadHistory();
    res.json({ messages });
  } catch (err) {
    req.log.error({ err }, "history error");
    res.status(500).json({ error: "failed to load history" });
  }
});

// Long-lived SSE connection for server-pushed events (currently the
// proactive pipeline's messages). Clients reconnect on drop; heartbeats
// come from the events module.
app.get("/events", (req, res) => {
  res.writeHead(200, {
    "Content-Type": "text/event-stream",
    "Cache-Control": "no-cache",
    Connection: "keep-alive",
  });
  res.write(": connected\n\n");
  registerClient(res);
});

initDb()
  .then(() => {
    app.listen(PORT, () => {
      logger.info({ port: PORT }, "Corvus backend listening");
    });
    startThoughtLoop();
  })
  .catch((err) => {
    logger.fatal({ err }, "failed to initialize database");
    process.exit(1);
  });
