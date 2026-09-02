import "dotenv/config";
import express from "express";
import cors from "cors";
import pinoHttp from "pino-http";
import { AIMessage, HumanMessage } from "@langchain/core/messages";
import {
  conversationExists,
  createConversation,
  initDb,
  loadHistory,
  saveMessage,
  saveMessageEmbedding,
} from "./db.js";
import { streamCorvus } from "./agent.js";
import { embed } from "./memory.js";
import { notifyChatRequest, startThoughtLoop } from "./brain.js";
import { logger } from "./logger.js";

const app = express();
app.use(cors());
app.use(express.json());
app.use(pinoHttp({ logger }));

const PORT = process.env.PORT;

app.get("/health", (_req, res) => {
  res.json({ status: "ok", assistant: "Corvus" });
});

app.post("/chat", async (req, res) => {
  try {
    const { message } = req.body ?? {};
    let { conversationId } = req.body ?? {};

    if (!message || typeof message !== "string") {
      req.log.warn("chat rejected: missing or invalid message");
      return res.status(400).json({ error: "message is required" });
    }

    if (conversationId) {
      if (!(await conversationExists(conversationId))) {
        req.log.warn({ conversationId }, "chat rejected: conversation not found");
        return res.status(404).json({ error: "conversation not found" });
      }
    } else {
      conversationId = await createConversation();
      req.log.info({ conversationId }, "conversation created");
    }

    // Log lengths rather than content so private conversations stay out of logs.
    req.log.info(
      { conversationId, messageLength: message.length },
      "chat message received"
    );
    // A validated chat message wakes the brain into active mode.
    notifyChatRequest();

    const history = await loadHistory(conversationId);
    // Deep-think turns persist a status line before the reply, producing
    // consecutive assistant rows; merge same-role runs so Gemini sees
    // strictly alternating roles.
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

    let clientConnected = true;
    req.on("close", () => {
      clientConnected = false;
    });

    // Payloads are JSON-encoded so token text can safely contain newlines.
    const sendEvent = (event, data) => {
      res.write(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`);
    };

    sendEvent("meta", { conversationId });

    const start = performance.now();
    let reply = "";
    let statusText = "";
    for await (const event of streamCorvus(messages, conversationId)) {
      if (!clientConnected) {
        req.log.info({ conversationId }, "chat stream abandoned by client");
        return res.end();
      }
      if (event.type === "status") {
        statusText = event.text;
        sendEvent("status", { text: event.text });
      } else {
        reply += event.text;
        sendEvent("token", { text: event.text });
      }
    }

    req.log.info(
      {
        conversationId,
        historyLength: history.length,
        replyLength: reply.length,
        durationMs: Math.round(performance.now() - start),
      },
      "chat reply generated"
    );

    const userMessageId = await saveMessage(conversationId, "user", message);
    if (statusText) {
      // Saved for history but never embedded: transient thinking-out-loud is
      // noise for past-conversation search.
      await saveMessage(conversationId, "assistant", statusText);
    }
    const replyId = await saveMessage(conversationId, "assistant", reply);

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

    sendEvent("done", { conversationId });
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

app.get("/history/:conversationId", async (req, res) => {
  try {
    const { conversationId } = req.params;
    if (!(await conversationExists(conversationId))) {
      return res.status(404).json({ error: "conversation not found" });
    }
    const messages = await loadHistory(conversationId);
    res.json({ conversationId, messages });
  } catch (err) {
    req.log.error({ err }, "history error");
    res.status(500).json({ error: "failed to load history" });
  }
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
