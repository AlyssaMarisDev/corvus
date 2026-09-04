import "dotenv/config";
// Must be imported before anything that might make an LLM call, so the
// OpenTelemetry SDK is registered first (see tracing.js).
import "./tracing.js";
import express from "express";
import cors from "cors";
import pinoHttp from "pino-http";
import { initDb, loadHistory } from "./db.js";
import { runChatTurn } from "./chatTurn.js";
import { startThoughtLoop } from "./brain.js";
import { registerClient } from "./events.js";
import { shutdownTracing } from "./tracing.js";
import { startDiscordBot } from "./discord.js";
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

app.post("/chat", async (req, res) => {
  try {
    const { message } = req.body ?? {};

    if (!message || typeof message !== "string") {
      req.log.warn("chat rejected: missing or invalid message");
      return res.status(400).json({ error: "message is required" });
    }

    // Log lengths rather than content so private conversations stay out of logs.
    req.log.info({ messageLength: message.length }, "chat message received");

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

    const { aborted } = await runChatTurn(message, {
      isAborted: () => !clientConnected,
      onStatus: (text) => sendEvent("status", { text }),
      onToken: (text) => sendEvent("token", { text }),
    });

    if (aborted) {
      req.log.info("chat stream abandoned by client");
      return res.end();
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
    // Discord connects outbound over its own gateway (WebSocket), so this
    // needs no Express route — unlike a webhook-based channel, there's no
    // public URL to expose at all.
    startDiscordBot();
  })
  .catch((err) => {
    logger.fatal({ err }, "failed to initialize database");
    process.exit(1);
  });

// Flush any buffered Langfuse spans before the process actually exits.
for (const signal of ["SIGTERM", "SIGINT"]) {
  process.on(signal, () => {
    void shutdownTracing().finally(() => process.exit(0));
  });
}
