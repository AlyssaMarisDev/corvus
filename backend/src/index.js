import "dotenv/config";
import express from "express";
import cors from "cors";
import { AIMessage, HumanMessage } from "@langchain/core/messages";
import {
  conversationExists,
  createConversation,
  initDb,
  loadHistory,
  saveMessage,
} from "./db.js";
import { runCorvus } from "./agent.js";

const app = express();
app.use(cors());
app.use(express.json());

const PORT = process.env.PORT ?? 4000;

app.get("/health", (_req, res) => {
  res.json({ status: "ok", assistant: "Corvus" });
});

app.post("/chat", async (req, res) => {
  try {
    const { message } = req.body ?? {};
    let { conversationId } = req.body ?? {};

    if (!message || typeof message !== "string") {
      return res.status(400).json({ error: "message is required" });
    }

    if (conversationId) {
      if (!(await conversationExists(conversationId))) {
        return res.status(404).json({ error: "conversation not found" });
      }
    } else {
      conversationId = await createConversation();
    }

    const history = await loadHistory(conversationId);
    const messages = history.map((m) =>
      m.role === "user" ? new HumanMessage(m.content) : new AIMessage(m.content)
    );
    messages.push(new HumanMessage(message));

    const reply = await runCorvus(messages);

    await saveMessage(conversationId, "user", message);
    await saveMessage(conversationId, "assistant", reply);

    res.json({ reply, conversationId });
  } catch (err) {
    console.error("chat error", err);
    res.status(500).json({ error: "failed to get a response from Corvus" });
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
    console.error("history error", err);
    res.status(500).json({ error: "failed to load history" });
  }
});

initDb()
  .then(() => {
    app.listen(PORT, () => {
      console.log(`Corvus backend listening on http://localhost:${PORT}`);
    });
  })
  .catch((err) => {
    console.error("Failed to initialize database", err);
    process.exit(1);
  });
