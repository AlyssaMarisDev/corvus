import { logger } from "./logger.js";

// Server-push channel for events that are not tied to a chat request —
// currently the proactive pipeline's messages. Clients hold a long-lived
// SSE connection on GET /events; every event is broadcast to all of them
// (Corvus is single-user, so no per-client routing is needed).
const clients = new Set();

export function registerClient(res) {
  clients.add(res);
  res.on("close", () => clients.delete(res));
  logger.debug({ clientCount: clients.size }, "events client connected");
}

// Payloads are JSON-encoded so text can safely contain newlines.
export function broadcast(event, data) {
  if (!clients.size) return;
  const payload = `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`;
  for (const res of clients) {
    try {
      res.write(payload);
    } catch (err) {
      logger.debug({ err }, "events client write failed; dropping client");
      clients.delete(res);
    }
  }
}

// Comment-line heartbeats keep mobile networks and proxies from reaping an
// idle SSE connection between proactive messages.
setInterval(() => {
  for (const res of clients) {
    try {
      res.write(": ping\n\n");
    } catch {
      clients.delete(res);
    }
  }
}, 25_000).unref();
