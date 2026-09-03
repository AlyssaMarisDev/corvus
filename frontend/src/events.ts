// Subscription to the backend's server-push channel (GET /events, SSE) for
// events not tied to a chat request — currently proactive messages from the
// thought loop. expo/fetch is required because React Native's built-in
// fetch does not support reading response bodies as streams.
import { fetch } from "expo/fetch";

const API_URL = process.env.EXPO_PUBLIC_API_URL ?? "http://localhost:4200";

export type ProactiveEventHandlers = {
  // A proactive message started streaming in; create its bubble.
  onStart: () => void;
  onToken: (text: string) => void;
  // Authoritative final text (streamed tokens may include whitespace the
  // backend trimmed).
  onDone: (data: { message: string }) => void;
};

// Keeps the event stream alive across drops with reconnect backoff
// (2s doubling to 30s). Returns an unsubscribe function.
export function subscribeToEvents(handlers: ProactiveEventHandlers): () => void {
  let stopped = false;
  let retryMs = 2_000;
  let retryTimer: ReturnType<typeof setTimeout> | null = null;
  let abort: AbortController | null = null;

  const dispatch = (rawEvent: string) => {
    let event = "message";
    const dataLines: string[] = [];
    for (const line of rawEvent.split("\n")) {
      if (line.startsWith("event:")) event = line.slice(6).trim();
      else if (line.startsWith("data:")) dataLines.push(line.slice(5).trim());
    }
    // Heartbeats and the connect greeting arrive as comment lines with no
    // data, so they fall through here.
    if (dataLines.length === 0) return;

    const data = JSON.parse(dataLines.join("\n"));
    if (event === "proactive_start") {
      handlers.onStart();
    } else if (event === "proactive_token") {
      handlers.onToken(data.text);
    } else if (event === "proactive_done") {
      handlers.onDone(data);
    }
  };

  const connect = async () => {
    abort = new AbortController();
    try {
      const res = await fetch(`${API_URL}/events`, {
        headers: { Accept: "text/event-stream" },
        signal: abort.signal,
      });
      if (!res.ok || !res.body) {
        throw new Error(`events request failed with status ${res.status}`);
      }
      // A healthy connection resets the backoff.
      retryMs = 2_000;
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        // SSE events are separated by a blank line; a chunk can carry
        // several events or half of one, hence the buffer.
        let sep = buffer.indexOf("\n\n");
        while (sep !== -1) {
          dispatch(buffer.slice(0, sep));
          buffer = buffer.slice(sep + 2);
          sep = buffer.indexOf("\n\n");
        }
      }
    } catch {
      // Unsubscribe aborts or connection failures land here; the stopped
      // flag decides whether a reconnect follows.
    }
    if (!stopped) {
      retryTimer = setTimeout(() => void connect(), retryMs);
      retryMs = Math.min(retryMs * 2, 30_000);
    }
  };

  void connect();

  return () => {
    stopped = true;
    if (retryTimer) clearTimeout(retryTimer);
    abort?.abort();
  };
}
