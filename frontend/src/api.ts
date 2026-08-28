// Backend base URL. Android emulator: use http://10.0.2.2:4000 instead of
// localhost. Physical device: use your machine's LAN IP.
import { fetch } from "expo/fetch";

const API_URL = process.env.EXPO_PUBLIC_API_URL ?? "http://localhost:4200";

export type ChatStreamHandlers = {
  onConversation: (conversationId: string) => void;
  onToken: (text: string) => void;
  onStatus?: (text: string) => void;
};

// Streams a reply from POST /chat (SSE). expo/fetch is required here because
// React Native's built-in fetch does not support reading response bodies as
// streams. Throws if the request fails or the server reports a stream error.
export async function streamChat(
  message: string,
  conversationId: string | undefined,
  handlers: ChatStreamHandlers
): Promise<void> {
  const res = await fetch(`${API_URL}/chat`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Accept: "text/event-stream",
    },
    body: JSON.stringify({ message, conversationId }),
  });

  if (!res.ok || !res.body) {
    throw new Error(`Chat request failed with status ${res.status}`);
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  const dispatch = (rawEvent: string) => {
    let event = "message";
    const dataLines: string[] = [];
    for (const line of rawEvent.split("\n")) {
      if (line.startsWith("event:")) event = line.slice(6).trim();
      else if (line.startsWith("data:")) dataLines.push(line.slice(5).trim());
    }
    if (dataLines.length === 0) return;

    const data = JSON.parse(dataLines.join("\n"));
    if (event === "meta") {
      handlers.onConversation(data.conversationId);
    } else if (event === "token") {
      handlers.onToken(data.text);
    } else if (event === "status") {
      handlers.onStatus?.(data.text);
    } else if (event === "error") {
      throw new Error(data.error ?? "stream error");
    }
  };

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    let sep = buffer.indexOf("\n\n");
    while (sep !== -1) {
      dispatch(buffer.slice(0, sep));
      buffer = buffer.slice(sep + 2);
      sep = buffer.indexOf("\n\n");
    }
  }
}
