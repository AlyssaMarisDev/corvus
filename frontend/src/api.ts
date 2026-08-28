// Backend base URL. Android emulator: use http://10.0.2.2:4000 instead of
// localhost. Physical device: use your machine's LAN IP.
const API_URL = process.env.EXPO_PUBLIC_API_URL ?? "http://localhost:4000";

export type ChatResponse = {
  reply: string;
  conversationId: string;
};

export async function sendChat(
  message: string,
  conversationId?: string
): Promise<ChatResponse> {
  const res = await fetch(`${API_URL}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message, conversationId }),
  });

  if (!res.ok) {
    throw new Error(`Chat request failed with status ${res.status}`);
  }

  return res.json();
}
