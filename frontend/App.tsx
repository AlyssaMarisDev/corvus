import { StatusBar } from "expo-status-bar";
import React, { useCallback, useEffect, useRef, useState } from "react";
import {
  FlatList,
  KeyboardAvoidingView,
  Platform,
  StyleSheet,
  Text,
  TextInput,
  TouchableOpacity,
  View,
} from "react-native";
import { SafeAreaProvider, SafeAreaView } from "react-native-safe-area-context";
import { fetchHistory, streamChat } from "./src/api";
import { subscribeToEvents } from "./src/events";

type Message = {
  id: string;
  role: "user" | "assistant";
  content: string;
  // Thinking-out-loud lines from deep recall; rendered as muted italics.
  status?: boolean;
};

const WELCOME: Message = {
  id: "welcome",
  role: "assistant",
  content: "Hi, I'm Corvus. How can I help you today?",
};

export default function App() {
  const [messages, setMessages] = useState<Message[]>([WELCOME]);
  const [input, setInput] = useState("");
  const [isThinking, setIsThinking] = useState(false);
  const listRef = useRef<FlatList<Message>>(null);

  // Load the conversation's history on launch, so proactive messages
  // sent while the app was closed show up in the timeline.
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const history = await fetchHistory();
        if (cancelled || !history.messages.length) return;
        setMessages(
          history.messages.map((m, i) => ({
            id: `history-${i}`,
            role: m.role,
            content: m.content,
          }))
        );
      } catch {
        // Backend unreachable — keep the welcome message; sending a chat
        // will surface the connection error if the user tries.
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  // Server-pushed proactive messages stream into their own bubble. The
  // backend never pushes while a chat reply is streaming, so this cannot
  // interleave with an in-flight answer.
  useEffect(() => {
    let proactiveId: string | null = null;
    const unsubscribe = subscribeToEvents({
      onStart: () => {
        proactiveId = `proactive-${Date.now()}`;
        setMessages((prev) => [
          ...prev,
          { id: proactiveId as string, role: "assistant", content: "" },
        ]);
      },
      onToken: (text) => {
        const id = proactiveId;
        if (!id) return;
        setMessages((prev) =>
          prev.map((m) => (m.id === id ? { ...m, content: m.content + text } : m))
        );
      },
      onDone: ({ message }) => {
        const bubbleId = proactiveId;
        proactiveId = null;
        // The done event carries the authoritative final text (streamed
        // tokens can include whitespace the backend trimmed).
        if (bubbleId) {
          setMessages((prev) =>
            prev.map((m) => (m.id === bubbleId ? { ...m, content: message } : m))
          );
        }
      },
    });
    return unsubscribe;
  }, []);

  const send = useCallback(async () => {
    const text = input.trim();
    if (!text || isThinking) return;

    setMessages((prev) => [
      ...prev,
      { id: `user-${Date.now()}`, role: "user", content: text },
    ]);
    setInput("");
    setIsThinking(true);

    const replyId = `corvus-${Date.now()}`;
    const statusId = `status-${Date.now()}`;
    let replyStarted = false;
    let statusStarted = false;

    try {
      await streamChat(text, {
        onStatus: (text) => {
          // Reasoning streams as fragments; fold them into one live status
          // bubble per turn. Keep the thinking indicator up since the reply
          // tokens are still coming.
          if (!statusStarted) {
            statusStarted = true;
            setMessages((prev) => [
              ...prev,
              { id: statusId, role: "assistant", content: text, status: true },
            ]);
          } else {
            setMessages((prev) =>
              prev.map((m) =>
                m.id === statusId ? { ...m, content: m.content + text } : m
              )
            );
          }
        },
        onToken: (token) => {
          if (!replyStarted) {
            replyStarted = true;
            setIsThinking(false);
            setMessages((prev) => [
              ...prev,
              { id: replyId, role: "assistant", content: token },
            ]);
          } else {
            setMessages((prev) =>
              prev.map((m) =>
                m.id === replyId ? { ...m, content: m.content + token } : m
              )
            );
          }
        },
      });
      if (!replyStarted) {
        throw new Error("stream ended without a reply");
      }
    } catch {
      setMessages((prev) => [
        ...prev,
        {
          id: `error-${Date.now()}`,
          role: "assistant",
          content: "I couldn't reach the Corvus backend. Is it running?",
        },
      ]);
    } finally {
      setIsThinking(false);
    }
  }, [input, isThinking]);

  const canSend = input.trim().length > 0 && !isThinking;

  return (
    <SafeAreaProvider>
      <SafeAreaView style={styles.safe} edges={["top", "bottom"]}>
        <StatusBar style="light" />
        <View style={styles.header}>
          <Text style={styles.title}>Corvus</Text>
          <Text style={styles.subtitle}>personal ai assistant</Text>
        </View>

        <FlatList
          ref={listRef}
          data={messages}
          keyExtractor={(item) => item.id}
          contentContainerStyle={styles.list}
          onContentSizeChange={() => listRef.current?.scrollToEnd({ animated: true })}
          renderItem={({ item }) => (
            <View
              style={[
                styles.bubble,
                item.role === "user" ? styles.userBubble : styles.assistantBubble,
              ]}
            >
              <Text style={item.status ? styles.thinkingText : styles.bubbleText}>
                {item.content}
              </Text>
            </View>
          )}
          ListFooterComponent={
            isThinking ? (
              <View style={[styles.bubble, styles.assistantBubble]}>
                <Text style={styles.thinkingText}>Corvus is thinking…</Text>
              </View>
            ) : null
          }
        />

        <KeyboardAvoidingView behavior={Platform.OS === "ios" ? "padding" : undefined}>
          <View style={styles.inputRow}>
            <TextInput
              style={styles.input}
              value={input}
              onChangeText={setInput}
              placeholder="Message Corvus…"
              placeholderTextColor="#6b7280"
              multiline
              editable={!isThinking}
              onSubmitEditing={send}
            />
            <TouchableOpacity
              style={[styles.sendButton, !canSend && styles.sendButtonDisabled]}
              onPress={send}
              disabled={!canSend}
            >
              <Text style={styles.sendButtonText}>Send</Text>
            </TouchableOpacity>
          </View>
        </KeyboardAvoidingView>
      </SafeAreaView>
    </SafeAreaProvider>
  );
}

const styles = StyleSheet.create({
  safe: {
    flex: 1,
    backgroundColor: "#0d0f14",
  },
  header: {
    paddingVertical: 14,
    alignItems: "center",
    borderBottomWidth: StyleSheet.hairlineWidth,
    borderBottomColor: "#262b36",
  },
  title: {
    color: "#f3f4f6",
    fontSize: 20,
    fontWeight: "700",
    letterSpacing: 0.5,
  },
  subtitle: {
    color: "#6b7280",
    fontSize: 12,
    marginTop: 2,
  },
  list: {
    padding: 16,
    gap: 10,
  },
  bubble: {
    maxWidth: "82%",
    borderRadius: 16,
    paddingHorizontal: 14,
    paddingVertical: 10,
  },
  userBubble: {
    alignSelf: "flex-end",
    backgroundColor: "#5b5bd6",
    borderBottomRightRadius: 4,
  },
  assistantBubble: {
    alignSelf: "flex-start",
    backgroundColor: "#1e222b",
    borderBottomLeftRadius: 4,
  },
  bubbleText: {
    color: "#f3f4f6",
    fontSize: 16,
    lineHeight: 22,
  },
  thinkingText: {
    color: "#9ca3af",
    fontSize: 14,
    fontStyle: "italic",
  },
  inputRow: {
    flexDirection: "row",
    alignItems: "flex-end",
    padding: 12,
    gap: 8,
    borderTopWidth: StyleSheet.hairlineWidth,
    borderTopColor: "#262b36",
  },
  input: {
    flex: 1,
    minHeight: 44,
    maxHeight: 120,
    backgroundColor: "#1e222b",
    borderRadius: 22,
    paddingHorizontal: 16,
    paddingVertical: 10,
    color: "#f3f4f6",
    fontSize: 16,
  },
  sendButton: {
    height: 44,
    borderRadius: 22,
    paddingHorizontal: 18,
    alignItems: "center",
    justifyContent: "center",
    backgroundColor: "#5b5bd6",
  },
  sendButtonDisabled: {
    opacity: 0.4,
  },
  sendButtonText: {
    color: "#ffffff",
    fontSize: 16,
    fontWeight: "600",
  },
});
