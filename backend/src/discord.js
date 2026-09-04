// Discord channel: lets the butler receive chat messages from, and send
// replies/status lines/proactive messages to, exactly one Discord user via
// DM. Opt-in like tracing.js — disabled cleanly whenever the Discord env
// vars aren't set, so the app runs fine without Discord configured.
//
// Unlike a webhook-based channel, this needs no public URL at all: the bot
// opens its own outbound WebSocket connection to Discord's gateway (via
// client.login) and receives events over that, the same way it would from
// behind a firewall or on a laptop with no port forwarding.
//
// Security invariant: nothing in this module (or its callers) ever takes a
// recipient as a parameter. sendDiscordMessage always DMs the single
// DISCORD_ALLOWED_USER_ID read from the environment, and inbound messages
// are only ever turned into a chat turn if they're a DM from that exact
// user — there is no code path, tool, or LLM output that can change who
// the butler talks to.
import { Client, GatewayIntentBits, Partials, ChannelType } from "discord.js";
import { runChatTurn } from "./chatTurn.js";
import { DEFAULT_SUBJECT, ensureSubject, getSubjectNameByDiscordId } from "./db.js";
import { logger } from "./logger.js";

const DISCORD_BOT_TOKEN = process.env.DISCORD_BOT_TOKEN;
const DISCORD_ALLOWED_USER_ID = process.env.DISCORD_ALLOWED_USER_ID;

export const discordEnabled = Boolean(DISCORD_BOT_TOKEN && DISCORD_ALLOWED_USER_ID);

let client = null;

// Discord caps a single message at 2000 characters; longer text is split
// on the nearest preceding newline (falling back to a hard cut) so replies
// don't get truncated or rejected outright.
const MAX_MESSAGE_LENGTH = 2000;

function chunkMessage(text) {
  const chunks = [];
  let rest = text;
  while (rest.length > MAX_MESSAGE_LENGTH) {
    let cut = rest.lastIndexOf("\n", MAX_MESSAGE_LENGTH);
    if (cut <= 0) cut = MAX_MESSAGE_LENGTH;
    chunks.push(rest.slice(0, cut));
    rest = rest.slice(cut).replace(/^\n+/, "");
  }
  if (rest) chunks.push(rest);
  return chunks;
}

// Sends text to the one allowed Discord user via DM, in order, chunked if
// needed. Never throws — a Discord outage must not break chat or the
// brain's proactive pipeline; failures are logged instead.
export async function sendDiscordMessage(text) {
  if (!discordEnabled || !client?.isReady()) return;
  if (!text || !text.trim()) return;
  try {
    const user = await client.users.fetch(DISCORD_ALLOWED_USER_ID);
    for (const chunk of chunkMessage(text)) {
      await user.send(chunk);
    }
  } catch (err) {
    logger.error({ err }, "discord send failed");
  }
}

// Connects to Discord's gateway and wires up the DM -> chat turn pipeline.
// Every inbound DM from DISCORD_ALLOWED_USER_ID becomes a turn exactly like
// /chat (same conversation, same runChatTurn); replies and status lines go
// back out over Discord via sendDiscordMessage. Anything else — a message
// from a different user, a message posted in a server channel rather than
// a DM, or the bot's own messages — is ignored outright.
export function startDiscordBot() {
  if (!discordEnabled) {
    logger.warn(
      "DISCORD_BOT_TOKEN/DISCORD_ALLOWED_USER_ID not fully set; discord channel disabled"
    );
    return;
  }

  // Seeds/re-affirms the identity mapping used to attribute inbound DMs to
  // a memory subject (see db.js's subjects table and getSubjectNameByDiscordId
  // below) — the one allowed sender maps to DEFAULT_SUBJECT by default.
  // Never blocks startup: a failure here just means messages fall back to
  // DEFAULT_SUBJECT until the next successful boot.
  void ensureSubject(DEFAULT_SUBJECT, DISCORD_ALLOWED_USER_ID).catch((err) =>
    logger.error({ err }, "failed to seed discord subject mapping")
  );

  client = new Client({
    // Guilds is required just to receive a usable client-ready state;
    // DirectMessages + MessageContent are what actually let the bot see
    // your DMs and read their text (MessageContent is a privileged intent
    // that must also be turned on for the bot in the Discord Developer
    // Portal — see .env.example).
    intents: [
      GatewayIntentBits.Guilds,
      GatewayIntentBits.DirectMessages,
      GatewayIntentBits.MessageContent,
    ],
    // DM channels/messages aren't always cached; partials let the client
    // still fire events for them instead of silently dropping uncached DMs.
    partials: [Partials.Channel, Partials.Message],
  });

  client.once("ready", () => {
    logger.info({ user: client.user.tag }, "discord bot connected");
  });

  client.on("messageCreate", (message) => {
    if (message.author.bot) return;
    if (message.channel.type !== ChannelType.DM) return;
    if (message.author.id !== DISCORD_ALLOWED_USER_ID) {
      logger.warn({ from: message.author.id }, "discord message from non-allowed user ignored");
      return;
    }
    const body = message.content;
    if (!body || !body.trim()) return;

    logger.info({ messageLength: body.length }, "discord message received");
    // Resolve who's actually messaging (a memory subject, e.g. "bree") via
    // the subjects table, falling back to DEFAULT_SUBJECT if the mapping
    // hasn't been seeded yet — never blocks the reply on this lookup
    // failing outright.
    getSubjectNameByDiscordId(message.author.id)
      .catch((err) => {
        logger.error({ err }, "discord subject lookup failed; using default");
        return null;
      })
      .then((subject) => {
        void runChatTurn(body, {
          subject: subject ?? DEFAULT_SUBJECT,
          onStatus: (text) => void sendDiscordMessage(text),
        })
          .then(({ reply, aborted }) => {
            if (!aborted && reply) void sendDiscordMessage(reply);
          })
          .catch((err) => {
            logger.error({ err }, "discord chat turn failed");
            void sendDiscordMessage("Sorry, something went wrong on my end.");
          });
      });
  });

  client.login(DISCORD_BOT_TOKEN).catch((err) => {
    logger.error({ err }, "discord login failed");
  });
}
