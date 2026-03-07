import { TelegramClient } from "@mtcute/bun";
import { tl } from "@mtcute/tl";
import { writeFileSync, existsSync, mkdirSync } from "fs";
import { join, resolve } from "path";
import { homedir } from "os";

// --- Config ---
const CHANNEL_USERNAME = "sutro_group";
const TOPIC_KEYWORD = "sparse parity";
const OUTPUT_DIR = resolve(import.meta.dir, "src/sparse_parity/telegram_sync");
const OUTPUT_FILE = join(OUTPUT_DIR, "messages.json");

// Reuse the tg CLI session directly
const SESSION_PATH = join(homedir(), ".telegram-sync-cli", "session_1.db");

const API_ID = parseInt(process.env.TELEGRAM_API_ID ?? "0", 10);
const API_HASH = process.env.TELEGRAM_API_HASH ?? "";

if (!API_ID || !API_HASH) {
  console.error("Set TELEGRAM_API_ID and TELEGRAM_API_HASH in .env");
  process.exit(1);
}

if (!existsSync(SESSION_PATH)) {
  console.error(`Session not found at ${SESSION_PATH}. Run 'tg auth login' first.`);
  process.exit(1);
}

const client = new TelegramClient({
  apiId: API_ID,
  apiHash: API_HASH,
  storage: SESSION_PATH,
});

async function main() {
  console.log("Connecting to Telegram...");
  await client.start();
  console.log("Connected.");

  // Resolve the channel
  const resolved = await client.resolvePeer(CHANNEL_USERNAME);
  console.log(`Resolved ${CHANNEL_USERNAME} -> ${JSON.stringify(resolved)}`);

  // Get forum topics
  const inputChannel: tl.TypeInputChannel = {
    _: "inputChannel",
    channelId: (resolved as tl.RawInputPeerChannel).channelId,
    accessHash: (resolved as tl.RawInputPeerChannel).accessHash,
  };

  console.log("Fetching forum topics...");
  const topics = await client.call({
    _: "channels.getForumTopics",
    channel: inputChannel,
    limit: 100,
    offsetDate: 0,
    offsetId: 0,
    offsetTopic: 0,
  });

  // Find the sparse parity topic
  const forumTopics = (topics as tl.RawMessagesForumTopics).topics;
  const target = forumTopics.find((t) => {
    if (t._ === "forumTopic") {
      return t.title.toLowerCase().includes(TOPIC_KEYWORD.toLowerCase());
    }
    return false;
  }) as tl.RawForumTopic | undefined;

  if (!target) {
    console.error(`Topic matching "${TOPIC_KEYWORD}" not found.`);
    console.log(
      "Available topics:",
      forumTopics
        .filter((t): t is tl.RawForumTopic => t._ === "forumTopic")
        .map((t) => `  [${t.id}] ${t.title}`)
        .join("\n")
    );
    await client.close();
    process.exit(1);
  }

  console.log(`Found topic: [${target.id}] "${target.title}"`);

  // Fetch all messages in this topic thread
  const allMessages: tl.RawMessage[] = [];
  let offsetId = 0;
  const BATCH_SIZE = 100;

  console.log("Fetching messages...");
  while (true) {
    const history = await client.call({
      _: "messages.getReplies",
      peer: resolved,
      msgId: target.id,
      offsetId,
      offsetDate: 0,
      addOffset: 0,
      limit: BATCH_SIZE,
      maxId: 0,
      minId: 0,
      hash: 0,
    });

    const msgs = (history as tl.RawMessagesChannelMessages).messages.filter(
      (m): m is tl.RawMessage => m._ === "message"
    );

    if (msgs.length === 0) break;

    allMessages.push(...msgs);
    console.log(`  fetched ${allMessages.length} messages so far...`);

    offsetId = msgs[msgs.length - 1].id;

    // Rate limit courtesy
    await new Promise((r) => setTimeout(r, 500));
  }

  // Extract useful fields
  const users = new Map<number, string>();
  if ("users" in topics) {
    for (const u of (topics as tl.RawMessagesForumTopics).users) {
      if (u._ === "user") {
        users.set(
          u.id,
          [u.firstName, u.lastName].filter(Boolean).join(" ")
        );
      }
    }
  }

  const output = allMessages.map((m) => ({
    id: m.id,
    date: new Date(m.date * 1000).toISOString(),
    sender: users.get(Number(m.fromId && "userId" in m.fromId ? m.fromId.userId : 0)) ?? String(m.fromId),
    text: m.message,
    replyTo: m.replyTo && "replyToMsgId" in m.replyTo ? m.replyTo.replyToMsgId : null,
  }));

  // Also fetch user info from message history responses
  // (the topics response may not have all users)

  // Write output
  if (!existsSync(OUTPUT_DIR)) mkdirSync(OUTPUT_DIR, { recursive: true });
  writeFileSync(OUTPUT_FILE, JSON.stringify(output, null, 2));
  console.log(`\nDone. ${output.length} messages written to ${OUTPUT_FILE}`);

  process.exit(0);
}

main().catch((e) => {
  console.error("Error:", e);
  process.exit(1);
});
