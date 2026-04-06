import { existsSync, readFileSync } from "fs";
import { join, dirname } from "path";
import { homedir } from "os";

/**
 * Load .env file if present (project root = 2 parent dirs above this source file).
 * Only sets env vars that aren't already defined (shell env takes precedence).
 */
export function loadEnv() {
  // resolve: src/telegram/env.ts → project root (2 directories up)
  const envPath = join(import.meta.dir, "..", "..", ".env");
  if (existsSync(envPath)) {
    try {
      for (const line of readFileSync(envPath, "utf-8").split("\n")) {
        const trimmed = line.trim();
        if (trimmed && !trimmed.startsWith("#") && trimmed.includes("=")) {
          const [key, ...rest] = trimmed.split("=");
          process.env[key.trim()] ??= rest.join("=").trim();
        }
      }
    } catch {}
  }
}

/** Path to the authenticated mtcute session file. */
export const SESSION_PATH = join(homedir(), ".telegram-sync-cli", "session_1.db");
