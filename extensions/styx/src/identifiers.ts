import { createHash } from "node:crypto";

/** Preserve ordinary host identifiers and hash oversized values without collisions. */
export function normalizeIdentifier(value: unknown, limit: number): string {
  const text = typeof value === "string" ? value.trim()
    : value == null ? ""
    : typeof value === "number" || typeof value === "boolean" ||
      typeof value === "bigint"
      ? String(value).trim()
      : `<${typeof value}>`;
  if (text.length <= limit) return text;
  const digest = createHash("sha256").update(text).digest("hex");
  const marker = `:sha256:${digest}`;
  if (marker.length >= limit) return digest.slice(0, limit);
  return text.slice(0, limit - marker.length) + marker;
}

export function runIdentity(value: unknown): string | null {
  const normalized = normalizeIdentifier(value, 252);
  return normalized ? `run:${normalized}` : null;
}

/** Prefer the host run coordinate, then its explicit physical turn id. */
export function runOrTurnIdentity(
  runValue: unknown,
  turnValue: unknown,
): string | null {
  const run = runIdentity(runValue);
  if (run) return run;
  const turn = normalizeIdentifier(turnValue, 251);
  return turn ? `turn:${turn}` : null;
}

/** Canonical durable key shared by preturn and terminal commit. */
export function openclawHostKey(identity: string): string {
  return `openclaw:${createHash("sha256").update(identity).digest("hex")}`;
}
