// Typed `agent_end` hook — the only OpenClaw surface that captures a
// finalized dialogue turn and submits it for causal affect observation.
//
// ContextEngine.afterTurn is not a finality guarantee: a host may invoke it
// between model/tool-loop iterations.  `agent_end` is emitted after the loop
// settles and therefore prevents intermediate assistant/tool-call messages
// from becoming diary entries or affect evidence.

import { createHash } from "node:crypto";

import {
  parseAgentIdFromSessionKey,
} from "../agent-id-shared.js";
import {
  fmtErr,
  type StyxClient,
  type StyxLogger,
  type StyxMessage,
} from "../client.js";
import { interceptDocumentAttachments } from "../media-attachments.js";
import {
  TerminalTurnBarrier,
  terminalScopeKey,
} from "./terminal-barrier.js";

export type AgentEndHookEvent = {
  runId?: string;
  // Some harness adapters expose an explicit turn id in addition to the
  // current public OpenClaw contract.  It is an identity, never a content hash.
  turnId?: string;
  messages: unknown[];
  success: boolean;
  error?: string;
  durationMs?: number;
};

export type AgentEndHookContext = {
  runId?: string;
  agentId?: string;
  sessionKey?: string;
  sessionId?: string;
  modelProviderId?: string;
  modelId?: string;
  [key: string]: unknown;
};

export type AgentEndHookHandler = (
  event: AgentEndHookEvent,
  ctx: AgentEndHookContext,
) => Promise<void> | void;

export type AgentEndHookParams = {
  client: StyxClient;
  logger: StyxLogger;
  resolveAgentId: (openclawAgentId: string) => Promise<string>;
  terminalBarrier: TerminalTurnBarrier;
  terminalWorkBudgetMs?: number;
};

export const MAX_SOURCE_MESSAGES = 256;
export const MAX_HISTORY_MESSAGES = 24;
export const MAX_TOOL_EVENTS = 32;
export const MAX_CONTENT_PARTS = 128;
export const MAX_HISTORY_CONTENT_CHARS = 4_000;
export const MAX_TOOL_EVENT_CONTENT_CHARS = 4_000;
export const MAX_USER_MESSAGE_CHARS = 20_000;
export const MAX_ASSISTANT_RESPONSE_CHARS = 40_000;
export const MAX_ATTACHMENT_MARKERS = 32;
export const MAX_ATTACHMENT_INGESTS = 16;
export const MAX_SERIALIZED_DEPTH = 3;
export const MAX_SERIALIZED_ITEMS = 64;
export const TERMINAL_WORK_BUDGET_MS = 9_000;

const RECENT_IDENTITY_LIMIT = 2_048;

function asString(value: unknown): string {
  if (typeof value === "string") return value;
  if (value == null) return "";
  return String(value);
}

function boundedProjection(value: unknown, depth = 0): unknown {
  if (value == null || typeof value === "boolean" || typeof value === "number") {
    return value;
  }
  if (typeof value === "string") {
    return value.slice(0, MAX_TOOL_EVENT_CONTENT_CHARS);
  }
  if (depth >= MAX_SERIALIZED_DEPTH) return `<${typeof value}:bounded>`;
  if (Array.isArray(value)) {
    const out = value.slice(0, MAX_SERIALIZED_ITEMS)
      .map((item) => boundedProjection(item, depth + 1));
    if (value.length > MAX_SERIALIZED_ITEMS) out.push("...[bounded]");
    return out;
  }
  if (typeof value === "object") {
    const out: Record<string, unknown> = {};
    let count = 0;
    for (const [key, item] of Object.entries(value as Record<string, unknown>)) {
      if (count++ >= MAX_SERIALIZED_ITEMS) {
        out["...[bounded]"] = true;
        break;
      }
      out[key.slice(0, 128)] = boundedProjection(item, depth + 1);
    }
    return out;
  }
  return `<${typeof value}>`;
}

export function bounded(value: unknown, limit: number): string {
  let text: string;
  if (typeof value === "string") {
    text = value;
  } else if (value == null) {
    text = "";
  } else if (typeof value === "object") {
    try {
      text = JSON.stringify(boundedProjection(value));
    } catch {
      text = "";
    }
  } else {
    text = String(value);
  }
  if (text.length <= limit) return text;
  const marker = "\n...[bounded by styx]...\n";
  const available = Math.max(0, limit - marker.length);
  const head = Math.floor(available / 2);
  return text.slice(0, head) + marker + text.slice(-(available - head));
}

/** Extract model-visible text after bounding the number of content parts. */
export function extractBoundedMessageContent(
  content: unknown,
  limit: number,
): string {
  if (typeof content === "string") return bounded(content, limit);
  if (!Array.isArray(content)) return bounded(content, limit);

  const parts: string[] = [];
  let remaining = Math.max(0, limit);
  // Bound before map/join: a hostile multimodal message cannot make the hook
  // traverse an unbounded content-parts array.
  for (const rawPart of content.slice(-MAX_CONTENT_PARTS).reverse()) {
    if (remaining <= 0) break;
    if (typeof rawPart === "string") {
      const piece = rawPart.slice(-remaining);
      parts.push(piece);
      remaining -= piece.length + (parts.length > 1 ? 1 : 0);
      continue;
    }
    if (!rawPart || typeof rawPart !== "object") continue;
    const part = rawPart as Record<string, unknown>;
    if (part["type"] === "text" || part["type"] === "input_text" ||
        part["type"] === "output_text" || part["type"] === undefined) {
      const text = part["text"];
      if (typeof text === "string" && text) {
        const piece = text.slice(-remaining);
        parts.push(piece);
        remaining -= piece.length + (parts.length > 1 ? 1 : 0);
      }
    }
  }
  parts.reverse();
  return parts.join("\n").slice(0, limit);
}

function boundedSource(messages: unknown[]): Array<Record<string, unknown>> {
  // The slice happens before every role filter, map, join, and tool scan.
  return messages
    .slice(-MAX_SOURCE_MESSAGES)
    .filter((message): message is Record<string, unknown> =>
      Boolean(message) && typeof message === "object");
}

export function extractFinalTurnMessages(
  messages: unknown[],
): Array<Record<string, unknown>> {
  const source = boundedSource(messages);
  let lastUserIndex = -1;
  for (let index = source.length - 1; index >= 0; index--) {
    if (source[index]?.["role"] === "user") {
      lastUserIndex = index;
      break;
    }
  }
  if (lastUserIndex >= 0) return source.slice(lastUserIndex);

  for (let index = source.length - 1; index >= 0; index--) {
    if (source[index]?.["role"] === "assistant") return [source[index]!];
  }
  return [];
}

export function buildConversationHistory(
  messages: unknown[],
): Array<{ role: "system" | "user" | "assistant"; content: string; name?: string }> {
  const history: Array<{
    role: "system" | "user" | "assistant";
    content: string;
    name?: string;
  }> = [];
  for (const message of boundedSource(messages)) {
    const role = message["role"];
    if (role !== "system" && role !== "user" && role !== "assistant") continue;
    const content = extractBoundedMessageContent(
      message["content"],
      MAX_HISTORY_CONTENT_CHARS,
    );
    if (!content) continue;
    const name = bounded(message["name"], 256);
    history.push({ role, content, ...(name ? { name } : {}) });
  }
  return history.slice(-MAX_HISTORY_MESSAGES);
}

export function buildToolEvents(
  turnMessages: unknown[],
): Array<{
  kind: "call" | "result";
  tool_call_id: string;
  name: string;
  content: string;
}> {
  const events: Array<{
    kind: "call" | "result";
    tool_call_id: string;
    name: string;
    content: string;
  }> = [];

  for (const message of boundedSource(turnMessages)) {
    const role = message["role"];
    if (role === "assistant") {
      // OpenAI-compatible shape.
      const rawCalls = Array.isArray(message["tool_calls"])
        ? (message["tool_calls"] as unknown[]).slice(-MAX_TOOL_EVENTS)
        : [];
      for (const rawCall of rawCalls) {
        if (!rawCall || typeof rawCall !== "object") continue;
        const call = rawCall as Record<string, unknown>;
        const fn = call["function"] && typeof call["function"] === "object"
          ? (call["function"] as Record<string, unknown>)
          : {};
        events.push({
          kind: "call",
          tool_call_id: bounded(call["id"], 256),
          name: bounded(fn["name"] ?? call["name"], 256),
          content: bounded(
            fn["arguments"] ?? call["arguments"],
            MAX_TOOL_EVENT_CONTENT_CHARS,
          ),
        });
      }

      // Native pi-agent shape: ToolCall parts live inside content.
      const content = Array.isArray(message["content"])
        ? (message["content"] as unknown[]).slice(-MAX_CONTENT_PARTS)
        : [];
      for (const rawPart of content) {
        if (!rawPart || typeof rawPart !== "object") continue;
        const part = rawPart as Record<string, unknown>;
        if (part["type"] !== "toolCall") continue;
        events.push({
          kind: "call",
          tool_call_id: bounded(part["id"], 256),
          name: bounded(part["name"], 256),
          content: bounded(part["arguments"], MAX_TOOL_EVENT_CONTENT_CHARS),
        });
      }
    } else if (role === "tool" || role === "toolResult") {
      events.push({
        kind: "result",
        tool_call_id: bounded(
          message["tool_call_id"] ?? message["toolCallId"],
          256,
        ),
        name: bounded(message["name"] ?? message["toolName"], 256),
        content: extractBoundedMessageContent(
          message["content"],
          MAX_TOOL_EVENT_CONTENT_CHARS,
        ),
      });
    }
  }
  return events.slice(-MAX_TOOL_EVENTS);
}

function identifierFingerprint(message: Record<string, unknown>): string | null {
  const identity = {
    role: message["role"],
    id: message["id"] ?? message["messageId"] ?? null,
    turnId: message["turnId"] ?? null,
    responseId: message["responseId"] ?? null,
    timestamp: message["timestamp"] ?? null,
  };
  if (identity.id == null && identity.turnId == null &&
      identity.responseId == null && identity.timestamp == null) return null;
  return JSON.stringify(identity);
}

/**
 * Return a host identity, never a dialogue-content hash.
 *
 * Different legitimate turns with identical text remain distinct because
 * OpenClaw run ids (or message ids/timestamps in the compatibility fallback)
 * differ. Re-delivery of the same terminal hook retains the same identity.
 */
export function deriveTurnIdentity(
  event: AgentEndHookEvent,
  ctx: AgentEndHookContext,
  turnMessages: Array<Record<string, unknown>>,
): string | null {
  const runId = asString(event.runId || ctx.runId).trim();
  if (runId) return `run:${runId}`;
  const turnId = asString(event.turnId).trim();
  if (turnId) return `turn:${turnId}`;

  const fingerprints = turnMessages
    .slice(-MAX_SOURCE_MESSAGES)
    .map(identifierFingerprint)
    .filter((value): value is string => value !== null);
  if (fingerprints.length === 0) return null;
  const digest = createHash("sha256")
    .update(asString(ctx.sessionId))
    .update("\0")
    .update(fingerprints.join("\0"))
    .digest("hex");
  return `messages:${digest}`;
}

function canonicalTurnId(identity: string): string {
  return createHash("sha256").update(identity).digest("hex");
}

function rememberBounded(set: Set<string>, value: string): void {
  set.add(value);
  if (set.size <= RECENT_IDENTITY_LIMIT) return;
  const oldest = set.values().next().value as string | undefined;
  if (oldest !== undefined) set.delete(oldest);
}

function deriveOpenclawAgentId(ctx: AgentEndHookContext): string | null {
  const direct = asString(ctx.agentId).trim();
  if (direct) return direct;
  return parseAgentIdFromSessionKey(asString(ctx.sessionKey));
}

export function createAgentEndHook(params: AgentEndHookParams): AgentEndHookHandler {
  const { client, logger, resolveAgentId, terminalBarrier } = params;
  const terminalWorkBudgetMs = Math.max(
    100,
    params.terminalWorkBudgetMs ?? TERMINAL_WORK_BUDGET_MS,
  );
  const bootstrapped = new Set<string>();
  const observed = new Set<string>();
  const ingested = new Set<string>();

  return function agentEnd(event, ctx) {
    if (!event.success || !Array.isArray(event.messages)) return;
    const openclawAgentId = deriveOpenclawAgentId(ctx);
    if (openclawAgentId === null) return;

    const turnMessages = extractFinalTurnMessages(event.messages);
    if (turnMessages.length === 0) return;
    const identity = deriveTurnIdentity(event, ctx, turnMessages);
    if (identity === null) {
      logger.warn?.(
        "[styx] agent_end has no run/turn/message identity; capture skipped",
      );
      return;
    }

    const scope = terminalScopeKey(
      openclawAgentId,
      ctx.sessionId,
      ctx.sessionKey,
    );
    return terminalBarrier.track(scope, identity, async () => {
      const deadlineAt = Date.now() + terminalWorkBudgetMs;
      const withinDeadline = async <T>(
        work: (signal: AbortSignal) => Promise<T>,
      ): Promise<T> => {
        const remaining = deadlineAt - Date.now();
        if (remaining <= 0) throw new Error("terminal work deadline exceeded");
        const controller = new AbortController();
        let timer: ReturnType<typeof setTimeout> | undefined;
        try {
          return await Promise.race([
            work(controller.signal),
            new Promise<never>((_, reject) => {
              timer = setTimeout(
                () => {
                  controller.abort();
                  reject(new Error("terminal work deadline exceeded"));
                },
                remaining,
              );
            }),
          ]);
        } finally {
          if (timer !== undefined) clearTimeout(timer);
        }
      };
      // Re-delivery after a completed single-flight is a no-op when both
      // durable operations already succeeded.
      if (observed.has(identity) && ingested.has(identity)) return;

    let agentId: string;
    try {
      agentId = await withinDeadline(() => resolveAgentId(openclawAgentId));
    } catch (err) {
      logger.warn?.(`[styx] agent_end resolveAgentId failed: ${fmtErr(err)}`);
      return;
    }

    const sessionId = asString(ctx.sessionId);
    const bootstrapKey = `${agentId}::${sessionId}`;
    if (!bootstrapped.has(bootstrapKey)) {
      try {
        await withinDeadline((signal) => client.contextBootstrap({
          agent_id: agentId,
          session_id: sessionId || null,
        }, { signal }));
        bootstrapped.add(bootstrapKey);
      } catch (err) {
        logger.warn?.(`[styx] agent_end bootstrap failed: ${fmtErr(err)}`);
        return;
      }
    }

    let dialogueMessages: StyxMessage[] = [];
    // Bound first; only user/assistant rows belong in the durable dialogue.
    for (const message of turnMessages.slice(-MAX_SOURCE_MESSAGES)) {
      const role = message["role"];
      if (role !== "user" && role !== "assistant") continue;
      const limit = role === "user"
        ? MAX_USER_MESSAGE_CHARS
        : MAX_ASSISTANT_RESPONSE_CHARS;
      const content = extractBoundedMessageContent(message["content"], limit);
      if (content) dialogueMessages.push({ role, content });
    }

    const userMessage = bounded(
      dialogueMessages
        .filter((message) => message.role === "user")
        .slice(-MAX_HISTORY_MESSAGES)
        .map((message) => bounded(message.content, MAX_HISTORY_CONTENT_CHARS))
        .join("\n"),
      MAX_USER_MESSAGE_CHARS,
    );
    const assistantResponse = bounded(
      dialogueMessages
        .filter((message) => message.role === "assistant")
        .slice(-MAX_HISTORY_MESSAGES)
        .map((message) => bounded(message.content, MAX_HISTORY_CONTENT_CHARS))
        .join("\n"),
      MAX_ASSISTANT_RESPONSE_CHARS,
    );
    const turnId = canonicalTurnId(identity);

    if (!observed.has(identity) && userMessage.trim() && assistantResponse.trim()) {
      try {
        const observation = await withinDeadline((signal) => client.affectObserveTurn({
          agent_id: agentId,
          idempotency_key: `openclaw:${turnId}`,
          turn_id: turnId,
          session_id: sessionId || null,
          user_message: userMessage,
          assistant_response: assistantResponse,
          conversation_history: buildConversationHistory(event.messages),
          tool_events: buildToolEvents(turnMessages),
          model: asString(ctx.modelId) || null,
          platform: "openclaw",
        }, { signal }));
        if (observation.accepted) {
          rememberBounded(observed, identity);
        } else {
          logger.warn?.(
            `[styx] affect observation not accepted: ${observation.reason ?? "unknown"}`,
          );
        }
      } catch (err) {
        logger.warn?.(`[styx] affect observation failed: ${fmtErr(err)}`);
      }
    }

    // Attachment archival is a separate best-effort phase. Causal state is
    // always attempted first and the whole terminal act shares one deadline.
    if (Date.now() < deadlineAt) {
      try {
        dialogueMessages = (await withinDeadline((signal) => interceptDocumentAttachments(
          dialogueMessages as Array<Record<string, unknown>>,
          agentId,
          client,
          logger,
          {
            maxMarkers: MAX_ATTACHMENT_MARKERS,
            maxIngests: MAX_ATTACHMENT_INGESTS,
            deadlineAt,
            signal,
          },
        ))) as StyxMessage[];
      } catch (err) {
        logger.warn?.(`[styx] attachment phase bounded: ${fmtErr(err)}`);
      }
    }

    // Affect must be committed first so these diary rows snapshot the state
    // caused by this turn. A retry can independently repair failed ingestion.
    if (!ingested.has(identity) && dialogueMessages.length > 0) {
      try {
        await withinDeadline((signal) => client.contextIngestBatch({
          agent_id: agentId,
          session_id: sessionId || null,
          messages: dialogueMessages,
          idempotency_key: `openclaw:${turnId}:dialogue`,
        }, { signal }));
        rememberBounded(ingested, identity);
      } catch (err) {
        logger.warn?.(`[styx] agent_end ingest_batch failed: ${fmtErr(err)}`);
      }
    }
    });
  };
}
