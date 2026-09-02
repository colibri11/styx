// Legacy typed `agent_end` adapter plus shared bounded message/tool projection.
// Current OpenClaw v2026.8.2 does not register this hook: accepted finality is
// owned by ContextEngine.commitTurn and its durable host outbox.
//
// ContextEngine.afterTurn is not a finality guarantee: a host may invoke it
// between model/tool-loop iterations.  `agent_end` is emitted after the loop
// settles and therefore prevents intermediate assistant/tool-call messages
// from becoming duplicate diary entries or partial cognitive acts.

import { createHash } from "node:crypto";

import {
  parseAgentIdFromSessionKey,
} from "../agent-id-shared.js";
import {
  fmtErr,
  StyxHttpError,
  type StyxClient,
  type StyxLogger,
  type StyxMessage,
} from "../client.js";
import { interceptDocumentAttachments } from "../media-attachments.js";
import {
  normalizeIdentifier,
  openclawHostKey,
  runOrTurnIdentity,
} from "../identifiers.js";
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
  turnId?: string;
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
export const MAX_TOOL_EVENTS = 64;
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
  kind: "call" | "result" | "error";
  tool_event_id: string;
  name: string;
  content: string;
  metadata: Record<string, string>;
}> {
  const events: Array<{
    kind: "call" | "result" | "error";
    tool_event_id: string;
    name: string;
    content: string;
    metadata: Record<string, string>;
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
          tool_event_id: normalizeIdentifier(call["id"], 256),
          name: bounded(fn["name"] ?? call["name"], 256),
          content: bounded(
            fn["arguments"] ?? call["arguments"],
            MAX_TOOL_EVENT_CONTENT_CHARS,
          ),
          metadata: {},
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
          tool_event_id: normalizeIdentifier(part["id"], 256),
          name: bounded(part["name"], 256),
          content: bounded(part["arguments"], MAX_TOOL_EVENT_CONTENT_CHARS),
          metadata: {},
        });
      }
    } else if (role === "tool" || role === "toolResult") {
      events.push({
        kind: message["isError"] === true || message["is_error"] === true
          ? "error" : "result",
        tool_event_id: normalizeIdentifier(
          message["tool_call_id"] ?? message["toolCallId"],
          256,
        ),
        name: bounded(message["name"] ?? message["toolName"], 256),
        content: extractBoundedMessageContent(
          message["content"],
          MAX_TOOL_EVENT_CONTENT_CHARS,
        ),
        metadata: {},
      });
    }
  }
  return events.slice(-MAX_TOOL_EVENTS);
}

function identifierFingerprint(message: Record<string, unknown>): string | null {
  const identity = {
    role: message["role"],
    id: normalizeIdentifier(message["id"] ?? message["messageId"], 256) || null,
    turnId: normalizeIdentifier(message["turnId"], 256) || null,
    responseId: normalizeIdentifier(message["responseId"], 256) || null,
    timestamp: normalizeIdentifier(message["timestamp"], 64) || null,
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
  const physicalIdentity = runOrTurnIdentity(
    event.runId || ctx.runId,
    event.turnId || ctx.turnId,
  );
  if (physicalIdentity) return physicalIdentity;

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
  const committed = new Set<string>();
  const legacyCompleted = new Set<string>();
  const legacyAffectCompleted = new Set<string>();
  const legacyDiaryCompleted = new Set<string>();

  return function agentEnd(event, ctx) {
    if (!Array.isArray(event.messages)) return;
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

    const sessionId = normalizeIdentifier(ctx.sessionId, 256);
    const scope = terminalScopeKey(
      normalizeIdentifier(openclawAgentId, 256),
      sessionId,
      normalizeIdentifier(ctx.sessionKey, 256),
    );
    // Declare the physical act before any awaited resolve/bootstrap work.
    // Even an early host-adapter failure therefore remains the predecessor of
    // the next act, and re-delivery gets identical coordinates.
    const hostKey = openclawHostKey(identity);
    const turnId = hostKey.slice("openclaw:".length);
    const completionKey = `${scope}\0${identity}`;
    const declared = terminalBarrier.declareAct(scope, identity, hostKey);
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
      // Re-delivery after a completed single-flight is a no-op.
      if (committed.has(completionKey) || legacyCompleted.has(completionKey)) return;

    let agentId: string;
    try {
      agentId = normalizeIdentifier(
        await withinDeadline(() => resolveAgentId(openclawAgentId)),
        256,
      );
    } catch (err) {
      logger.warn?.(`[styx] agent_end resolveAgentId failed: ${fmtErr(err)}`);
      return;
    }

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
    const history = buildConversationHistory(event.messages);
    const toolEvents = buildToolEvents(turnMessages);

    try {
      const result = await withinDeadline((signal) => client.cognitionCommit({
          agent_id: agentId,
          host_key: declared.actKey,
          parent_host_key: declared.parentActKey,
          session_id: sessionId || null,
          snapshot_token: declared.snapshotToken,
          status: event.success ? "completed" : "failed",
          user_message: userMessage,
          assistant_response: assistantResponse,
          conversation_history: history,
          tool_events: toolEvents,
          consequences: [],
          model: normalizeIdentifier(ctx.modelId, 512) || null,
          platform: "openclaw",
          extra: {
            projection_scope: "finalized_channel_output",
            run_id: normalizeIdentifier(event.runId ?? ctx.runId, 256),
            duration_ms: typeof event.durationMs === "number"
              ? String(event.durationMs) : "",
            terminal_error: bounded(event.error, 1_000),
          },
        }, { signal }));
      if (result.committed || result.duplicate) {
        rememberBounded(committed, completionKey);
      }
    } catch (err) {
      if (err instanceof StyxHttpError && err.status === 404) {
        // Mixed-version deployment only. Affect and diary are independent:
        // failure in one phase cannot suppress the other, and successful
        // phases remain locally duplicate-safe on a partial retry.
        const affectRequired = Boolean(
          event.success && userMessage.trim() && assistantResponse.trim(),
        );
        let affectDone = !affectRequired || legacyAffectCompleted.has(completionKey);
        if (!affectDone) {
          try {
            await withinDeadline((signal) => client.affectObserveTurn({
              agent_id: agentId,
              idempotency_key: hostKey,
              turn_id: turnId,
              session_id: sessionId || null,
              user_message: userMessage,
              assistant_response: assistantResponse,
              conversation_history: history,
              tool_events: toolEvents
                .filter((item) => item.kind !== "error")
                .map((item) => ({
                  kind: item.kind as "call" | "result",
                  tool_call_id: item.tool_event_id,
                  name: item.name,
                  content: item.content,
                })),
              model: normalizeIdentifier(ctx.modelId, 512) || null,
              platform: "openclaw",
            }, { signal }));
            rememberBounded(legacyAffectCompleted, completionKey);
            affectDone = true;
          } catch (affectErr) {
            logger.warn?.(`[styx] terminal legacy affect failed: ${fmtErr(affectErr)}`);
          }
        }

        const diaryRequired = dialogueMessages.length > 0;
        let diaryDone = !diaryRequired || legacyDiaryCompleted.has(completionKey);
        if (!diaryDone) {
          try {
            await withinDeadline((signal) => client.syncTurn({
              agent_id: agentId,
              session_id: sessionId || null,
              user_content: userMessage,
              assistant_content: assistantResponse,
              tool_calls: toolEvents,
              idempotency_key: hostKey,
            }, { signal }));
            rememberBounded(legacyDiaryCompleted, completionKey);
            diaryDone = true;
          } catch (diaryErr) {
            logger.warn?.(`[styx] terminal legacy diary failed: ${fmtErr(diaryErr)}`);
          }
        }
        if (affectDone && diaryDone) rememberBounded(legacyCompleted, completionKey);
      } else {
        logger.warn?.(`[styx] cognition commit failed: ${fmtErr(err)}`);
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

    });
  };
}
