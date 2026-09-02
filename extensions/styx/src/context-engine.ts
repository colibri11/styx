// ContextEngine factory — lifecycle bridge между OpenClaw runtime
// и styx-core HTTP daemon.
//
// ## Соответствие OpenClaw context-engine SDK
//
// Lifecycle params типизированы по фактическому контракту OpenClaw
// 2026.8.2 (`src/context-engine/types.ts`):
//
//   bootstrap({sessionId, sessionKey?, sessionFile})
//   ingest({sessionId, sessionKey?, message, isHeartbeat?})
//   ingestBatch({sessionId, sessionKey?, messages, isHeartbeat?})
//   assemble({sessionId, sessionKey?, messages, tokenBudget?,
//             availableTools?, citationsMode?, model?, prompt?})
//   compact({sessionId, sessionKey?, sessionFile, tokenBudget?, ...})
//   afterTurn({sessionId, sessionKey?, sessionFile, messages, ...})
//   dispose()
//
// **agentId напрямую в lifecycle params не передаётся.** Источники
// agentId в OpenClaw runtime (см. `dist/session-key-C0K0uhmG.js`):
//
// 1. `sessionKey` имеет format `agent:<agentId>:<rest>` —
//    `resolveAgentIdFromSessionKey()` парсит его. Это первичный
//    источник, доступный per-call.
// 2. `factoryCtx.agentDir` — путь типа `${stateDir}/agents/<agentId>/agent`.
//    Базируется на DEFAULT_AGENT_ID="main" если override не задан.
//    Доступен один раз в factory.
//
// **Если оба источника не дают agentId** (sessionKey в legacy/alias
// формате, agentDir не распарсивается) — это **нормальный результат**.
// Engine работает в pure passthrough: ingest/bootstrap/dispose — no-op,
// assemble возвращает messages без изменений. У такого потока нет
// дополнительной памяти из Styx — только то, что runtime уже положил
// в контекст (актуальная задача, статические системные данные). Это
// Без подтверждённого agent scope нельзя безопасно выбрать изолированную
// память; это техническая граница tenancy, не утверждение о личности.
//
// OpenClaw 2026.8.2 принимает durable turn через ContextEngine.commitTurn.
// assemble создаёт session-fenced cognitive snapshot; commitTurn атомарно
// связывает его с принятым logical turn и допускает restart-safe retries.

import { buildMemorySystemPromptAddition } from "openclaw/plugin-sdk/core";
import {
  parseAgentIdFromAgentDir,
  parseAgentIdFromSessionKey,
} from "./agent-id-shared.js";
import {
  fmtErr,
  type StyxClient,
  type StyxLogger,
  type StyxMessage,
} from "./client.js";
import { interceptDocumentAttachments } from "./media-attachments.js";
import { fetchCanonicalPreturn } from "./cognition-preturn.js";
import {
  normalizeIdentifier,
  openclawHostKey,
} from "./identifiers.js";
import { boundedPreturnMessages } from "./hooks/before-prompt-build.js";
import {
  bounded,
  buildConversationHistory,
  buildToolEvents,
  extractBoundedMessageContent,
  extractFinalTurnMessages,
  MAX_ASSISTANT_RESPONSE_CHARS,
  MAX_HISTORY_CONTENT_CHARS,
  MAX_HISTORY_MESSAGES,
  MAX_SOURCE_MESSAGES,
  MAX_USER_MESSAGE_CHARS,
} from "./hooks/agent-end.js";

export type ResolveAgentId = (openclawAgentId: string) => Promise<string>;

export type StyxContextEngineParams = {
  client: StyxClient;
  ctx: Record<string, unknown>;
  logger: StyxLogger;
  resolveAgentId: ResolveAgentId;
  ownsCompaction: boolean;
};

type LifecycleParams = Record<string, unknown>;

function asString(v: unknown, fallback = ""): string {
  if (typeof v === "string") return v;
  if (v == null) return fallback;
  return String(v);
}

function extractSessionId(params: LifecycleParams): string {
  return asString(params["sessionId"]);
}

function extractMessages(params: LifecycleParams): StyxMessage[] {
  const raw = params["messages"];
  if (!Array.isArray(raw)) return [];
  return raw.map((m) => {
    if (m && typeof m === "object") {
      const msg = m as Record<string, unknown>;
      return {
        role: asString(msg["role"], "user"),
        content: extractMessageContent(msg["content"]),
      };
    }
    return { role: "user", content: "" };
  });
}

/**
 * Преобразует AgentMessage.content в плоский string для записи в Styx.
 *
 * pi-agent-core / OpenClaw используют расширенный shape: `content` может
 * быть либо string, либо array of content parts:
 *   [{type:'text', text:'...'}, {type:'image', url:'...'}, ...]
 * (см. OpenClaw plugin-sdk AgentMessage).
 *
 * Styx core хранит plain string. Для multimodal turn'а склеиваем текст
 * всех `text`-частей через '\n' — image/audio/tool-call parts в memories
 * не пишем (это решит будущая волна на multimodal recall). Если ничего
 * текстового не нашли — возвращаем "".
 *
 * Без этого преобразования lifecycle ingest слал `String([object Object])`
 * в core, и в memories.content попадало `[object Object]`.
 */
function extractMessageContent(content: unknown): string {
  if (typeof content === "string") return content;
  if (content == null) return "";
  if (Array.isArray(content)) {
    const parts: string[] = [];
    for (const part of content) {
      if (typeof part === "string") {
        parts.push(part);
      } else if (part && typeof part === "object") {
        const p = part as Record<string, unknown>;
        const t = p["type"];
        if (t === "text" || t === undefined) {
          const txt = p["text"];
          if (typeof txt === "string" && txt) parts.push(txt);
        }
        // image/audio/tool_use parts intentionally skipped — Styx
        // memories хранит только text (см. concept выше).
      }
    }
    return parts.join("\n");
  }
  // Произвольный shape (нечасто) — сериализуем как fallback.
  if (typeof content === "object") {
    try {
      return JSON.stringify(content);
    } catch {
      return "";
    }
  }
  return String(content);
}

/**
 * Извлечение openclawAgentId из lifecycle params + factory ctx.
 *
 * Возвращает null когда ни один источник не даёт identifier — это
 * нормальная ветка (sessionKey в legacy/alias format или anonymous
 * runtime), не ошибка. Caller должен интерпретировать null как
 * «engine работает в passthrough, Styx не подключается».
 */
export function deriveOpenclawAgentId(
  params: LifecycleParams,
  ctx: Record<string, unknown>,
): string | null {
  const directTarget = params["sessionTarget"];
  if (directTarget && typeof directTarget === "object") {
    const direct = normalizeIdentifier(
      (directTarget as Record<string, unknown>)["agentId"],
      256,
    );
    if (direct) return direct;
  }
  const runtimeContext = params["runtimeContext"];
  if (runtimeContext && typeof runtimeContext === "object") {
    const sessionTarget = (runtimeContext as Record<string, unknown>)["sessionTarget"];
    if (sessionTarget && typeof sessionTarget === "object") {
      const direct = normalizeIdentifier(
        (sessionTarget as Record<string, unknown>)["agentId"],
        256,
      );
      if (direct) return direct;
    }
  }
  const fromSession = parseAgentIdFromSessionKey(asString(params["sessionKey"]));
  if (fromSession) return fromSession;
  // Fallback: распознаём `${stateDir}/agents/<agentId>/agent` или
  // `${stateDir}/agents/<agentId>` (legacy форма без trailing /agent).
  return parseAgentIdFromAgentDir(asString(ctx["agentDir"]));
}

export function createStyxContextEngine(params: StyxContextEngineParams) {
  const {
    client, ctx, logger, resolveAgentId, ownsCompaction,
  } = params;

  // (agentId, sessionId) ⇒ уже bootstrap'нули в core. Защищает от
  // повторных bootstrap при ingest, если runtime не зовёт engine.bootstrap
  // явно перед ingest.
  const bootstrapped = new Set<string>();

  function bootstrapKey(agentId: string, sessionId: string): string {
    return `${agentId}::${sessionId}`;
  }

  /**
   * Resolve openclawAgentId → Styx agent_id и обеспечить, что core
   * проинициализирован для этой (agent, session) пары.
   *
   * Возвращает Styx agent_id когда engine должен делать работу;
   * null — passthrough ветка (нет linked agent или bootstrap упал).
   */
  async function ensureAgentForCall(
    openclawAgentId: string | null,
    sessionId: string,
  ): Promise<string | null> {
    if (openclawAgentId === null) {
      return null;
    }
    let agentId: string;
    try {
      agentId = normalizeIdentifier(await resolveAgentId(openclawAgentId), 256);
    } catch (err) {
      // Одноаргументный формат — OpenClaw createPluginLogger глотает
      // второй arg, err тонул silent'но и маскировал ingest faults
      // (Phase E root cause).
      logger.warn?.(
        `[styx] resolveAgentId(${openclawAgentId}) failed: ${fmtErr(err)}`,
      );
      return null;
    }
    const k = bootstrapKey(agentId, sessionId);
    if (!bootstrapped.has(k)) {
      try {
        await client.contextBootstrap({
          agent_id: agentId,
          session_id: sessionId || null,
        });
        bootstrapped.add(k);
      } catch (err) {
        logger.warn?.(
          `[styx] bootstrap failed (${agentId}/${sessionId}): ${fmtErr(err)}`,
        );
        return null;
      }
    }
    return agentId;
  }

  return {
    info: {
      id: "styx",
      name: "Styx",
      version: "0.4.0",
      ownsCompaction,
      acceptedHostParams: ["sessionTarget", "runtimeSettings", "runtimeContext"],
      transcriptSemantics: {
        currentTurnFence: "before-current-turn-entry-v1" as const,
        turnAdvancementIdempotency: "atomic-idempotent-v1" as const,
      },
      turnMaintenanceMode: "background" as const,
    },

    async bootstrap(opts: LifecycleParams) {
      const openclawAgentId = deriveOpenclawAgentId(opts, ctx);
      const sessionId = normalizeIdentifier(extractSessionId(opts), 256);
      const agentId = await ensureAgentForCall(openclawAgentId, sessionId);
      // BootstrapResult contract: {bootstrapped, importedMessages?, reason?}.
      return {
        bootstrapped: agentId !== null,
        ...(agentId === null
          ? { reason: "no-styx-agent" as const }
          : {}),
      };
    },

    async ingest(_opts: LifecycleParams) {
      // Durable accepted turns are owned by commitTurn. This method remains a
      // compatibility no-op so host lifecycle fallbacks cannot double-write.
      return { ingested: false };
    },

    async ingestBatch(_opts: LifecycleParams) {
      return { ingestedCount: 0 };
    },

    async assemble(opts: LifecycleParams) {
      // Сохраняем оригинальные messages с богатым AgentMessage shape
      // (tool_calls/name/timestamps...) — passthrough и обратный путь
      // должны возвращать их без потерь. extractMessages используется
      // только для оценки токенов и для отправки в core (которому
      // достаточно role+content).
      const rawMessages = Array.isArray(opts["messages"])
        ? (opts["messages"] as Array<Record<string, unknown>>)
        : [];
      const styxMessages = extractMessages(opts);
      const openclawAgentId = deriveOpenclawAgentId(opts, ctx);
      const sessionId = normalizeIdentifier(extractSessionId(opts), 256);

      // Anonymous поток (нет связанного Styx agent'а) — pure passthrough.
      // Без agent scope нельзя безопасно выбрать tenant. Runtime получает
      // исходные messages без Styx-инъекций и без доступа к чужим traces.
      if (openclawAgentId === null) {
        return {
          messages: rawMessages,
          estimatedTokens: roughTokenEstimate(styxMessages),
        };
      }

      const agentId = await ensureAgentForCall(openclawAgentId, sessionId);
      if (agentId === null) {
        return {
          messages: rawMessages,
          estimatedTokens: roughTokenEstimate(styxMessages),
        };
      }

      try {
        const request = {
          agent_id: agentId,
          session_id: sessionId || null,
          messages: boundedPreturnMessages(rawMessages),
          query: bounded(opts["prompt"], 20_000) || null,
          // OpenClaw owns windowing and may retry the same logical prompt on a
          // fallback model. Keep the cognitive envelope stable across attempts.
          token_budget: null,
          model: null,
          platform: "openclaw",
          extra: {},
        };
        const resp = await fetchCanonicalPreturn(client, request);
        const memoryAddition = buildMemorySystemPromptAddition({
          availableTools: opts["availableTools"] instanceof Set
            ? (opts["availableTools"] as Set<string>)
            : new Set<string>(),
          citationsMode: typeof opts["citationsMode"] === "string"
            ? opts["citationsMode"] as never
            : undefined,
        });
        const additions = [memoryAddition, resp.system_prompt_addition]
          .filter((value): value is string => Boolean(value?.trim()));
        const out: {
          messages: Array<Record<string, unknown>>;
          estimatedTokens: number;
          systemPromptAddition?: string;
          promptAuthority?: "assembled" | "preassembly_may_overflow";
        } = {
          // Preserve exact OpenClaw AgentMessage objects, including tool calls,
          // multimodal parts and provider metadata. Styx only adds context.
          messages: rawMessages,
          estimatedTokens: roughTokenEstimate(styxMessages),
        };
        if (additions.length > 0) {
          out.systemPromptAddition = additions.join("\n\n");
        }
        return out;
      } catch (err) {
        logger.warn?.(`[styx] canonical preturn failed (passthrough): ${fmtErr(err)}`);
        return {
          messages: rawMessages,
          estimatedTokens: roughTokenEstimate(styxMessages),
        };
      }
    },

    async commitTurn(opts: LifecycleParams) {
      if (Boolean(opts["isHeartbeat"])) {
        return { status: "committed" as const };
      }
      const advancementKey = normalizeIdentifier(opts["advancementKey"], 384);
      if (!advancementKey) {
        throw new Error("OpenClaw commitTurn has no advancementKey");
      }
      const runtimeTarget = opts["sessionTarget"];
      const target = runtimeTarget && typeof runtimeTarget === "object"
        ? runtimeTarget as Record<string, unknown>
        : {};
      const callerSessionId = normalizeIdentifier(opts["sessionId"], 256);
      const targetSessionId = normalizeIdentifier(target["sessionId"], 256);
      const callerSessionKey = normalizeIdentifier(opts["sessionKey"], 256);
      const targetSessionKey = normalizeIdentifier(target["sessionKey"], 256);
      if (callerSessionId && targetSessionId && callerSessionId !== targetSessionId) {
        throw new Error("OpenClaw commitTurn sessionTarget conflicts with sessionId");
      }
      if (callerSessionKey && targetSessionKey && callerSessionKey !== targetSessionKey) {
        throw new Error("OpenClaw commitTurn sessionTarget conflicts with sessionKey");
      }
      const sessionId = targetSessionId || callerSessionId;
      const openclawAgentId = normalizeIdentifier(target["agentId"], 256)
        || deriveOpenclawAgentId({
          ...opts,
          sessionKey: targetSessionKey || callerSessionKey,
        }, ctx);
      const sessionKeyAgentId = parseAgentIdFromSessionKey(
        targetSessionKey || callerSessionKey,
      );
      if (
        openclawAgentId && sessionKeyAgentId
        && openclawAgentId !== sessionKeyAgentId
      ) {
        throw new Error("OpenClaw commitTurn sessionTarget conflicts with agent identity");
      }
      const agentId = await ensureAgentForCall(openclawAgentId, sessionId);
      if (agentId === null) {
        throw new Error("Styx agent cannot be resolved for durable commitTurn");
      }

      const acceptedMessages = Array.isArray(opts["messages"])
        ? opts["messages"] as unknown[]
        : [];
      const turnMessages = extractFinalTurnMessages(acceptedMessages);
      const dialogue: StyxMessage[] = [];
      for (const message of turnMessages.slice(-MAX_SOURCE_MESSAGES)) {
        const role = message["role"];
        if (role !== "user" && role !== "assistant") continue;
        const limit = role === "user"
          ? MAX_USER_MESSAGE_CHARS
          : MAX_ASSISTANT_RESPONSE_CHARS;
        const content = extractBoundedMessageContent(message["content"], limit);
        if (content) dialogue.push({ role, content });
      }
      const userMessage = bounded(
        dialogue.filter((message) => message.role === "user")
          .slice(-MAX_HISTORY_MESSAGES)
          .map((message) => bounded(message.content, MAX_HISTORY_CONTENT_CHARS))
          .join("\n"),
        MAX_USER_MESSAGE_CHARS,
      );
      const assistantResponse = bounded(
        dialogue.filter((message) => message.role === "assistant")
          .slice(-MAX_HISTORY_MESSAGES)
          .map((message) => bounded(message.content, MAX_HISTORY_CONTENT_CHARS))
          .join("\n"),
        MAX_ASSISTANT_RESPONSE_CHARS,
      );
      const admission = opts["admission"] && typeof opts["admission"] === "object"
        ? opts["admission"] as Record<string, unknown>
        : {};
      const terminal = opts["terminal"] && typeof opts["terminal"] === "object"
        ? opts["terminal"] as Record<string, unknown>
        : {};
      const result = await client.cognitionCommit({
        agent_id: agentId,
        host_key: openclawHostKey(advancementKey),
        session_id: sessionId || null,
        snapshot_policy: "latest_session",
        parent_policy: "latest_session",
        status: "completed",
        user_message: userMessage,
        assistant_response: assistantResponse,
        conversation_history: buildConversationHistory(acceptedMessages),
        tool_events: buildToolEvents(turnMessages),
        consequences: [],
        model: null,
        platform: "openclaw",
        extra: {
          projection_scope: "accepted_durable_turn",
          logical_turn_id: bounded(admission["logicalTurnId"], 256),
          admission_entry_id: bounded(admission["entryId"], 256),
          admission_generation: bounded(admission["generation"], 64),
          terminal_entry_id: bounded(terminal["entryId"], 256),
          terminal_generation: bounded(terminal["generation"], 64),
        },
      });

      // Document archival is independent from causal advancement. A failure
      // must not keep OpenClaw's accepted-turn outbox stuck forever.
      try {
        await interceptDocumentAttachments(
          dialogue as Array<Record<string, unknown>>,
          agentId,
          client,
          logger,
        );
      } catch (err) {
        logger.warn?.(`[styx] accepted-turn attachment archival failed: ${fmtErr(err)}`);
      }
      return { status: result.duplicate ? "duplicate" as const : "committed" as const };
    },

    async compact(opts: LifecycleParams) {
      const openclawAgentId = deriveOpenclawAgentId(opts, ctx);
      const sessionId = normalizeIdentifier(extractSessionId(opts), 256);
      const agentId = await ensureAgentForCall(openclawAgentId, sessionId);
      if (agentId === null) {
        // Anonymous поток — runtime может /compact, но Styx нечего
        // сжимать (он не writeл ничего). compacted:false означает «no
        // change», runtime продолжит с теми же messages.
        return { ok: true, compacted: false, reason: "no-styx-agent" };
      }
      try {
        const resp = await client.contextCompact({
          agent_id: agentId,
          session_id: sessionId || null,
          force: Boolean(opts["force"]),
        });
        return {
          ok: resp.ok,
          compacted: resp.compacted,
          ...(resp.reason ? { reason: resp.reason } : {}),
        };
      } catch (err) {
        logger.warn?.(`[styx] compact failed: ${fmtErr(err)}`);
        return { ok: true, compacted: false, reason: "compact-error" };
      }
    },

    async afterTurn(opts: LifecycleParams) {
      const openclawAgentId = deriveOpenclawAgentId(opts, ctx);
      const sessionId = normalizeIdentifier(extractSessionId(opts), 256);
      const agentId = await ensureAgentForCall(openclawAgentId, sessionId);
      if (agentId === null) {
        return;
      }
      // Maintenance only. Durable dialogue/affect ownership belongs to the
      // accepted-turn commitTurn outbox contract.
      const rawMessages = Array.isArray(opts["messages"])
        ? (opts["messages"] as Array<Record<string, unknown>>)
        : [];
      try {
        await client.contextAfterTurn({
          agent_id: agentId,
          session_id: sessionId || null,
          messages: rawMessages,
        });
      } catch (err) {
        // afterTurn — fire-and-forget, ошибка не должна влиять на
        // следующий turn.
        logger.warn?.(`[styx] after_turn failed: ${fmtErr(err)}`);
      }
    },

    async dispose() {
      try {
        await client.contextDispose({});
      } catch (err) {
        logger.warn?.(`[styx] dispose failed: ${fmtErr(err)}`);
      }
      bootstrapped.clear();
    },
  };
}

function roughTokenEstimate(messages: StyxMessage[]): number {
  return messages.reduce(
    (acc, m) => acc + Math.ceil((m.content ?? "").length / 4),
    0,
  );
}
