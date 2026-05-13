// ContextEngine factory — lifecycle bridge между OpenClaw runtime
// и styx-core HTTP daemon.
//
// ## Соответствие OpenClaw context-engine SDK
//
// Lifecycle params типизированы по фактическому контракту OpenClaw
// 2026.5.7 (`dist/plugin-sdk/src/context-engine/types.d.ts`):
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
// соответствует IAmBook §IV: Locus принадлежит конкретному агенту-как-
// личности; нет имени → нет линии `я` → нет Styx-обвязки.
//
// ## Phase C — assemble + compact + afterTurn
//
// Phase B сделал bootstrap/ingest/ingestBatch/dispose. Phase C добавляет
// HTTP вызовы для assemble (через /context/assemble → StyxComposer
// head+tail+salient), compact (через /context/compact → memory_
// consolidation sweep) и afterTurn (drift check + sweep ticks). Stub'ы
// удалены.

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
 * (см. @mariozechner/pi-agent-core).
 *
 * Styx core хранит plain string. Для multimodal turn'а склеиваем текст
 * всех `text`-частей через '\n' — image/audio/tool-call parts в memories
 * не пишем (это решит будущая волна на multimodal recall). Если ничего
 * текстового не нашли — возвращаем "".
 *
 * Без этого фикса ContextEngine.afterTurn слал `String([object Object])`
 * в /context/ingest_batch, и в memories.content попадало `[object Object]`
 * (Phase E e2e наблюдение).
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
  const fromSession = parseAgentIdFromSessionKey(asString(params["sessionKey"]));
  if (fromSession) return fromSession;
  // Fallback: распознаём `${stateDir}/agents/<agentId>/agent` или
  // `${stateDir}/agents/<agentId>` (legacy форма без trailing /agent).
  return parseAgentIdFromAgentDir(asString(ctx["agentDir"]));
}

export function createStyxContextEngine(params: StyxContextEngineParams) {
  const { client, ctx, logger, resolveAgentId, ownsCompaction } = params;

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
      agentId = await resolveAgentId(openclawAgentId);
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
      ownsCompaction,
    },

    async bootstrap(opts: LifecycleParams) {
      const openclawAgentId = deriveOpenclawAgentId(opts, ctx);
      const sessionId = extractSessionId(opts);
      const agentId = await ensureAgentForCall(openclawAgentId, sessionId);
      // BootstrapResult contract: {bootstrapped, importedMessages?, reason?}.
      return {
        bootstrapped: agentId !== null,
        ...(agentId === null
          ? { reason: "no-styx-agent" as const }
          : {}),
      };
    },

    async ingest(opts: LifecycleParams) {
      if (Boolean(opts["isHeartbeat"])) {
        return { ingested: false };
      }
      const message = opts["message"];
      if (!message || typeof message !== "object") {
        return { ingested: false };
      }
      const openclawAgentId = deriveOpenclawAgentId(opts, ctx);
      const sessionId = extractSessionId(opts);
      const agentId = await ensureAgentForCall(openclawAgentId, sessionId);
      if (agentId === null) {
        return { ingested: false };
      }
      const m = message as Record<string, unknown>;
      try {
        const resp = await client.contextIngest({
          agent_id: agentId,
          session_id: sessionId || null,
          message: {
            role: asString(m["role"], "user"),
            content: asString(m["content"], ""),
          },
          is_heartbeat: false,
        });
        return { ingested: resp.ingested };
      } catch (err) {
        logger.warn?.(`[styx] ingest failed: ${fmtErr(err)}`);
        return { ingested: false };
      }
    },

    async ingestBatch(opts: LifecycleParams) {
      if (Boolean(opts["isHeartbeat"])) {
        return { ingestedCount: 0 };
      }
      const messages = extractMessages(opts);
      if (messages.length === 0) {
        return { ingestedCount: 0 };
      }
      const openclawAgentId = deriveOpenclawAgentId(opts, ctx);
      const sessionId = extractSessionId(opts);
      const agentId = await ensureAgentForCall(openclawAgentId, sessionId);
      if (agentId === null) {
        return { ingestedCount: 0 };
      }
      try {
        const resp = await client.contextIngestBatch({
          agent_id: agentId,
          session_id: sessionId || null,
          messages,
        });
        return { ingestedCount: resp.ingested_count };
      } catch (err) {
        logger.warn?.(`[styx] ingest_batch failed: ${fmtErr(err)}`);
        return { ingestedCount: 0 };
      }
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
      const sessionId = extractSessionId(opts);

      // Anonymous поток (нет связанного Styx agent'а) — pure passthrough.
      // По концепции (IAmBook §IV) Locus формируется только для линии
      // `я` конкретного агента; runtime передаёт сюда messages в виде
      // как оно есть, и engine ничего не добавляет (никаких salient
      // injections, никакого recall из чужих memories).
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
        const resp = await client.contextAssemble({
          agent_id: agentId,
          session_id: sessionId || null,
          messages: rawMessages,
          token_budget:
            typeof opts["tokenBudget"] === "number"
              ? (opts["tokenBudget"] as number)
              : null,
          available_tools: extractAvailableTools(opts),
          citations_mode: asString(opts["citationsMode"]) || null,
          model: asString(opts["model"]) || null,
          prompt: asString(opts["prompt"]) || null,
        });
        const out: {
          messages: Array<Record<string, unknown>>;
          estimatedTokens: number;
          systemPromptAddition?: string;
          promptAuthority?: "assembled" | "preassembly_may_overflow";
        } = {
          messages: resp.messages,
          estimatedTokens: resp.estimated_tokens,
        };
        if (resp.system_prompt_addition) {
          out.systemPromptAddition = resp.system_prompt_addition;
        }
        if (resp.prompt_authority === "assembled" ||
            resp.prompt_authority === "preassembly_may_overflow") {
          out.promptAuthority = resp.prompt_authority;
        }
        return out;
      } catch (err) {
        logger.warn?.(`[styx] assemble failed (passthrough): ${fmtErr(err)}`);
        return {
          messages: rawMessages,
          estimatedTokens: roughTokenEstimate(styxMessages),
        };
      }
    },

    async compact(opts: LifecycleParams) {
      const openclawAgentId = deriveOpenclawAgentId(opts, ctx);
      const sessionId = extractSessionId(opts);
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
      const sessionId = extractSessionId(opts);
      const agentId = await ensureAgentForCall(openclawAgentId, sessionId);
      if (agentId === null) {
        return;
      }
      const rawMessages = Array.isArray(opts["messages"])
        ? (opts["messages"] as Array<Record<string, unknown>>)
        : [];

      // OpenClaw runtime в `agent --local` (Phase E observation) НЕ зовёт
      // engine.ingest()/ingestBatch() в lifecycle. Только bootstrap →
      // assemble → afterTurn → dispose. Чтобы реально записать turn в
      // память (это основное требование Phase E «write+recall работает»),
      // выполняем ingest именно здесь.
      //
      // afterTurn получает полный transcript turn'а (включая историю).
      // Чтобы не плодить дубликаты, шлём только хвостовой фрагмент:
      // последние user+assistant сообщения. Если turn содержит больше
      // одной user-реплики (multi-step с tools), берём всё начиная с
      // последнего user-message — это и есть «новые» messages турна.
      //
      // Альтернатива (full transcript ingest_batch + content_hash dedup
      // на стороне core) сейчас не работает: /context/ingest_batch не
      // имеет content_hash idempotency (см. routes/context.py::sync_turn).
      const tail = extractLastTurnMessages(rawMessages);
      if (tail.length > 0) {
        // Свернуть до StyxMessage shape (role+content) — core
        // /context/ingest_batch принимает только это; tool_calls/name
        // и прочие OpenClaw-специфичные поля игнорируются на core
        // (model_config extra=ignore).
        const styxTail: StyxMessage[] = tail.map((m) => ({
          role: asString(m["role"], "user"),
          // extractMessageContent — handle multimodal AgentMessage shape
          // (content: string | Array<{type:'text', text:'...'}>) — без
          // него получим '[object Object]' в memories.
          content: extractMessageContent(m["content"]),
        }));
        try {
          await client.contextIngestBatch({
            agent_id: agentId,
            session_id: sessionId || null,
            messages: styxTail,
          });
        } catch (err) {
          logger.warn?.(`[styx] after_turn ingest_batch failed: ${fmtErr(err)}`);
        }
      }

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

/**
 * Извлекает «свежие» messages последнего turn'а из transcript'а.
 *
 * Стратегия: ищем индекс последнего user-message в списке (с конца) и
 * возвращаем хвост [user, ...assistant/tool/...] от него. Это покрывает
 * типовой случай OpenClaw turn'а:
 *   single-step: [..., user, assistant] → [user, assistant]
 *   multi-step (tools): [..., user, assistant(tool_call), tool, assistant]
 *                       → [user, assistant, tool, assistant]
 *
 * Если user-сообщений нет вовсе (системный init turn), возвращаем
 * только последний assistant message (если есть) — он представляет
 * «выход» turn'а. Если нет ни user ни assistant — пустой массив.
 *
 * Для записи в memories через /context/ingest_batch ядро (sync_turn)
 * фильтрует role∈{user,assistant} само, поэтому system/tool messages
 * проигнорируются на стороне core безопасно.
 */
function extractLastTurnMessages(
  rawMessages: Array<Record<string, unknown>>,
): Array<Record<string, unknown>> {
  if (rawMessages.length === 0) return [];

  let lastUserIdx = -1;
  for (let i = rawMessages.length - 1; i >= 0; i--) {
    const role = asString(rawMessages[i]?.["role"]);
    if (role === "user") {
      lastUserIdx = i;
      break;
    }
  }

  if (lastUserIdx >= 0) {
    return rawMessages.slice(lastUserIdx);
  }

  // Нет user в transcript'е — fallback на последний assistant.
  for (let i = rawMessages.length - 1; i >= 0; i--) {
    const role = asString(rawMessages[i]?.["role"]);
    if (role === "assistant") {
      return [rawMessages[i]!];
    }
  }
  return [];
}

function roughTokenEstimate(messages: StyxMessage[]): number {
  return messages.reduce(
    (acc, m) => acc + Math.ceil((m.content ?? "").length / 4),
    0,
  );
}

function extractAvailableTools(params: LifecycleParams): string[] | null {
  const raw = params["availableTools"];
  if (raw instanceof Set) return Array.from(raw).map(String);
  if (Array.isArray(raw)) return raw.map(String);
  return null;
}
