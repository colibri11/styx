// Legacy before_prompt_build compatibility adapter.
// Current OpenClaw v2026.8.2 injects cognition from ContextEngine.assemble.
// через appendSystemContext (мини-волна 26.8).
//
// Архитектурный контекст. Pi-embedded-runner (от @mariozechner/
// pi-coding-agent, используется при `runner=embedded` +
// `winnerProvider=openai-codex`) НЕ применяет результат
// ContextEngine.assemble lifecycle к финальному provider request.
// У него собственный `_rebuildSystemPrompt` который строит system
// prompt из workspace-файлов плюс **hook-результата**:
// `joinPresentTextSegments([prependSystem, baseSystemPrompt,
// appendSystem])` (см. checkout 2026.4.15:
// `src/agents/pi-embedded-runner/run/attempt.thread-helpers.ts:23`).
//
// `appendSystem` приходит как `appendSystemContext` поле из
// результата hook'а `before_prompt_build` (recommended) или legacy
// `before_agent_start` (deprecated — `event.messages` может быть
// undefined). См. SDK типы:
//   - `plugin-sdk/.../hook-types.d.ts:553` — handler signature.
//   - `plugin-sdk/.../hook-before-agent-start.types.d.ts:17-36` —
//     Event + Result shapes.
//
// Wave 37 makes `/cognition/preturn` the one core pre-generation call.
// ContextEngine.assemble applies its normalized messages; this hook is the
// single injection seam for both embedded and harness runners. Both surfaces
// share the same per-run result through TerminalTurnBarrier.
//
// Loose typing — SDK не реэкспортирует `PluginHookBeforePromptBuildEvent`
// / `PluginHookAgentContext` / `PluginHookBeforePromptBuildResult`
// через корневой entry-point `openclaw/plugin-sdk/plugin-entry`.
// Используем структурные интерфейсы локально; runtime передаст
// объекты с правильным shape'ом.
//
// Anonymous passthrough: если из (ctx.agentId / ctx.sessionKey /
// ctx.agentDir) не удаётся вывести openclawAgentId — hook возвращает
// undefined (no-op). Симметрия с `assemble` lifecycle.

import {
  parseAgentIdFromAgentDir,
  parseAgentIdFromSessionKey,
} from "../agent-id-shared.js";
import {
  fmtErr,
  type StyxClient,
  type StyxLogger,
} from "../client.js";
import { fetchCanonicalPreturn } from "../cognition-preturn.js";
import {
  normalizeIdentifier,
  openclawHostKey,
  runOrTurnIdentity,
} from "../identifiers.js";
import { bounded, extractBoundedMessageContent } from "./agent-end.js";
import {
  TerminalTurnBarrier,
  terminalScopeKey,
} from "./terminal-barrier.js";

export type BeforePromptBuildHookEvent = {
  prompt?: string;
  messages?: unknown[];
  runId?: string;
  turnId?: string;
  // SDK может присылать другие поля — мы их игнорируем (passthrough).
  [key: string]: unknown;
};

export type BeforePromptBuildHookContext = {
  agentId?: string;
  sessionId?: string;
  sessionKey?: string;
  agentDir?: string;
  workspaceDir?: string;
  modelId?: string;
  modelProviderId?: string;
  runId?: string;
  turnId?: string;
  // ... runtime передаёт более богатый shape; нас интересуют только
  // identifiers + model.
  [key: string]: unknown;
};

export type BeforePromptBuildHookResult = {
  appendSystemContext?: string;
  prependSystemContext?: string;
  appendContext?: string;
  prependContext?: string;
};

export type BeforePromptBuildHookHandler = (
  event: BeforePromptBuildHookEvent,
  ctx: BeforePromptBuildHookContext,
) => Promise<BeforePromptBuildHookResult | undefined> | BeforePromptBuildHookResult | undefined;

export type BeforePromptBuildHookParams = {
  client: StyxClient;
  logger: StyxLogger;
  resolveAgentId: (openclawAgentId: string) => Promise<string>;
  terminalBarrier: TerminalTurnBarrier;
  terminalDrainTimeoutMs: number;
};

const MAX_PRETURN_MESSAGES = 256;
const MAX_PRETURN_CONTENT_CHARS = 4_000;

export function boundedPreturnMessages(messages: unknown[]): Array<{
  role: "system" | "user" | "assistant" | "tool";
  content: string;
  name?: string;
  tool_call_id?: string;
}> {
  const out: Array<{
    role: "system" | "user" | "assistant" | "tool";
    content: string;
    name?: string;
    tool_call_id?: string;
  }> = [];
  for (const raw of messages.slice(-MAX_PRETURN_MESSAGES)) {
    if (!raw || typeof raw !== "object") continue;
    const message = raw as Record<string, unknown>;
    const role = message["role"];
    if (role !== "system" && role !== "user" && role !== "assistant" && role !== "tool") {
      continue;
    }
    const content = extractBoundedMessageContent(
      message["content"], MAX_PRETURN_CONTENT_CHARS,
    );
    const name = bounded(message["name"], 256);
    const toolCallId = bounded(
      message["tool_call_id"] ?? message["toolCallId"], 256,
    );
    out.push({
      role,
      content,
      ...(name ? { name } : {}),
      ...(toolCallId ? { tool_call_id: toolCallId } : {}),
    });
  }
  return out;
}

/**
 * Извлечь openclawAgentId из hook ctx. Симметрия с
 * `deriveOpenclawAgentId` в context-engine.ts, но ctx hook'а имеет
 * шире shape — `agentId`/`sessionKey`/`agentDir` приходят как
 * top-level поля, не через `params`.
 */
function deriveOpenclawAgentIdFromHookCtx(
  ctx: BeforePromptBuildHookContext,
): string | null {
  // 1. Если ctx.agentId есть напрямую — используем его. Это
  //    OpenClaw scope key (lowercased, "main" / "agent-a" / etc.),
  //    тот же что в lifecycle params.sessionKey "agent:<id>:...".
  const fromAgentId = typeof ctx.agentId === "string" ? ctx.agentId.trim() : "";
  if (fromAgentId) return fromAgentId;

  // 2. Парсинг sessionKey ("agent:<id>:session:<sid>"). Тот же regex
  //    что в lifecycle path — `AGENT_SESSION_KEY_RE`.
  const fromSession = parseAgentIdFromSessionKey(
    typeof ctx.sessionKey === "string" ? ctx.sessionKey : "",
  );
  if (fromSession) return fromSession;

  // 3. agentDir fallback (legacy form `<stateDir>/agents/<id>[/agent]`).
  return parseAgentIdFromAgentDir(
    typeof ctx.agentDir === "string" ? ctx.agentDir : "",
  );
}

/**
 * Factory для hook handler'а. Возвращает async function с замкнутым
 * `client` / `logger` / `resolveAgentId` + локальным `bootstrapped`
 * Set для idempotent bootstrap'а.
 *
 * Bootstrap state локальный — НЕ shared с ContextEngine. Это
 * сознательно: pi-embedded может никогда не звать engine методы
 * (см. audit findings — assemble lifecycle игнорируется этим
 * runner'ом). Hook должен работать standalone. Дублирование state =
 * один лишний `/context/bootstrap` на сессию (idempotent на core
 * стороне — registry уже initialized).
 */
export function createBeforePromptBuildHook(
  params: BeforePromptBuildHookParams,
): BeforePromptBuildHookHandler {
  const {
    client,
    logger,
    resolveAgentId,
    terminalBarrier,
    terminalDrainTimeoutMs,
  } = params;
  const bootstrapped = new Set<string>();

  return async function beforePromptBuild(event, ctx) {
    const openclawAgentId = deriveOpenclawAgentIdFromHookCtx(ctx);
    if (openclawAgentId === null) {
      // Anonymous поток — нет привязанного Styx-агента. Passthrough,
      // ничего не добавляем.
      return undefined;
    }

    const sessionId = normalizeIdentifier(ctx.sessionId, 256);
    const terminalScope = terminalScopeKey(
      normalizeIdentifier(openclawAgentId, 256),
      sessionId,
      normalizeIdentifier(ctx.sessionKey, 256),
    );
    // Establish the causal hand-off before fetching the next fenced snapshot.
    // A timeout is fail-open but the predecessor remains tracked and its host
    // identity is disclosed to core; late work cannot be silently reordered.
    const drain = await terminalBarrier.drain(
      terminalScope,
      terminalDrainTimeoutMs,
    );
    if (!drain.completed) {
      logger.warn?.(
        `[styx] before_prompt_build timed out after ${terminalDrainTimeoutMs}ms waiting for ${drain.pending} previous agent_end task(s); preturn is marked predecessor-pending`,
      );
    }

    let agentId: string;
    try {
      agentId = normalizeIdentifier(await resolveAgentId(openclawAgentId), 256);
    } catch (err) {
      logger.warn?.(
        `[styx] before_prompt_build resolveAgentId(${openclawAgentId}) failed: ${fmtErr(err)}`,
      );
      return undefined;
    }

    const bootstrapKey = `${agentId}::${sessionId}`;

    if (!bootstrapped.has(bootstrapKey)) {
      try {
        await client.contextBootstrap({
          agent_id: agentId,
          session_id: sessionId || null,
        });
        bootstrapped.add(bootstrapKey);
      } catch (err) {
        logger.warn?.(
          `[styx] before_prompt_build bootstrap (${agentId}/${sessionId}) failed: ${fmtErr(err)}`,
        );
        return undefined;
      }
    }

    // Messages могут отсутствовать (legacy hook phase в deprecated
    // before_agent_start). Передаём что есть — core composer
    // справится с пустым списком (no-op путь).
    const rawMessages = Array.isArray(event.messages)
      ? (event.messages as Array<Record<string, unknown>>)
      : [];

    const messages = boundedPreturnMessages(rawMessages);
    const identity = runOrTurnIdentity(
      event.runId || ctx.runId,
      event.turnId || ctx.turnId,
    );
    const hostKey = identity ? openclawHostKey(identity) : null;
    const parentHostKey = terminalBarrier.predecessorActKey(terminalScope);
    try {
      const request = {
        agent_id: agentId,
        ...(hostKey ? { host_key: hostKey } : {}),
        ...(parentHostKey ? { parent_host_key: parentHostKey } : {}),
        session_id: sessionId || null,
        messages,
        query: bounded(event.prompt, 20_000) || null,
        token_budget: null,
        model: normalizeIdentifier(ctx.modelId, 512) || null,
        platform: "openclaw",
        extra: {
          current_event: {
            hook: "before_prompt_build",
            run_id: normalizeIdentifier(event.runId || ctx.runId, 256) || null,
            model_provider: normalizeIdentifier(ctx.modelProviderId, 128) || null,
            predecessor_pending: !drain.completed,
            pending_predecessor_host_keys: drain.pendingIdentities
              .slice(0, 2).map((item) => bounded(item, 128)),
          },
        },
      };
      const resp = await terminalBarrier.getOrCreatePreturn(
        terminalScope,
        identity,
        () => fetchCanonicalPreturn(client, request),
      );
      terminalBarrier.rememberSnapshot(
        terminalScope,
        resp.legacy ? null : (resp.snapshot_token || null),
        identity,
      );
      if (resp.system_prompt_addition) {
        // Возвращаем salient через appendSystemContext — это поле
        // которое pi-embedded-runner буквально склеивает с
        // baseSystemPrompt (через joinPresentTextSegments в
        // attempt.thread-helpers.ts).
        return { appendSystemContext: resp.system_prompt_addition };
      }
      return undefined;
    } catch (err) {
      logger.warn?.(
        `[styx] before_prompt_build canonical preturn failed: ${fmtErr(err)}`,
      );
      return undefined;
    }
  };
}
