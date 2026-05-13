// Hook before_prompt_build — доставка salient в pi-embedded runner
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
// Volna 26.7 fix через `system_prompt_addition` в
// `/context/assemble` остаётся правильным каналом для **non-embedded**
// OpenClaw harness runner (cf. `selection-*.js:7710-7714`). Эта
// hook'овая регистрация дополнительный канал для pi-embedded.
// Core HTTP endpoint один и тот же — `/context/assemble`.
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
import { fmtErr, type StyxClient, type StyxLogger } from "../client.js";

export type BeforePromptBuildHookEvent = {
  prompt?: string;
  messages?: unknown[];
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
};

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
  const { client, logger, resolveAgentId } = params;
  const bootstrapped = new Set<string>();

  return async function beforePromptBuild(event, ctx) {
    const openclawAgentId = deriveOpenclawAgentIdFromHookCtx(ctx);
    if (openclawAgentId === null) {
      // Anonymous поток — нет привязанного Styx-агента. Passthrough,
      // ничего не добавляем.
      return undefined;
    }

    let agentId: string;
    try {
      agentId = await resolveAgentId(openclawAgentId);
    } catch (err) {
      logger.warn?.(
        `[styx] before_prompt_build resolveAgentId(${openclawAgentId}) failed: ${fmtErr(err)}`,
      );
      return undefined;
    }

    const sessionId =
      typeof ctx.sessionId === "string" ? ctx.sessionId : "";
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

    try {
      const resp = await client.contextAssemble({
        agent_id: agentId,
        session_id: sessionId || null,
        messages: rawMessages,
        token_budget: null,
        available_tools: null,
        citations_mode: null,
        model: typeof ctx.modelId === "string" ? ctx.modelId : null,
        prompt: typeof event.prompt === "string" ? event.prompt : null,
      });
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
        `[styx] before_prompt_build /context/assemble failed: ${fmtErr(err)}`,
      );
      return undefined;
    }
  };
}
