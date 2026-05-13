// Определение openclawAgentId по контексту tool-вызова.
//
// Tools и ContextEngine.lifecycle получают разные shapes от runtime'а:
//
// - ContextEngine: lifecycle params (sessionKey/sessionId) + factory ctx
//   (agentDir). Логика — `deriveOpenclawAgentId` в `context-engine.ts`.
// - Tools: `OpenClawPluginToolContext` имеет уже `agentId?` и
//   `sessionKey?` напрямую. agentId — первичный источник; для backward
//   compat с runtime'ами, где agentId не выставлен, fallback на парсинг
//   sessionKey.
//
// Возвращает null когда identifier не выводится — это **нормальная
// ветка** (anonymous поток без линии `я`); tool отвечает понятной
// disabled-нагрузкой через `jsonResult`, не throw.
//
// Regex'ы и парсеры — в `../agent-id-shared.ts` (один источник истины
// для context-engine + tools).

import type { OpenClawPluginToolContext } from "openclaw/plugin-sdk/plugin-entry";

import {
  AGENT_ID_RE,
  parseAgentIdFromAgentDir,
  parseAgentIdFromSessionKey,
} from "../agent-id-shared.js";

export function deriveOpenclawAgentIdFromTool(
  ctx: OpenClawPluginToolContext,
): string | null {
  const direct = ctx.agentId;
  if (direct && AGENT_ID_RE.test(direct)) {
    return direct.toLowerCase();
  }
  const fromSession = parseAgentIdFromSessionKey(ctx.sessionKey ?? "");
  if (fromSession) return fromSession;
  // agentDir fallback (на случай когда runtime передаёт только его).
  // OpenClawPluginToolContext не объявляет agentDir, но рантайм может
  // включить его в sandboxed-объект — читаем безопасно через cast.
  const ctxRecord = ctx as unknown as Record<string, unknown>;
  const agentDir = typeof ctxRecord["agentDir"] === "string"
    ? (ctxRecord["agentDir"] as string)
    : "";
  return parseAgentIdFromAgentDir(agentDir);
}
