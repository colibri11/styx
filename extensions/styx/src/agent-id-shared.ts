// Общие helper'ы для извлечения openclawAgentId из разных источников
// runtime'а (sessionKey / agentDir). Вынесены в shared module чтобы
// `context-engine.ts::deriveOpenclawAgentId` (lifecycle params + ctx) и
// `tools/agent-id.ts::deriveOpenclawAgentIdFromTool` (per-call tool ctx)
// использовали одну и ту же логику и одно и то же regex'ы — без drift'а.
//
// Format'ы соответствуют OpenClaw 2026.5.7 (`dist/session-key-*.js`):
//
//   sessionKey  := `agent:<agentId>:<rest>`
//   agentDir    := `${stateDir}/agents/<agentId>/agent`  (или legacy `${stateDir}/agents/<agentId>`)
//   agentId     := /^[a-z0-9][a-z0-9_-]{0,63}$/i  (VALID_ID_RE из runtime)

const AGENT_ID_RE = /^[a-z0-9][a-z0-9_-]{0,63}$/i;
export const AGENT_SESSION_KEY_RE = /^agent:([a-z0-9][a-z0-9_-]{0,63}):/i;

/**
 * Извлечь agentId из sessionKey формата `agent:<agentId>:<rest>`.
 * Возвращает lowercased agentId, либо null если sessionKey пустой /
 * не соответствует format'у (например legacy/alias).
 */
export function parseAgentIdFromSessionKey(
  sessionKey: string,
): string | null {
  if (!sessionKey) return null;
  const m = sessionKey.match(AGENT_SESSION_KEY_RE);
  if (m) return m[1].toLowerCase();
  return null;
}

/**
 * Извлечь agentId из path вида `${stateDir}/agents/<agentId>/agent` или
 * `${stateDir}/agents/<agentId>`. Возвращает lowercased agentId либо
 * null когда path не парсится (нет сегмента `agents` или candidate не
 * матчится regex'у valid id).
 */
export function parseAgentIdFromAgentDir(agentDir: string): string | null {
  if (!agentDir) return null;
  const segments = agentDir.split("/").filter((s) => s.length > 0);
  const idx = segments.lastIndexOf("agents");
  if (idx >= 0 && idx + 1 < segments.length) {
    const candidate = segments[idx + 1];
    if (candidate && AGENT_ID_RE.test(candidate)) {
      return candidate.toLowerCase();
    }
  }
  return null;
}

export { AGENT_ID_RE };
