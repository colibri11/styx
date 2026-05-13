// agent_id mapping: OpenClaw agentId (string id) → Styx agent_id (UUID).
//
// Источник правды — config.agentMapping. Поддерживаются три варианта:
//   - явный UUID: "alyona": "00000000-..." → используем напрямую.
//   - "auto" / отсутствует: динамически создаём через
//     POST /agent/initialize и кешируем in-memory на время процесса.
//
// File-backed persistence (~/.openclaw/state/styx-agent-mapping.json)
// откладывается на Phase C/D — для текущего теста на свежем агенте
// in-memory кеша достаточно (один процесс gateway = один lifetime).

import type { StyxClient } from "./client.js";

export type AgentMappingConfig = Record<string, string> | undefined;

export type ResolveAgentIdParams = {
  openclawAgentId: string;
  mapping: AgentMappingConfig;
  client: StyxClient;
};

const cache = new Map<string, string>();

export async function resolveAgentId(
  params: ResolveAgentIdParams,
): Promise<string> {
  const { openclawAgentId, mapping, client } = params;

  const fromCache = cache.get(openclawAgentId);
  if (fromCache) {
    return fromCache;
  }

  const explicit = mapping?.[openclawAgentId];
  if (explicit && explicit !== "auto") {
    cache.set(openclawAgentId, explicit);
    return explicit;
  }

  // "auto" или нет mapping'а → динамическое создание.
  // Используем openclawAgentId как наш agent_id напрямую — Styx позволяет
  // произвольную строку. Это даёт читаемые ids в БД ("main", "alyona", ...)
  // и idempotent agent/initialize: повторный вызов вернёт ту же сессию.
  const resp = await client.agentInitialize({
    agent_id: openclawAgentId,
    agent_identity: openclawAgentId,
  });
  cache.set(openclawAgentId, resp.agent_id);
  return resp.agent_id;
}

export function clearAgentMappingCache(): void {
  cache.clear();
}
