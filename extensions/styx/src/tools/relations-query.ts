// styx_relations_query — плоский фильтр-запрос по таблице relations
// (knowledge graph rib-edges).

import type { AnyAgentTool } from "openclaw/plugin-sdk/plugin-entry";

import type {
  StyxToolExecuteParams,
  StyxToolFactoryParams,
} from "./factory-types.js";

const RelationsQueryParametersSchema = {
  type: "object",
  properties: {
    source_type: {
      type: "string",
      description: "Фильтр по типу source ('memory' / 'document' / 'dialogue').",
    },
    source_id: {
      type: "string",
      description: "Найти все связи ИЗ этой сущности.",
    },
    target_type: {
      type: "string",
      description: "Фильтр по типу target.",
    },
    target_id: {
      type: "string",
      description: "Найти все связи В эту сущность.",
    },
    relation: {
      type: "string",
      description:
        "Фильтр по типу связи ('related_to' / 'discussed_in' / 'mentions' / 'co_retrieved' / ...).",
    },
    limit: {
      type: "integer",
      minimum: 1,
      maximum: 500,
      description: "Максимум rows (default 50).",
    },
  },
  additionalProperties: false,
} as const;

let implPromise:
  | Promise<typeof import("./relations-query-impl.js")>
  | undefined;
function loadImpl() {
  implPromise ??= import("./relations-query-impl.js");
  return implPromise;
}

export function createRelationsQueryTool(
  params: StyxToolFactoryParams,
): AnyAgentTool | null {
  return {
    label: "Styx Relations Query",
    name: "styx_relations_query",
    description:
      "Плоский запрос по graph relations (memories, documents, dialogues — узлы; связи между ними — рёбра). Связи создаются через styx_link, через store с related_to, или auto-link sweeper'ом + Hebbian co-retrieval reinforcement. Пример: чтобы найти, в каких диалогах обсуждалась memory — source_type='memory', source_id=<id>, target_type='dialogue'. Knowledge graph — shared cross-agent пространство смыслов (ADR § 33.2 / § 34.1): edges/entities видны всем агентам Styx. Передаваемый agent_id определяет origin write, не visibility.",
    parameters: RelationsQueryParametersSchema,
    execute: async (toolCallId, toolParams, signal, onUpdate) => {
      const impl = await loadImpl();
      return impl.executeRelationsQuery({
        ...params,
        toolCallId,
        toolParams: toolParams as StyxToolExecuteParams["toolParams"],
        signal,
        onUpdate,
      });
    },
  };
}
