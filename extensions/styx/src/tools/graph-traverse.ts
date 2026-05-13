// styx_graph_traverse — recursive CTE traversal от entity_id.

import type { AnyAgentTool } from "openclaw/plugin-sdk/plugin-entry";

import type {
  StyxToolExecuteParams,
  StyxToolFactoryParams,
} from "./factory-types.js";

const GraphTraverseParametersSchema = {
  type: "object",
  properties: {
    entity_id: {
      type: "string",
      description: "UUID стартовой сущности.",
    },
    entity_type: {
      type: "string",
      description: "Тип сущности ('memory' / 'document' / 'dialogue').",
    },
    depth: {
      type: "integer",
      minimum: 1,
      maximum: 3,
      description: "Глубина обхода (default 1, max 3).",
    },
    relation_filter: {
      type: "string",
      description: "Идти только по связям заданного типа.",
    },
    limit: {
      type: "integer",
      minimum: 1,
      maximum: 20,
      description: "Максимум nodes (default 20).",
    },
  },
  required: ["entity_id"],
  additionalProperties: false,
} as const;

let implPromise:
  | Promise<typeof import("./graph-traverse-impl.js")>
  | undefined;
function loadImpl() {
  implPromise ??= import("./graph-traverse-impl.js");
  return implPromise;
}

export function createGraphTraverseTool(
  params: StyxToolFactoryParams,
): AnyAgentTool | null {
  return {
    label: "Styx Graph Traverse",
    name: "styx_graph_traverse",
    description:
      "Обход knowledge graph от стартовой сущности (memory / document / dialogue) рекурсивно, depth ≤ 3. Возвращает связанные nodes с типами связей и направлением (outgoing / incoming). Полезно когда нужно увидеть semantic neighborhood: к чему привязана конкретная memory, что обсуждалось рядом, какие документы переплетаются. Knowledge graph — shared cross-agent пространство смыслов (ADR § 33.2 / § 34.1): traversal видит edges/entities всех агентов Styx. Передаваемый agent_id определяет origin write, не visibility.",
    parameters: GraphTraverseParametersSchema,
    execute: async (toolCallId, toolParams, signal, onUpdate) => {
      const impl = await loadImpl();
      return impl.executeGraphTraverse({
        ...params,
        toolCallId,
        toolParams: toolParams as StyxToolExecuteParams["toolParams"],
        signal,
        onUpdate,
      });
    },
  };
}
