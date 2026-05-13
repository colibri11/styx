// styx_link — manual edge insert в knowledge graph. Идемпотентен по
// UNIQUE constraint миграции 0004.

import type { AnyAgentTool } from "openclaw/plugin-sdk/plugin-entry";

import type {
  StyxToolExecuteParams,
  StyxToolFactoryParams,
} from "./factory-types.js";

const LinkParametersSchema = {
  type: "object",
  properties: {
    source_type: {
      type: "string",
      description: "Тип source ('memory' / 'document' / 'dialogue').",
    },
    source_id: { type: "string", description: "UUID source сущности." },
    target_type: {
      type: "string",
      description: "Тип target ('memory' / 'document' / 'dialogue').",
    },
    target_id: { type: "string", description: "UUID target сущности." },
    relation: {
      type: "string",
      description:
        "Тип связи (например 'related_to' / 'discussed_in' / 'mentions' / 'caused_by').",
    },
    weight: {
      type: "number",
      minimum: 0,
      description: "Вес связи (default 1.0).",
    },
    metadata: {
      type: "object",
      description: "Произвольный metadata связи (контекст, причина).",
    },
  },
  required: ["source_type", "source_id", "target_type", "target_id", "relation"],
  additionalProperties: false,
} as const;

let implPromise: Promise<typeof import("./link-impl.js")> | undefined;
function loadImpl() {
  implPromise ??= import("./link-impl.js");
  return implPromise;
}

export function createLinkTool(
  params: StyxToolFactoryParams,
): AnyAgentTool | null {
  return {
    label: "Styx Link",
    name: "styx_link",
    description:
      "Создать edge в knowledge graph вручную. Идемпотентен (UNIQUE source/target/relation): повторный вызов с теми же параметрами вернёт `{created: false}`. Применяй когда видишь явную связь между memories/documents/dialogues, которую auto-link и Hebbian co-retrieval не уловили (например, узнал что событие X стало причиной решения Y). Это расширение поверх memorybox surface — нативный для Styx KG (волна 21). Knowledge graph — shared cross-agent пространство смыслов (ADR § 33.2 / § 34.1): создаваемое ребро видно всем агентам Styx. Передаваемый agent_id определяет origin write, не visibility.",
    parameters: LinkParametersSchema,
    execute: async (toolCallId, toolParams, signal, onUpdate) => {
      const impl = await loadImpl();
      return impl.executeLink({
        ...params,
        toolCallId,
        toolParams: toolParams as StyxToolExecuteParams["toolParams"],
        signal,
        onUpdate,
      });
    },
  };
}
