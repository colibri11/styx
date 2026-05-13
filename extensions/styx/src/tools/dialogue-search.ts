// styx_dialogue_search — hybrid либо pure-vector поиск по дневнику.

import type { AnyAgentTool } from "openclaw/plugin-sdk/plugin-entry";

import type {
  StyxToolExecuteParams,
  StyxToolFactoryParams,
} from "./factory-types.js";

const DialogueSearchParametersSchema = {
  type: "object",
  properties: {
    query: {
      type: "string",
      minLength: 1,
      maxLength: 2000,
      description: "Поисковый запрос.",
    },
    session_id: {
      type: "string",
      description: "Опц. UUID — ограничить одной сессией.",
    },
    after: {
      type: "string",
      format: "date-time",
      description: "ISO-8601 нижняя граница (опц.).",
    },
    before: {
      type: "string",
      format: "date-time",
      description: "ISO-8601 верхняя граница (опц.).",
    },
    semantic_only: {
      type: "boolean",
      description:
        "Если true — pure cosine similarity без BM25 (default false → hybrid).",
      default: false,
    },
    limit: {
      type: "integer",
      minimum: 1,
      maximum: 50,
      description: "Максимум реплик (default 10).",
    },
  },
  required: ["query"],
  additionalProperties: false,
} as const;

let implPromise:
  | Promise<typeof import("./dialogue-search-impl.js")>
  | undefined;
function loadImpl() {
  implPromise ??= import("./dialogue-search-impl.js");
  return implPromise;
}

export function createDialogueSearchTool(
  params: StyxToolFactoryParams,
): AnyAgentTool | null {
  return {
    label: "Styx Dialogue Search",
    name: "styx_dialogue_search",
    description:
      "Поиск в дневнике (прошлые user/assistant реплики). По умолчанию hybrid FTS+vector; pure vector если semantic_only=true. Фильтры: session_id, after/before. Cross-agent НЕТ — каждый агент видит только свой дневник. Diff с styx_search_archive scope='dialogue': здесь есть session/after/before фильтры и pure-vector mode (полезно когда keywords не матчат корпус). Не auto-injected в контекст — caller использует результаты явно.",
    parameters: DialogueSearchParametersSchema,
    execute: async (toolCallId, toolParams, signal, onUpdate) => {
      const impl = await loadImpl();
      return impl.executeDialogueSearch({
        ...params,
        toolCallId,
        toolParams: toolParams as StyxToolExecuteParams["toolParams"],
        signal,
        onUpdate,
      });
    },
  };
}
