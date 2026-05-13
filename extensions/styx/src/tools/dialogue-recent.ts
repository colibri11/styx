// styx_dialogue_recent — pure chronological retrieval из дневника.

import type { AnyAgentTool } from "openclaw/plugin-sdk/plugin-entry";

import type {
  StyxToolExecuteParams,
  StyxToolFactoryParams,
} from "./factory-types.js";

const DialogueRecentParametersSchema = {
  type: "object",
  properties: {
    session_id: {
      type: "string",
      description: "Опц. UUID — ограничить одной сессией.",
    },
    before: {
      type: "string",
      format: "date-time",
      description: "Опц. ISO-8601 cutoff — исключить реплики после метки.",
    },
    limit: {
      type: "integer",
      minimum: 1,
      maximum: 200,
      description: "Максимум реплик (default 20).",
    },
  },
  additionalProperties: false,
} as const;

let implPromise:
  | Promise<typeof import("./dialogue-recent-impl.js")>
  | undefined;
function loadImpl() {
  implPromise ??= import("./dialogue-recent-impl.js");
  return implPromise;
}

export function createDialogueRecentTool(
  params: StyxToolFactoryParams,
): AnyAgentTool | null {
  return {
    label: "Styx Dialogue Recent",
    name: "styx_dialogue_recent",
    description:
      "Хронологический срез последних реплик дневника (oldest-first). Без semantic search — pure ordering по времени. Применяй в начале новой сессии чтобы прочитать как закончилась предыдущая. Реплики role tool/system/summary не попадают (только user/assistant). Фильтр по session_id (одна сессия) или before (cutoff timestamp).",
    parameters: DialogueRecentParametersSchema,
    execute: async (toolCallId, toolParams, signal, onUpdate) => {
      const impl = await loadImpl();
      return impl.executeDialogueRecent({
        ...params,
        toolCallId,
        toolParams: toolParams as StyxToolExecuteParams["toolParams"],
        signal,
        onUpdate,
      });
    },
  };
}
