// styx_dialogue_prepare_summary — формирует transcript сессии для
// summarizer-агента. Сам ничего не суммирует.

import type { AnyAgentTool } from "openclaw/plugin-sdk/plugin-entry";

import type {
  StyxToolExecuteParams,
  StyxToolFactoryParams,
} from "./factory-types.js";

const DialoguePrepareSummaryParametersSchema = {
  type: "object",
  properties: {
    session_id: {
      type: "string",
      minLength: 1,
      description: "UUID сессии для подготовки. Обязательное поле.",
    },
    limit: {
      type: "integer",
      minimum: 1,
      maximum: 1000,
      description: "Максимум реплик в transcript (default 200, max 1000).",
    },
  },
  required: ["session_id"],
  additionalProperties: false,
} as const;

let implPromise:
  | Promise<typeof import("./dialogue-prepare-summary-impl.js")>
  | undefined;
function loadImpl() {
  implPromise ??= import("./dialogue-prepare-summary-impl.js");
  return implPromise;
}

export function createDialoguePrepareSummaryTool(
  params: StyxToolFactoryParams,
): AnyAgentTool | null {
  return {
    label: "Styx Dialogue Prepare Summary",
    name: "styx_dialogue_prepare_summary",
    description:
      "Подготовить transcript одной сессии для последующего summarization. Возвращает форматированные строки `[YYYY-MM-DD HH:MM:SS] Human/Agent: content` плюс message_count и first/last timestamps. Workflow: (1) styx_dialogue_sessions для поиска session_id; (2) styx_dialogue_prepare_summary для transcript'а; (3) сам составляешь summary; (4) сохраняешь через styx_store(kind='episode', metadata={session_id, type:'session_summary'}). Tool НЕ генерирует summary — он готовит сырой материал. Пустая сессия — пустой transcript, не 404.",
    parameters: DialoguePrepareSummaryParametersSchema,
    execute: async (toolCallId, toolParams, signal, onUpdate) => {
      const impl = await loadImpl();
      return impl.executeDialoguePrepareSummary({
        ...params,
        toolCallId,
        toolParams: toolParams as StyxToolExecuteParams["toolParams"],
        signal,
        onUpdate,
      });
    },
  };
}
