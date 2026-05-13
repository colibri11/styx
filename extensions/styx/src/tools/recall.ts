// styx_recall — hybrid (vector + FTS) recall в линию `я` агента.

import type { AnyAgentTool } from "openclaw/plugin-sdk/plugin-entry";

import type {
  StyxToolExecuteParams,
  StyxToolFactoryParams,
} from "./factory-types.js";

const RecallParametersSchema = {
  type: "object",
  properties: {
    query: {
      type: "string",
      description:
        "Тема, вопрос или ключевые слова. Семантика: что ищем в линии `я` — какой опыт, какое решение, какое осмысление.",
    },
    limit: {
      type: "integer",
      minimum: 1,
      maximum: 20,
      description: "Сколько memories вернуть (default 10).",
    },
    min_score: {
      type: "number",
      minimum: 0,
      maximum: 1,
      description: "Опц. порог composite score (0..1). Ниже — не возвращаем.",
    },
  },
  required: ["query"],
  additionalProperties: false,
} as const;

let implPromise: Promise<typeof import("./recall-impl.js")> | undefined;
function loadImpl() {
  implPromise ??= import("./recall-impl.js");
  return implPromise;
}

export function createRecallTool(
  params: StyxToolFactoryParams,
): AnyAgentTool | null {
  return {
    label: "Styx Recall",
    name: "styx_recall",
    description:
      "Вспомнить то что вошло в линию `я` агента (long-tier memory). Hybrid поиск (vector similarity + FTS) с composite scoring (релевантность × recency × frequency × lifecycle × importance × decay × usage × emotional_resonance). Возвращает memories ranked. Используй когда нужен контекст вне текущего окна — что было решено, понято, прожито раньше. НЕ для подгрузки внешних справочников (это не RAG); для архива (длинных документов, прошлых диалогов) — styx_search_archive.",
    parameters: RecallParametersSchema,
    execute: async (toolCallId, toolParams, signal, onUpdate) => {
      const impl = await loadImpl();
      return impl.executeRecall({
        ...params,
        toolCallId,
        toolParams: toolParams as StyxToolExecuteParams["toolParams"],
        signal,
        onUpdate,
      });
    },
  };
}
