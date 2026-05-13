// styx_search_archive — pull-канал в архив (documents+chunks, dialogue
// history). Не auto-injected в контекст, в отличие от styx_recall.

import type { AnyAgentTool } from "openclaw/plugin-sdk/plugin-entry";

import type {
  StyxToolExecuteParams,
  StyxToolFactoryParams,
} from "./factory-types.js";

const SearchArchiveParametersSchema = {
  type: "object",
  properties: {
    query: {
      type: "string",
      description: "Поисковый запрос (semantic для documents, FTS для dialogue).",
    },
    scope: {
      type: "string",
      enum: ["documents", "chunks", "dialogue", "all"],
      description:
        "documents — stitched регионы документов (subject writes >2400 + ingested файлы); chunks — отдельные chunk hits без склейки; dialogue — прошлые user/assistant реплики; all — fair-share interleave documents+dialogue.",
      default: "all",
    },
    limit: {
      type: "integer",
      minimum: 1,
      maximum: 50,
      description: "Максимум результатов (default 10).",
    },
    date_from: {
      type: "string",
      format: "date-time",
      description: "ISO-8601 нижняя граница (опц.).",
    },
    date_to: {
      type: "string",
      format: "date-time",
      description: "ISO-8601 верхняя граница (опц.).",
    },
  },
  required: ["query"],
  additionalProperties: false,
} as const;

let implPromise:
  | Promise<typeof import("./search-archive-impl.js")>
  | undefined;
function loadImpl() {
  implPromise ??= import("./search-archive-impl.js");
  return implPromise;
}

export function createSearchArchiveTool(
  params: StyxToolFactoryParams,
): AnyAgentTool | null {
  return {
    label: "Styx Search Archive",
    name: "styx_search_archive",
    description:
      "Pull-канал в архив агента: длинные документы, chunks, прошлые диалоги. FTS+vector hybrid. Результаты НЕ инжектятся автоматически в контекст — caller использует их явно. Применяй для цитат, fact-lookup, восстановления текста, выгруженного из active tier. Различие с styx_recall: recall — поиск в памяти линии `я` (что вошло в траекторию); search_archive — поиск в архивных слоях (что было записано как материал). Для diff с dialogue: scope='dialogue' даёт hybrid но без session/before/after фильтров — для них styx_dialogue_search.",
    parameters: SearchArchiveParametersSchema,
    execute: async (toolCallId, toolParams, signal, onUpdate) => {
      const impl = await loadImpl();
      return impl.executeSearchArchive({
        ...params,
        toolCallId,
        toolParams: toolParams as StyxToolExecuteParams["toolParams"],
        signal,
        onUpdate,
      });
    },
  };
}
