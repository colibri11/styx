// styx_dialogue_save — explicit ad-hoc запись одной реплики в дневник.

import type { AnyAgentTool } from "openclaw/plugin-sdk/plugin-entry";

import type {
  StyxToolExecuteParams,
  StyxToolFactoryParams,
} from "./factory-types.js";

const DialogueSaveParametersSchema = {
  type: "object",
  properties: {
    role: {
      type: "string",
      enum: ["user", "assistant"],
      description: "Кто автор реплики: user или assistant.",
    },
    content: {
      type: "string",
      minLength: 1,
      maxLength: 2400,
      description: "Текст реплики (≤2400 символов).",
    },
    session_id: {
      type: "string",
      description:
        "Опц. id сессии. Если задан — upsert_session идемпотентно; если не задан — FK NULL.",
    },
    metadata: {
      type: "object",
      description: "Произвольный metadata (channel, message_id, отправитель).",
    },
  },
  required: ["role", "content"],
  additionalProperties: false,
} as const;

let implPromise:
  | Promise<typeof import("./dialogue-save-impl.js")>
  | undefined;
function loadImpl() {
  implPromise ??= import("./dialogue-save-impl.js");
  return implPromise;
}

export function createDialogueSaveTool(
  params: StyxToolFactoryParams,
): AnyAgentTool | null {
  return {
    label: "Styx Dialogue Save",
    name: "styx_dialogue_save",
    description:
      "Записать одну реплику в дневник (отдельный поток от линии `я`). Не запускает auto-link / classifier / sentiment — pipeline-канал, не natural turn (для полного pipeline с побочными эффектами используется внутренний /sync_turn, который вызывается ContextEngine'ом автоматически). Применяй только для manual corrections или для реплик из других каналов, которые не попали в обычный turn loop. Для поиска по сохранённым репликам — styx_dialogue_search или styx_dialogue_recent.",
    parameters: DialogueSaveParametersSchema,
    execute: async (toolCallId, toolParams, signal, onUpdate) => {
      const impl = await loadImpl();
      return impl.executeDialogueSave({
        ...params,
        toolCallId,
        toolParams: toolParams as StyxToolExecuteParams["toolParams"],
        signal,
        onUpdate,
      });
    },
  };
}
