// styx_dialogue_sessions — список последних сессий с counts/timestamps.

import type { AnyAgentTool } from "openclaw/plugin-sdk/plugin-entry";

import type {
  StyxToolExecuteParams,
  StyxToolFactoryParams,
} from "./factory-types.js";

const DialogueSessionsParametersSchema = {
  type: "object",
  properties: {
    limit: {
      type: "integer",
      minimum: 1,
      maximum: 100,
      description: "Сколько последних сессий вернуть (default 10).",
    },
  },
  additionalProperties: false,
} as const;

let implPromise:
  | Promise<typeof import("./dialogue-sessions-impl.js")>
  | undefined;
function loadImpl() {
  implPromise ??= import("./dialogue-sessions-impl.js");
  return implPromise;
}

export function createDialogueSessionsTool(
  params: StyxToolFactoryParams,
): AnyAgentTool | null {
  return {
    label: "Styx Dialogue Sessions",
    name: "styx_dialogue_sessions",
    description:
      "Список последних сессий агента: session_id + message_count + first/last timestamps. Каждый агент видит только свои сессии. Применяй чтобы найти id предыдущей сессии и потом прочитать её через styx_dialogue_recent(session_id=...) или styx_dialogue_prepare_summary.",
    parameters: DialogueSessionsParametersSchema,
    execute: async (toolCallId, toolParams, signal, onUpdate) => {
      const impl = await loadImpl();
      return impl.executeDialogueSessions({
        ...params,
        toolCallId,
        toolParams: toolParams as StyxToolExecuteParams["toolParams"],
        signal,
        onUpdate,
      });
    },
  };
}
