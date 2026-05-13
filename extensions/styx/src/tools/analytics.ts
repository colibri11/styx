// styx_analytics — общая статистика памяти агента. GET /analytics с
// query param agent_id (это единственный GET endpoint в нашем surface).

import type { AnyAgentTool } from "openclaw/plugin-sdk/plugin-entry";

import type {
  StyxToolExecuteParams,
  StyxToolFactoryParams,
} from "./factory-types.js";

const AnalyticsParametersSchema = {
  type: "object",
  properties: {},
  additionalProperties: false,
} as const;

let implPromise: Promise<typeof import("./analytics-impl.js")> | undefined;
function loadImpl() {
  implPromise ??= import("./analytics-impl.js");
  return implPromise;
}

export function createAnalyticsTool(
  params: StyxToolFactoryParams,
): AnyAgentTool | null {
  return {
    label: "Styx Analytics",
    name: "styx_analytics",
    description:
      "Статистика памяти агента: own memories по kind, own documents+chunks, dialogue_messages count видимый под RLS, relations, размер БД, плюс pending_indexing (сколько записей ждут embedding в фоне и timestamp самой старой). Полезно для самодиагностики: «сколько у меня всего memories», «всё ли проиндексировано», «когда оператор последний раз делал ingest». Без аргументов.",
    parameters: AnalyticsParametersSchema,
    execute: async (toolCallId, toolParams, signal, onUpdate) => {
      const impl = await loadImpl();
      return impl.executeAnalytics({
        ...params,
        toolCallId,
        toolParams: toolParams as StyxToolExecuteParams["toolParams"],
        signal,
        onUpdate,
      });
    },
  };
}
