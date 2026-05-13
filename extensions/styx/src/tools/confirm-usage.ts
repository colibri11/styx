// styx_confirm_usage — explicit `used_in_output=true` для recall_event'ов.

import type { AnyAgentTool } from "openclaw/plugin-sdk/plugin-entry";

import type {
  StyxToolExecuteParams,
  StyxToolFactoryParams,
} from "./factory-types.js";

const ConfirmUsageParametersSchema = {
  type: "object",
  properties: {
    memory_ids: {
      type: "array",
      items: {
        type: "string",
        pattern:
          "^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$",
      },
      minItems: 1,
      maxItems: 100,
      description:
        "UUIDs memories на которые ты опирался при формировании ответа.",
    },
  },
  required: ["memory_ids"],
  additionalProperties: false,
} as const;

let implPromise:
  | Promise<typeof import("./confirm-usage-impl.js")>
  | undefined;
function loadImpl() {
  implPromise ??= import("./confirm-usage-impl.js");
  return implPromise;
}

export function createConfirmUsageTool(
  params: StyxToolFactoryParams,
): AnyAgentTool | null {
  return {
    label: "Styx Confirm Usage",
    name: "styx_confirm_usage",
    description:
      "Пометить recall_event'ы как used_in_output=true для конкретных memories. Применяй после того как реально оперся на эти memories при формулировке ответа (не просто recall'ил, а использовал в выводе). Влияет на usage_factor в композитном scoring при будущих recall'ах. Cross-agent guard: чужие memory_ids проигнорируются и попадут в `missing` response'а, без побочных эффектов. Post-hoc usage classifier (фоновый sweeper) только заполняет gaps что ты не отметил — не перезаписывает explicit signal.",
    parameters: ConfirmUsageParametersSchema,
    execute: async (toolCallId, toolParams, signal, onUpdate) => {
      const impl = await loadImpl();
      return impl.executeConfirmUsage({
        ...params,
        toolCallId,
        toolParams: toolParams as StyxToolExecuteParams["toolParams"],
        signal,
        onUpdate,
      });
    },
  };
}
