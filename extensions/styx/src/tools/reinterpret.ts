// styx_reinterpret — переосмысление существующей memory через blend
// embeddings (IAmBook §V «Переосмысление через взвешенное усреднение»).

import type { AnyAgentTool } from "openclaw/plugin-sdk/plugin-entry";

import type {
  StyxToolExecuteParams,
  StyxToolFactoryParams,
} from "./factory-types.js";

const ReinterpretParametersSchema = {
  type: "object",
  properties: {
    memory_id: {
      type: "string",
      pattern:
        "^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$",
      description: "UUID memory которую переосмысляешь.",
    },
    new_understanding_text: {
      type: "string",
      minLength: 1,
      maxLength: 2400,
      description:
        "Что добавилось в понимании. 1–3 предложения, на русском, в первом лице если оригинал в первом лице. LLM-handler в core склеит prev+new в merged_text (не как дополнение через запятую).",
    },
    weight: {
      type: "number",
      minimum: 0,
      maximum: 1,
      description:
        "Опц. вес нового понимания при blend embedding'а (default 0.5). 0.5 — равноправный микс; больше — новое сильнее тянет recall к себе.",
    },
  },
  required: ["memory_id", "new_understanding_text"],
  additionalProperties: false,
} as const;

let implPromise:
  | Promise<typeof import("./reinterpret-impl.js")>
  | undefined;
function loadImpl() {
  implPromise ??= import("./reinterpret-impl.js");
  return implPromise;
}

export function createReinterpretTool(
  params: StyxToolFactoryParams,
): AnyAgentTool | null {
  return {
    label: "Styx Reinterpret",
    name: "styx_reinterpret",
    description:
      "Переосмыслить существующую memory: добавить координату смысла, не переписывая историю. memory_id сохраняется, граф цел. Применяется когда новое понимание встроилось в прежнее. НЕ для исправления опечаток (используй styx_store + supersede через gatekeeper) и НЕ для противоречий (новое понимание идёт отдельной memory). Cooldown 24h на memory; повторный вызов в этот период вернёт status=cooldown. Apply deferred — переосмысление применится после закрытия текущего turn'а (обычно 30–90s через reinterpret_apply_sweeper).",
    parameters: ReinterpretParametersSchema,
    execute: async (toolCallId, toolParams, signal, onUpdate) => {
      const impl = await loadImpl();
      return impl.executeReinterpret({
        ...params,
        toolCallId,
        toolParams: toolParams as StyxToolExecuteParams["toolParams"],
        signal,
        onUpdate,
      });
    },
  };
}
