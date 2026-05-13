// styx_store — explicit subjective write через selective gatekeeper.
//
// Symmetria с memorybox_store, но описывает это в терминах Styx Locus:
// «положить в линию `я`», а не «положить в memory». LLM должен
// различать тонкости — gatekeeper решит, что записать, что
// поглотить как merge, что заменить через supersede.

import type { AnyAgentTool } from "openclaw/plugin-sdk/plugin-entry";

import type {
  StyxToolExecuteParams,
  StyxToolFactoryParams,
} from "./factory-types.js";

const StoreParametersSchema = {
  type: "object",
  properties: {
    content: {
      type: "string",
      maxLength: 500_000,
      description:
        "Текст memory. Короткий (≤2400 символов) — пишется одной memory через gatekeeper. Длинный — routит'ся в documents+chunks с tail-memory сводкой (store-routing). На русском, в первом лице — как описание прожитого, не цитата чужого источника.",
    },
    kind: {
      type: "string",
      enum: ["fact", "episode", "decision", "concept", "note"],
      description:
        "Тип записи: fact — проверенная информация; episode — событие; decision — выбор; concept — абстрактное понятие; note — всё остальное.",
      default: "note",
    },
    metadata: {
      type: "object",
      description: "Произвольный metadata (источник, контекст, теги).",
    },
    importance_provisional: {
      type: "number",
      minimum: 0,
      maximum: 1,
      description:
        "Опциональная провизорная важность [0..1]. Без указания gatekeeper подставит default 0.5.",
    },
  },
  required: ["content"],
  additionalProperties: false,
} as const;

let implPromise: Promise<typeof import("./store-impl.js")> | undefined;
function loadImpl() {
  implPromise ??= import("./store-impl.js");
  return implPromise;
}

export function createStoreTool(
  params: StyxToolFactoryParams,
): AnyAgentTool | null {
  return {
    label: "Styx Store",
    name: "styx_store",
    description:
      "Сохранить новый фрагмент линии `я` (subjective write). Перед записью пройдёт через selective gatekeeper: новый ряд, merge с похожим существующим или supersede старого. Длинный текст автоматически уйдёт в documents+chunks с tail-memory сводкой. Используй для прожитого, осмысленного, того что входит в траекторию агента-как-личности — а не как RAG-загрузку справки. Если только хочешь занести фразу диалога, используй styx_dialogue_save (другой канал).",
    parameters: StoreParametersSchema,
    execute: async (toolCallId, toolParams, signal, onUpdate) => {
      const impl = await loadImpl();
      return impl.executeStore({
        ...params,
        toolCallId,
        toolParams: toolParams as StyxToolExecuteParams["toolParams"],
        signal,
        onUpdate,
      });
    },
  };
}
