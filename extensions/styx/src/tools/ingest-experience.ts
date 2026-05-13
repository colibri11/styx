// styx_ingest_experience — pipeline-канал ingest. Идемпотентен по
// content_hash; короче styx_store, не проходит через gatekeeper merge/
// supersede (это другой semantic — закрытая запись, не попытка влить
// в существующий ряд).

import type { AnyAgentTool } from "openclaw/plugin-sdk/plugin-entry";

import type {
  StyxToolExecuteParams,
  StyxToolFactoryParams,
} from "./factory-types.js";

const IngestExperienceParametersSchema = {
  type: "object",
  properties: {
    content: {
      type: "string",
      minLength: 1,
      maxLength: 2400,
      description: "Текст переживания (≤2400). Длинные документы — другой канал.",
    },
    kind: {
      type: "string",
      enum: ["fact", "episode", "decision", "concept", "note"],
      description: "Тип записи (default note).",
      default: "note",
    },
    metadata: {
      type: "object",
      description: "Произвольный metadata (источник, контекст).",
    },
    importance_provisional: {
      type: "number",
      minimum: 0,
      maximum: 1,
      description: "Опц. провизорная важность [0..1].",
    },
    content_hash: {
      type: "string",
      maxLength: 256,
      description:
        "Опц. явный hash для idempotency. Если не задан, core попробует auto-compute из (pipeline_id, pipeline_version, content_ref).",
    },
    pipeline_id: {
      type: "string",
      maxLength: 64,
      description: "Идентификатор pipeline для auto-hash.",
    },
    pipeline_version: {
      type: "string",
      maxLength: 64,
      description: "Версия pipeline для auto-hash.",
    },
    content_ref: {
      type: "object",
      description: "Опционная ссылка на исходник (URL/file_id/...) для auto-hash.",
    },
  },
  required: ["content"],
  additionalProperties: false,
} as const;

let implPromise:
  | Promise<typeof import("./ingest-experience-impl.js")>
  | undefined;
function loadImpl() {
  implPromise ??= import("./ingest-experience-impl.js");
  return implPromise;
}

export function createIngestExperienceTool(
  params: StyxToolFactoryParams,
): AnyAgentTool | null {
  return {
    label: "Styx Ingest Experience",
    name: "styx_ingest_experience",
    description:
      "Pipeline ingest нового переживания. Идемпотентен по content_hash: повторный вызов с тем же payload вернёт `deduplicated=true` и старый memory_id, без побочных эффектов. Используй когда поток приносит material из внешнего канала (telegram, email, sensor, scheduled job) — не для interactive write от LLM (для этого styx_store). content ≤ 2400 chars; длинные документы — отдельный pipeline (вне волны 26). Pipeline-канал: поля `pipeline_id` / `pipeline_version` / `content_ref` предназначены для machine-driven ingestion (telegram bot, scheduled job, audio pipeline); LLM может опустить их или указать только для отладочной идемпотентности через `content_hash`. Для interactive subjective write используй styx_store.",
    parameters: IngestExperienceParametersSchema,
    execute: async (toolCallId, toolParams, signal, onUpdate) => {
      const impl = await loadImpl();
      return impl.executeIngestExperience({
        ...params,
        toolCallId,
        toolParams: toolParams as StyxToolExecuteParams["toolParams"],
        signal,
        onUpdate,
      });
    },
  };
}
