// styx_ingest_document — file-ingest pipeline (волна 28). Core читает
// файл по абсолютному пути, парсит (PDF/DOCX/XLSX/MD/TXT), режет на
// chunks, embed'ит, INSERT'ит document + chunks. tail-memory НЕ
// создаётся — pull-only архив (D5). Документы доступны через
// styx_search_archive, НЕ через styx_recall.

import type { AnyAgentTool } from "openclaw/plugin-sdk/plugin-entry";

import type {
  StyxToolExecuteParams,
  StyxToolFactoryParams,
} from "./factory-types.js";

const IngestDocumentParametersSchema = {
  type: "object",
  properties: {
    path: {
      type: "string",
      minLength: 1,
      description:
        "Абсолютный путь к файлу. Поддерживаемые форматы: .pdf, .docx, .xlsx, .md, .markdown, .txt, .text. Plugin передаёт path, core читает файл сам.",
    },
    source_ref: {
      type: "string",
      maxLength: 512,
      description:
        "Опц. ссылка на источник (URL, ticket id, channel:message) для трассируемости.",
    },
    visibility: {
      type: "string",
      maxLength: 32,
      description:
        "Опц. метка видимости ('private' / 'shared'). Пока cosmetic — queries её не используют.",
    },
    metadata: {
      type: "object",
      description:
        "Опц. произвольный metadata (upload context, tags). Хранится в documents.metadata JSONB.",
    },
    content_hash: {
      type: "string",
      maxLength: 256,
      description:
        "Опц. явный hash override. Без него core вычисляет SHA256(file_bytes) сам. Partial UNIQUE на (agent_id, content_hash) даёт идемпотентность повторного ingest того же файла.",
    },
  },
  required: ["path"],
  additionalProperties: false,
} as const;

let implPromise:
  | Promise<typeof import("./ingest-document-impl.js")>
  | undefined;
function loadImpl() {
  implPromise ??= import("./ingest-document-impl.js");
  return implPromise;
}

export function createIngestDocumentTool(
  params: StyxToolFactoryParams,
): AnyAgentTool | null {
  return {
    label: "Styx Ingest Document",
    name: "styx_ingest_document",
    description:
      "File-ingest pipeline: парсит PDF/DOCX/XLSX/Markdown/text по абсолютному пути и сохраняет в архив документов. Идемпотентен по SHA256 содержимого файла: повторный вызов того же файла возвращает существующий document_id с deduplicated=true, без повторных INSERT'ов. Документы доступны через styx_search_archive (pull-only архивный поиск); они НЕ инжектятся в recall (это не subjective опыт, а внешний материал). Используй когда пользователь приложил/попросил прочитать документ. Path должен быть абсолютным; если задан STYX_INGEST_DOC_ROOTS — внутри whitelist'а. OCR не поддерживается: image-only PDF (скан без текстового слоя) вернёт 422.",
    parameters: IngestDocumentParametersSchema,
    execute: async (toolCallId, toolParams, signal, onUpdate) => {
      const impl = await loadImpl();
      return impl.executeIngestDocument({
        ...params,
        toolCallId,
        toolParams: toolParams as StyxToolExecuteParams["toolParams"],
        signal,
        onUpdate,
      });
    },
  };
}
