// Перехват документов-вложений (Defect-fix A).
//
// ## Зачем
//
// OpenClaw runtime, получив непротекстовое вложение (PDF/DOCX/MD/...),
// сохраняет файл на диск (`media://inbound/<id>`) и вставляет в текст
// turn-сообщения маркер `[media attached: media://inbound/<id>]`, а
// нередко и сам текст документа inline'ом. В итоге документ-Markdown
// 15K+ символов приезжал бы в Styx как обычная turn-реплика.
//
// По концепции (IAmBook §V) документ — это артефакт; его место архив
// (`documents`+`chunks`), а не дневник (`memories`). Поэтому плагин
// перехватывает media-вложения turn'а и шлёт каждый файл
// documents-каналом через `/ingest_document` (path-mode). В сам
// turn-текст вместо маркера/содержания подставляется короткая ссылка
// — документ documents-каналом уходит, turn-каналом как текст не
// едет.
//
// ## Граница ответственности
//
// `media://inbound/<id>` резолвится в абсолютный путь файла через
// `resolveMediaBufferPath` (стабильный реэкспорт
// `openclaw/plugin-sdk/media-store`). Core читает файл path-mode'ом и
// сам парсит/валидирует (whitelist `STYX_INGEST_DOC_ROOTS` + size +
// magic bytes). Плагин не парсит содержимое — только резолвит путь и
// делает HTTP-вызов.

import { resolveMediaBufferPath } from "openclaw/plugin-sdk/media-store";

import type { StyxClient, StyxLogger } from "./client.js";

// `[media attached: media://inbound/<id>]` либо
// `[media attached 1/3: media://inbound/<id>]` — id может содержать
// расширение и original-filename-префикс (буквы/цифры/.-_), но не
// слэши/пробелы/скобки.
const MEDIA_ATTACHMENT_RE =
  /\[media attached(?:\s+\d+\/\d+)?:\s*media:\/\/inbound\/([^\]\s/\\]+)\]/gi;

export type MediaAttachmentRef = {
  /** Полный matched-фрагмент `[media attached: ...]` для вырезания. */
  marker: string;
  /** media id (часть после `media://inbound/`), напр. `<uuid>.md`. */
  mediaId: string;
};

/**
 * Найти все маркеры media-вложений в тексте сообщения.
 *
 * Дубликаты одного mediaId схлопываются (один файл — один ingest),
 * но каждый маркер сохраняется для последующего вырезания из текста.
 */
export function extractMediaAttachments(text: string): MediaAttachmentRef[] {
  if (!text || text.indexOf("media://inbound/") === -1) return [];
  const out: MediaAttachmentRef[] = [];
  MEDIA_ATTACHMENT_RE.lastIndex = 0;
  let m: RegExpExecArray | null;
  while ((m = MEDIA_ATTACHMENT_RE.exec(text)) !== null) {
    out.push({ marker: m[0], mediaId: m[1] });
  }
  return out;
}

/**
 * Убрать маркеры media-вложений из текста сообщения.
 *
 * Маркер вырезается вместе с ведущим переводом строки (OpenClaw
 * вставляет его как `\n[media attached: ...]`). Результат trim'ится
 * по краям.
 */
export function stripMediaMarkers(
  text: string,
  refs: MediaAttachmentRef[],
): string {
  let out = text;
  for (const ref of refs) {
    out = out.split("\n" + ref.marker).join("");
    out = out.split(ref.marker).join("");
  }
  return out.trim();
}

export type IngestedAttachment = {
  mediaId: string;
  documentId: string;
  originalName: string;
  deduplicated: boolean;
};

/**
 * Заингестить один media-файл documents-каналом.
 *
 * Резолвит `media://inbound/<id>` в абсолютный путь и вызывает
 * `/ingest_document`. Возвращает `null` если файл не резолвится
 * (TTL-cleanup убрал, unsafe id) или ingest упал — это не должно
 * ронять turn (документ просто не попадёт в архив, но реплика
 * сохранится в дневнике как обычно).
 */
export async function ingestMediaAttachment(
  ref: MediaAttachmentRef,
  agentId: string,
  client: StyxClient,
  logger: StyxLogger,
): Promise<IngestedAttachment | null> {
  let absPath: string;
  try {
    absPath = await resolveMediaBufferPath(ref.mediaId, "inbound");
  } catch (err) {
    logger.warn?.(
      `[styx] media attachment ${ref.mediaId} не резолвится: ${String(err)}`,
    );
    return null;
  }
  try {
    const resp = await client.ingestDocument({
      agent_id: agentId,
      path: absPath,
      source_ref: `media://inbound/${ref.mediaId}`,
      visibility: null,
      metadata: { intake: "turn_attachment" },
      content_hash: null,
    });
    return {
      mediaId: ref.mediaId,
      documentId: resp.document_id,
      originalName: resp.original_name,
      deduplicated: resp.deduplicated,
    };
  } catch (err) {
    logger.warn?.(
      `[styx] ingest_document для ${ref.mediaId} упал: ${String(err)}`,
    );
    return null;
  }
}

/**
 * Перехватить документы-вложения в turn-сообщениях.
 *
 * Для каждого сообщения с маркерами `[media attached: ...]`:
 *   1. ingest каждого файла documents-каналом;
 *   2. в content вместо маркеров — короткая ссылка на архив
 *      (`[документ в архиве: <имя> · styx://store/<id>]`).
 *
 * Так документ уходит documents-каналом, а turn-каналом едет только
 * ссылка — а не 15K-символьный текст. Сообщения без маркеров и
 * non-string content проходят без изменений.
 *
 * Возвращает новый массив сообщений (вход не мутируется).
 */
export async function interceptDocumentAttachments(
  messages: Array<Record<string, unknown>>,
  agentId: string,
  client: StyxClient,
  logger: StyxLogger,
): Promise<Array<Record<string, unknown>>> {
  const out: Array<Record<string, unknown>> = [];
  for (const msg of messages) {
    const content = msg["content"];
    if (typeof content !== "string") {
      out.push(msg);
      continue;
    }
    const refs = extractMediaAttachments(content);
    if (refs.length === 0) {
      out.push(msg);
      continue;
    }
    // Дедуп по mediaId — один файл ingest'им один раз. Результат
    // ingest'а складываем по mediaId: успешный → IngestedAttachment,
    // неуспешный → null. Это разделяет refs на два класса (см. ниже).
    const seen = new Set<string>();
    const ingestByMediaId = new Map<string, IngestedAttachment | null>();
    const archived: IngestedAttachment[] = [];
    for (const ref of refs) {
      if (seen.has(ref.mediaId)) continue;
      seen.add(ref.mediaId);
      const res = await ingestMediaAttachment(ref, agentId, client, logger);
      ingestByMediaId.set(ref.mediaId, res);
      if (res !== null) archived.push(res);
    }
    // M1: стрипаем маркеры ТОЛЬКО для успешно заархивированных refs.
    // Маркер неудавшегося вложения (ingest вернул null — файл не
    // резолвится / TTL-cleanup / /ingest_document упал) остаётся в
    // turn-тексте как есть. Так вложение не теряется: реплика уходит
    // turn-каналом с видимым маркером (и с inline-контентом документа,
    // если OpenClaw его вставил) — обычное сообщение дневника. Если
    // оно длиннее лимита, core-путь B нарежет его при записи.
    // Альтернатива — молчаливое исчезновение вложения.
    const archivedRefs = refs.filter(
      (r) => ingestByMediaId.get(r.mediaId) != null,
    );
    let newContent = stripMediaMarkers(content, archivedRefs);
    for (const a of archived) {
      const ref = `[документ в архиве: ${a.originalName} · styx://store/${a.documentId}]`;
      newContent = newContent ? `${newContent}\n${ref}` : ref;
    }
    // Если после strip'а контент пуст и ничего не заархивировано —
    // оставляем сообщение как было (защита от потери реплики).
    if (!newContent) {
      out.push(msg);
      continue;
    }
    out.push({ ...msg, content: newContent });
  }
  return out;
}
