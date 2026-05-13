// Универсальное преобразование тех ошибок, что могут возникнуть при
// HTTP-вызове styx-core daemon, в shape возвращаемого tool-результата.
//
// Возвращаем `jsonResult` с ключом `error` в payload'е (а не SDK
// `failedTextResult` или подобное). Memorybox plugin использует тот же
// шаблон. SDK предоставляет `failedTextResult`, но `jsonResult`-with-
// `error` работает универсально с любой LLM, которая умеет parse JSON
// — runtime считывает payload без специальной маркировки isError.
// Вместо throw — чтобы LLM получил предметное сообщение и мог
// скорректировать дальнейшие шаги, а не поломать весь turn.

import { jsonResult } from "openclaw/plugin-sdk/core";
import type { AgentToolResult } from "@mariozechner/pi-agent-core";

import { StyxHttpError } from "../client.js";

// Усечение `responseText` в HTTP error payload — иначе stack-truncated
// HTML / multi-MB body уйдёт в LLM context и сожжёт окно.
const MAX_RESPONSE_BODY_SLICE = 1000;

export function styxErrorResult(
  toolName: string,
  err: unknown,
): AgentToolResult<unknown> {
  if (err instanceof StyxHttpError) {
    return jsonResult({
      error: `[styx error] ${toolName}: HTTP ${err.status}`,
      status: err.status,
      response_text: err.responseText.slice(0, MAX_RESPONSE_BODY_SLICE),
    });
  }
  if (err instanceof Error && err.name === "AbortError") {
    return jsonResult({
      error: `[styx error] ${toolName}: timeout`,
    });
  }
  const message = err instanceof Error ? err.message : String(err);
  return jsonResult({
    error: `[styx error] ${toolName}: ${message}`,
  });
}

export function styxDisabledResult(
  toolName: string,
  reason: string,
): AgentToolResult<unknown> {
  return jsonResult({
    disabled: true,
    error: `[styx unavailable] ${toolName}: ${reason}`,
    reason,
  });
}
