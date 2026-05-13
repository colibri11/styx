// Result-helper для LLM-facing styx tools (волна 30, Phase D).
//
// Core daemon при header'е `X-Wrap-For-LLM: 1` (выставляется
// автоматически в `client.ts` для всех LLM-facing paths) добавляет в
// response поле `llm_text` — pre-rendered обёрнутую строку с
// маркером таксономии волны 30 (`<styx-{channel}>...</styx-{channel}>`).
// Plugin'у не нужно знать какой channel у tool'а — он просто берёт
// готовую строку и подаёт LLM как text content. Это даёт LLM один
// чёткий маркер «вот результат от styx_{tool}», вместо raw JSON
// dump'а который смешивается с входом собеседника.
//
// Fallback: если по какой-то причине daemon вернул response без
// `llm_text` (старая версия core, или endpoint не входит в taxonomy),
// падаем на стандартный `jsonResult(resp)` — старое поведение.

import { jsonResult } from "openclaw/plugin-sdk/core";
import type { AgentToolResult } from "@mariozechner/pi-agent-core";

type WrappableResponse = { llm_text?: string | null };

export function styxLlmToolResult<T extends WrappableResponse>(
  resp: T,
): AgentToolResult<unknown> {
  const llmText = resp.llm_text;
  if (typeof llmText === "string" && llmText.length > 0) {
    return {
      content: [{ type: "text", text: llmText }],
      details: resp,
    };
  }
  return jsonResult(resp);
}
