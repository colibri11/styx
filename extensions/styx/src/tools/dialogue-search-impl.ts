// Реализация styx_dialogue_search: HTTP POST /dialogue/search.

import { jsonResult } from "openclaw/plugin-sdk/core";
import type { AgentToolResult } from "@mariozechner/pi-agent-core";

import { deriveOpenclawAgentIdFromTool } from "./agent-id.js";
import { fmtErr } from "../client.js";
import { styxDisabledResult, styxErrorResult } from "./errors.js";
import type { StyxToolExecuteParams } from "./factory-types.js";
import { styxLlmToolResult } from "./styx-result.js";

const TOOL_NAME = "styx_dialogue_search";

type DialogueSearchInput = {
  query?: string;
  session_id?: string;
  after?: string;
  before?: string;
  semantic_only?: boolean;
  limit?: number;
};

export async function executeDialogueSearch(
  params: StyxToolExecuteParams<DialogueSearchInput>,
): Promise<AgentToolResult<unknown>> {
  const { client, toolCtx, logger, resolveAgentId, toolParams } = params;

  const openclawAgentId = deriveOpenclawAgentIdFromTool(toolCtx);
  if (openclawAgentId === null) {
    return styxDisabledResult(TOOL_NAME, "no styx agent context");
  }
  const query = (toolParams.query ?? "").trim();
  if (!query) {
    return jsonResult({
      error: `[styx error] ${TOOL_NAME}: query is required`,
    });
  }

  let agentId: string;
  try {
    agentId = await resolveAgentId(openclawAgentId);
  } catch (err) {
    logger.warn?.(`[styx] ${TOOL_NAME}: resolveAgentId failed: ${fmtErr(err)}`);
    return styxErrorResult(TOOL_NAME, err);
  }

  try {
    const resp = await client.dialogueSearch({
      agent_id: agentId,
      query,
      session_id:
        typeof toolParams.session_id === "string"
          ? toolParams.session_id
          : null,
      after: typeof toolParams.after === "string" ? toolParams.after : null,
      before:
        typeof toolParams.before === "string" ? toolParams.before : null,
      semantic_only: Boolean(toolParams.semantic_only),
      // Hardcoded TS-default снят (Fix 5) — server применит default из
      // Pydantic (DialogueSearchRequest.limit=10). Conditional spread:
      // если LLM не передал — поле просто отсутствует в JSON body.
      ...(typeof toolParams.limit === "number"
        ? { limit: toolParams.limit }
        : {}),
    });
    return styxLlmToolResult(resp);
  } catch (err) {
    logger.warn?.(`[styx] ${TOOL_NAME}: HTTP failure: ${fmtErr(err)}`);
    return styxErrorResult(TOOL_NAME, err);
  }
}
