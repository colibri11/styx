// Реализация styx_dialogue_recent: HTTP POST /dialogue/recent.

import { jsonResult } from "openclaw/plugin-sdk/core";
import type { AgentToolResult } from "@mariozechner/pi-agent-core";

import { deriveOpenclawAgentIdFromTool } from "./agent-id.js";
import { fmtErr } from "../client.js";
import { styxDisabledResult, styxErrorResult } from "./errors.js";
import type { StyxToolExecuteParams } from "./factory-types.js";
import { styxLlmToolResult } from "./styx-result.js";

const TOOL_NAME = "styx_dialogue_recent";

type DialogueRecentInput = {
  session_id?: string;
  before?: string;
  limit?: number;
};

export async function executeDialogueRecent(
  params: StyxToolExecuteParams<DialogueRecentInput>,
): Promise<AgentToolResult<unknown>> {
  const { client, toolCtx, logger, resolveAgentId, toolParams } = params;

  const openclawAgentId = deriveOpenclawAgentIdFromTool(toolCtx);
  if (openclawAgentId === null) {
    return styxDisabledResult(TOOL_NAME, "no styx agent context");
  }

  let agentId: string;
  try {
    agentId = await resolveAgentId(openclawAgentId);
  } catch (err) {
    logger.warn?.(`[styx] ${TOOL_NAME}: resolveAgentId failed: ${fmtErr(err)}`);
    return styxErrorResult(TOOL_NAME, err);
  }

  try {
    const resp = await client.dialogueRecent({
      agent_id: agentId,
      session_id:
        typeof toolParams.session_id === "string"
          ? toolParams.session_id
          : null,
      before:
        typeof toolParams.before === "string" ? toolParams.before : null,
      // Hardcoded TS-default снят (Fix 5) — server применит default из
      // Pydantic (DialogueRecentRequest.limit=20). Conditional spread:
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
