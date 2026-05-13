// Реализация styx_dialogue_sessions: HTTP POST /dialogue/sessions.

import { jsonResult } from "openclaw/plugin-sdk/core";
import type { AgentToolResult } from "@mariozechner/pi-agent-core";

import { deriveOpenclawAgentIdFromTool } from "./agent-id.js";
import { fmtErr } from "../client.js";
import { styxDisabledResult, styxErrorResult } from "./errors.js";
import type { StyxToolExecuteParams } from "./factory-types.js";
import { styxLlmToolResult } from "./styx-result.js";

const TOOL_NAME = "styx_dialogue_sessions";

type DialogueSessionsInput = {
  limit?: number;
};

export async function executeDialogueSessions(
  params: StyxToolExecuteParams<DialogueSessionsInput>,
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
    const resp = await client.dialogueSessions({
      agent_id: agentId,
      // Hardcoded TS-default снят (Fix 5) — server применит Pydantic
      // default (DialogueSessionsRequest.limit=10).
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
