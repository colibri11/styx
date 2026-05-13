// Реализация styx_dialogue_prepare_summary: HTTP POST /dialogue/prepare_summary.

import { jsonResult } from "openclaw/plugin-sdk/core";
import type { AgentToolResult } from "@mariozechner/pi-agent-core";

import { deriveOpenclawAgentIdFromTool } from "./agent-id.js";
import { fmtErr } from "../client.js";
import { styxDisabledResult, styxErrorResult } from "./errors.js";
import type { StyxToolExecuteParams } from "./factory-types.js";
import { styxLlmToolResult } from "./styx-result.js";

const TOOL_NAME = "styx_dialogue_prepare_summary";

type DialoguePrepareSummaryInput = {
  session_id?: string;
  limit?: number;
};

export async function executeDialoguePrepareSummary(
  params: StyxToolExecuteParams<DialoguePrepareSummaryInput>,
): Promise<AgentToolResult<unknown>> {
  const { client, toolCtx, logger, resolveAgentId, toolParams } = params;

  const openclawAgentId = deriveOpenclawAgentIdFromTool(toolCtx);
  if (openclawAgentId === null) {
    return styxDisabledResult(TOOL_NAME, "no styx agent context");
  }
  const sessionId = (toolParams.session_id ?? "").trim();
  if (!sessionId) {
    return jsonResult({
      error: `[styx error] ${TOOL_NAME}: session_id is required`,
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
    const resp = await client.dialoguePrepareSummary({
      agent_id: agentId,
      session_id: sessionId,
      // Hardcoded TS-default снят (Fix 5) — server применит Pydantic
      // default (DialoguePrepareSummaryRequest.limit=200).
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
