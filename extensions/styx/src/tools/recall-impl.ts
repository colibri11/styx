// Реализация styx_recall: HTTP POST /recall.

import { jsonResult } from "openclaw/plugin-sdk/core";
import type { AgentToolResult } from "@mariozechner/pi-agent-core";

import { deriveOpenclawAgentIdFromTool } from "./agent-id.js";
import { fmtErr } from "../client.js";
import { styxDisabledResult, styxErrorResult } from "./errors.js";
import type { StyxToolExecuteParams } from "./factory-types.js";
import { styxLlmToolResult } from "./styx-result.js";
import { coerceSessionUuid } from "./uuid.js";

const TOOL_NAME = "styx_recall";

type RecallInput = {
  query?: string;
  limit?: number;
  min_score?: number;
  session_id?: string;
};

export async function executeRecall(
  params: StyxToolExecuteParams<RecallInput>,
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

  // Current-turn tool: LLM session param > toolCtx.sessionId > null.
  // UUID-validation silent-drop (Fix 1).
  const sessionId =
    coerceSessionUuid(toolParams.session_id) ??
    coerceSessionUuid(toolCtx.sessionId);

  try {
    const resp = await client.recall({
      agent_id: agentId,
      query,
      limit: typeof toolParams.limit === "number" ? toolParams.limit : null,
      min_score:
        typeof toolParams.min_score === "number"
          ? toolParams.min_score
          : null,
      session_id: sessionId,
    });
    return styxLlmToolResult(resp);
  } catch (err) {
    logger.warn?.(`[styx] ${TOOL_NAME}: HTTP failure: ${fmtErr(err)}`);
    return styxErrorResult(TOOL_NAME, err);
  }
}
