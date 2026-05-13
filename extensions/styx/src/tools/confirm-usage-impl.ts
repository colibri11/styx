// Реализация styx_confirm_usage: HTTP POST /confirm_usage.

import { jsonResult } from "openclaw/plugin-sdk/core";
import type { AgentToolResult } from "@mariozechner/pi-agent-core";

import { deriveOpenclawAgentIdFromTool } from "./agent-id.js";
import { fmtErr } from "../client.js";
import { styxDisabledResult, styxErrorResult } from "./errors.js";
import type { StyxToolExecuteParams } from "./factory-types.js";

const TOOL_NAME = "styx_confirm_usage";

type ConfirmUsageInput = {
  memory_ids?: string[];
};

export async function executeConfirmUsage(
  params: StyxToolExecuteParams<ConfirmUsageInput>,
): Promise<AgentToolResult<unknown>> {
  const { client, toolCtx, logger, resolveAgentId, toolParams } = params;

  const openclawAgentId = deriveOpenclawAgentIdFromTool(toolCtx);
  if (openclawAgentId === null) {
    return styxDisabledResult(TOOL_NAME, "no styx agent context");
  }
  const memoryIds = Array.isArray(toolParams.memory_ids)
    ? toolParams.memory_ids.filter((s): s is string => typeof s === "string")
    : [];
  if (memoryIds.length === 0) {
    return jsonResult({
      error: `[styx error] ${TOOL_NAME}: memory_ids must contain at least one UUID`,
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
    const resp = await client.confirmUsage({
      agent_id: agentId,
      memory_ids: memoryIds,
    });
    return jsonResult(resp);
  } catch (err) {
    logger.warn?.(`[styx] ${TOOL_NAME}: HTTP failure: ${fmtErr(err)}`);
    return styxErrorResult(TOOL_NAME, err);
  }
}
