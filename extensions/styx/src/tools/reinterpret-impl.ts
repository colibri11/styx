// Реализация styx_reinterpret: HTTP POST /reinterpret.

import { jsonResult } from "openclaw/plugin-sdk/core";
import type { AgentToolResult } from "@mariozechner/pi-agent-core";

import { deriveOpenclawAgentIdFromTool } from "./agent-id.js";
import { fmtErr } from "../client.js";
import { styxDisabledResult, styxErrorResult } from "./errors.js";
import type { StyxToolExecuteParams } from "./factory-types.js";

const TOOL_NAME = "styx_reinterpret";

type ReinterpretInput = {
  memory_id?: string;
  new_understanding_text?: string;
  weight?: number;
};

export async function executeReinterpret(
  params: StyxToolExecuteParams<ReinterpretInput>,
): Promise<AgentToolResult<unknown>> {
  const { client, toolCtx, logger, resolveAgentId, toolParams } = params;

  const openclawAgentId = deriveOpenclawAgentIdFromTool(toolCtx);
  if (openclawAgentId === null) {
    return styxDisabledResult(TOOL_NAME, "no styx agent context");
  }
  const memoryId = (toolParams.memory_id ?? "").trim();
  const text = (toolParams.new_understanding_text ?? "").trim();
  if (!memoryId || !text) {
    return jsonResult({
      error: `[styx error] ${TOOL_NAME}: memory_id and new_understanding_text are required`,
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
    const resp = await client.reinterpret({
      agent_id: agentId,
      memory_id: memoryId,
      new_understanding_text: text,
      weight:
        typeof toolParams.weight === "number" ? toolParams.weight : null,
    });
    return jsonResult(resp);
  } catch (err) {
    logger.warn?.(`[styx] ${TOOL_NAME}: HTTP failure: ${fmtErr(err)}`);
    return styxErrorResult(TOOL_NAME, err);
  }
}
