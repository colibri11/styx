// Реализация styx_link: HTTP POST /link.

import { jsonResult } from "openclaw/plugin-sdk/core";
import type { AgentToolResult } from "@mariozechner/pi-agent-core";

import { deriveOpenclawAgentIdFromTool } from "./agent-id.js";
import { fmtErr } from "../client.js";
import { styxDisabledResult, styxErrorResult } from "./errors.js";
import type { StyxToolExecuteParams } from "./factory-types.js";
import { isUuid } from "./uuid.js";

const TOOL_NAME = "styx_link";

type LinkInput = {
  source_type?: string;
  source_id?: string;
  target_type?: string;
  target_id?: string;
  relation?: string;
  weight?: number;
  metadata?: Record<string, unknown>;
};

export async function executeLink(
  params: StyxToolExecuteParams<LinkInput>,
): Promise<AgentToolResult<unknown>> {
  const { client, toolCtx, logger, resolveAgentId, toolParams } = params;

  const openclawAgentId = deriveOpenclawAgentIdFromTool(toolCtx);
  if (openclawAgentId === null) {
    return styxDisabledResult(TOOL_NAME, "no styx agent context");
  }

  const sourceType = (toolParams.source_type ?? "").trim();
  const sourceId = (toolParams.source_id ?? "").trim();
  const targetType = (toolParams.target_type ?? "").trim();
  const targetId = (toolParams.target_id ?? "").trim();
  const relation = (toolParams.relation ?? "").trim();
  if (!sourceType || !sourceId || !targetType || !targetId || !relation) {
    return jsonResult({
      error: `[styx error] ${TOOL_NAME}: source_type/source_id/target_type/target_id/relation are required`,
    });
  }
  // UUID validation (Fix 4) — оба endpoint'а это UUID-shaped.
  if (!isUuid(sourceId)) {
    return jsonResult({
      error: `[styx error] ${TOOL_NAME}: source_id must be a valid UUID`,
    });
  }
  if (!isUuid(targetId)) {
    return jsonResult({
      error: `[styx error] ${TOOL_NAME}: target_id must be a valid UUID`,
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
    const resp = await client.link({
      agent_id: agentId,
      source_type: sourceType,
      source_id: sourceId,
      target_type: targetType,
      target_id: targetId,
      relation,
      // Hardcoded TS-default снят (Fix 5) — server применит Pydantic
      // default (LinkRequest.weight=1.0).
      ...(typeof toolParams.weight === "number"
        ? { weight: toolParams.weight }
        : {}),
      metadata: toolParams.metadata ?? {},
    });
    return jsonResult(resp);
  } catch (err) {
    logger.warn?.(`[styx] ${TOOL_NAME}: HTTP failure: ${fmtErr(err)}`);
    return styxErrorResult(TOOL_NAME, err);
  }
}
