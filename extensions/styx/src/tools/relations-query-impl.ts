// Реализация styx_relations_query: HTTP POST /relations/query.

import { jsonResult } from "openclaw/plugin-sdk/core";
import type { AgentToolResult } from "@mariozechner/pi-agent-core";

import { deriveOpenclawAgentIdFromTool } from "./agent-id.js";
import { fmtErr } from "../client.js";
import { styxDisabledResult, styxErrorResult } from "./errors.js";
import type { StyxToolExecuteParams } from "./factory-types.js";
import { styxLlmToolResult } from "./styx-result.js";

const TOOL_NAME = "styx_relations_query";

type RelationsQueryInput = {
  source_type?: string;
  source_id?: string;
  target_type?: string;
  target_id?: string;
  relation?: string;
  limit?: number;
};

export async function executeRelationsQuery(
  params: StyxToolExecuteParams<RelationsQueryInput>,
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
    const resp = await client.relationsQuery({
      agent_id: agentId,
      source_type:
        typeof toolParams.source_type === "string"
          ? toolParams.source_type
          : null,
      source_id:
        typeof toolParams.source_id === "string"
          ? toolParams.source_id
          : null,
      target_type:
        typeof toolParams.target_type === "string"
          ? toolParams.target_type
          : null,
      target_id:
        typeof toolParams.target_id === "string"
          ? toolParams.target_id
          : null,
      relation:
        typeof toolParams.relation === "string"
          ? toolParams.relation
          : null,
      // Hardcoded TS-default снят (Fix 5) — server применит Pydantic
      // default (RelationsQueryRequest.limit=50).
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
