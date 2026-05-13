// Реализация styx_graph_traverse: HTTP POST /graph/traverse.

import { jsonResult } from "openclaw/plugin-sdk/core";
import type { AgentToolResult } from "@mariozechner/pi-agent-core";

import { deriveOpenclawAgentIdFromTool } from "./agent-id.js";
import { fmtErr } from "../client.js";
import { styxDisabledResult, styxErrorResult } from "./errors.js";
import type { StyxToolExecuteParams } from "./factory-types.js";
import { styxLlmToolResult } from "./styx-result.js";
import { isUuid } from "./uuid.js";

const TOOL_NAME = "styx_graph_traverse";

type GraphTraverseInput = {
  entity_id?: string;
  entity_type?: string;
  depth?: number;
  relation_filter?: string;
  limit?: number;
};

export async function executeGraphTraverse(
  params: StyxToolExecuteParams<GraphTraverseInput>,
): Promise<AgentToolResult<unknown>> {
  const { client, toolCtx, logger, resolveAgentId, toolParams } = params;

  const openclawAgentId = deriveOpenclawAgentIdFromTool(toolCtx);
  if (openclawAgentId === null) {
    return styxDisabledResult(TOOL_NAME, "no styx agent context");
  }
  const entityId = (toolParams.entity_id ?? "").trim();
  if (!entityId) {
    return jsonResult({
      error: `[styx error] ${TOOL_NAME}: entity_id is required`,
    });
  }
  // UUID validation (Fix 4): explicit reject — caller получает осмысленный
  // ответ, а не HTTP 422 от Pydantic после round-trip'а.
  if (!isUuid(entityId)) {
    return jsonResult({
      error: `[styx error] ${TOOL_NAME}: entity_id must be a valid UUID`,
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
    const resp = await client.graphTraverse({
      agent_id: agentId,
      entity_id: entityId,
      entity_type:
        typeof toolParams.entity_type === "string"
          ? toolParams.entity_type
          : null,
      // Hardcoded TS-defaults сняты (Fix 5) — server применит Pydantic
      // defaults (GraphTraverseRequest: depth=1, limit=20).
      ...(typeof toolParams.depth === "number"
        ? { depth: toolParams.depth }
        : {}),
      relation_filter:
        typeof toolParams.relation_filter === "string"
          ? toolParams.relation_filter
          : null,
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
