// Реализация styx_search_archive: HTTP POST /search_archive.

import { jsonResult } from "openclaw/plugin-sdk/core";
import type { AgentToolResult } from "@mariozechner/pi-agent-core";

import { deriveOpenclawAgentIdFromTool } from "./agent-id.js";
import { fmtErr } from "../client.js";
import { styxDisabledResult, styxErrorResult } from "./errors.js";
import type { StyxToolExecuteParams } from "./factory-types.js";
import { styxLlmToolResult } from "./styx-result.js";

const TOOL_NAME = "styx_search_archive";

type SearchArchiveInput = {
  query?: string;
  scope?: string;
  limit?: number;
  date_from?: string;
  date_to?: string;
};

export async function executeSearchArchive(
  params: StyxToolExecuteParams<SearchArchiveInput>,
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
    const resp = await client.searchArchive({
      agent_id: agentId,
      query,
      // Hardcoded TS-default снят (Fix 5) — server применит Pydantic
      // default (SearchArchiveRequest.scope="all").
      ...(typeof toolParams.scope === "string"
        ? { scope: toolParams.scope }
        : {}),
      limit: typeof toolParams.limit === "number" ? toolParams.limit : null,
      date_from:
        typeof toolParams.date_from === "string"
          ? toolParams.date_from
          : null,
      date_to:
        typeof toolParams.date_to === "string" ? toolParams.date_to : null,
    });
    return styxLlmToolResult(resp);
  } catch (err) {
    logger.warn?.(`[styx] ${TOOL_NAME}: HTTP failure: ${fmtErr(err)}`);
    return styxErrorResult(TOOL_NAME, err);
  }
}
