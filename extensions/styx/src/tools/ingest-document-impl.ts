// Реализация styx_ingest_document: HTTP POST /ingest_document.

import { jsonResult } from "openclaw/plugin-sdk/core";
import type { AgentToolResult } from "@mariozechner/pi-agent-core";

import { deriveOpenclawAgentIdFromTool } from "./agent-id.js";
import { fmtErr } from "../client.js";
import { styxDisabledResult, styxErrorResult } from "./errors.js";
import type { StyxToolExecuteParams } from "./factory-types.js";

const TOOL_NAME = "styx_ingest_document";

type IngestDocumentInput = {
  path?: string;
  source_ref?: string;
  visibility?: string;
  metadata?: Record<string, unknown>;
  content_hash?: string;
};

export async function executeIngestDocument(
  params: StyxToolExecuteParams<IngestDocumentInput>,
): Promise<AgentToolResult<unknown>> {
  const { client, toolCtx, logger, resolveAgentId, toolParams } = params;

  const openclawAgentId = deriveOpenclawAgentIdFromTool(toolCtx);
  if (openclawAgentId === null) {
    return styxDisabledResult(TOOL_NAME, "no styx agent context");
  }

  const path = (toolParams.path ?? "").trim();
  if (!path) {
    return jsonResult({
      error: `[styx error] ${TOOL_NAME}: path is required`,
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
    const resp = await client.ingestDocument({
      agent_id: agentId,
      path,
      source_ref:
        typeof toolParams.source_ref === "string"
          ? toolParams.source_ref
          : null,
      visibility:
        typeof toolParams.visibility === "string"
          ? toolParams.visibility
          : null,
      metadata: toolParams.metadata ?? {},
      content_hash:
        typeof toolParams.content_hash === "string"
          ? toolParams.content_hash
          : null,
    });
    return jsonResult(resp);
  } catch (err) {
    logger.warn?.(`[styx] ${TOOL_NAME}: HTTP failure: ${fmtErr(err)}`);
    return styxErrorResult(TOOL_NAME, err);
  }
}
