// Реализация styx_ingest_experience: HTTP POST /ingest_experience.

import { jsonResult } from "openclaw/plugin-sdk/core";
import type { AgentToolResult } from "@mariozechner/pi-agent-core";

import { deriveOpenclawAgentIdFromTool } from "./agent-id.js";
import { fmtErr } from "../client.js";
import { styxDisabledResult, styxErrorResult } from "./errors.js";
import type { StyxToolExecuteParams } from "./factory-types.js";

const TOOL_NAME = "styx_ingest_experience";

type IngestExperienceInput = {
  content?: string;
  kind?: string;
  metadata?: Record<string, unknown>;
  importance_provisional?: number;
  content_hash?: string;
  pipeline_id?: string;
  pipeline_version?: string;
  content_ref?: Record<string, unknown>;
};

export async function executeIngestExperience(
  params: StyxToolExecuteParams<IngestExperienceInput>,
): Promise<AgentToolResult<unknown>> {
  const { client, toolCtx, logger, resolveAgentId, toolParams } = params;

  const openclawAgentId = deriveOpenclawAgentIdFromTool(toolCtx);
  if (openclawAgentId === null) {
    return styxDisabledResult(TOOL_NAME, "no styx agent context");
  }

  const content = (toolParams.content ?? "").trim();
  if (!content) {
    return jsonResult({
      error: `[styx error] ${TOOL_NAME}: content is required`,
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
    const resp = await client.ingestExperience({
      agent_id: agentId,
      content,
      // Hardcoded TS-default "note" снят (Fix 5) — server применит
      // Pydantic default (IngestExperienceRequest.kind="note").
      ...(typeof toolParams.kind === "string"
        ? { kind: toolParams.kind }
        : {}),
      metadata: toolParams.metadata ?? {},
      importance_provisional:
        typeof toolParams.importance_provisional === "number"
          ? toolParams.importance_provisional
          : null,
      content_hash:
        typeof toolParams.content_hash === "string"
          ? toolParams.content_hash
          : null,
      pipeline_id:
        typeof toolParams.pipeline_id === "string"
          ? toolParams.pipeline_id
          : null,
      pipeline_version:
        typeof toolParams.pipeline_version === "string"
          ? toolParams.pipeline_version
          : null,
      content_ref:
        toolParams.content_ref && typeof toolParams.content_ref === "object"
          ? toolParams.content_ref
          : null,
    });
    return jsonResult(resp);
  } catch (err) {
    logger.warn?.(`[styx] ${TOOL_NAME}: HTTP failure: ${fmtErr(err)}`);
    return styxErrorResult(TOOL_NAME, err);
  }
}
