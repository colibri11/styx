// Реализация styx_store: HTTP POST /memory_store через client.

import { jsonResult } from "openclaw/plugin-sdk/core";
import type { AgentToolResult } from "@mariozechner/pi-agent-core";

import { deriveOpenclawAgentIdFromTool } from "./agent-id.js";
import { fmtErr } from "../client.js";
import { styxDisabledResult, styxErrorResult } from "./errors.js";
import type { StyxToolExecuteParams } from "./factory-types.js";
import { coerceSessionUuid } from "./uuid.js";

const TOOL_NAME = "styx_store";

type StoreInput = {
  content?: string;
  kind?: string;
  metadata?: Record<string, unknown>;
  importance_provisional?: number;
  session_id?: string;
};

export async function executeStore(
  params: StyxToolExecuteParams<StoreInput>,
): Promise<AgentToolResult<unknown>> {
  const { client, toolCtx, logger, resolveAgentId, toolParams } = params;

  const openclawAgentId = deriveOpenclawAgentIdFromTool(toolCtx);
  if (openclawAgentId === null) {
    return styxDisabledResult(TOOL_NAME, "no styx agent context");
  }

  const content = typeof toolParams.content === "string"
    ? toolParams.content
    : "";
  if (!content.trim()) {
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

  // Current-turn tool: LLM session param > toolCtx.sessionId > null.
  // UUID-validation silent-drop (Fix 1): non-UUID sessionKey не падает.
  const sessionId =
    coerceSessionUuid(toolParams.session_id) ??
    coerceSessionUuid(toolCtx.sessionId);

  try {
    const resp = await client.memoryStore({
      agent_id: agentId,
      content,
      // Hardcoded TS-default "note" снят (Fix 5) — server применит
      // Pydantic default (MemoryStoreRequest.kind="note").
      ...(typeof toolParams.kind === "string"
        ? { kind: toolParams.kind }
        : {}),
      metadata: toolParams.metadata ?? {},
      importance_provisional:
        typeof toolParams.importance_provisional === "number"
          ? toolParams.importance_provisional
          : null,
      session_id: sessionId,
    });
    return jsonResult(resp);
  } catch (err) {
    logger.warn?.(`[styx] ${TOOL_NAME}: HTTP failure: ${fmtErr(err)}`);
    return styxErrorResult(TOOL_NAME, err);
  }
}
