// Реализация styx_dialogue_save: HTTP POST /dialogue/save.

import { jsonResult } from "openclaw/plugin-sdk/core";
import type { AgentToolResult } from "@mariozechner/pi-agent-core";

import { deriveOpenclawAgentIdFromTool } from "./agent-id.js";
import { fmtErr } from "../client.js";
import { styxDisabledResult, styxErrorResult } from "./errors.js";
import type { StyxToolExecuteParams } from "./factory-types.js";
import { styxLlmToolResult } from "./styx-result.js";
import { coerceSessionUuid } from "./uuid.js";

const TOOL_NAME = "styx_dialogue_save";

type DialogueSaveInput = {
  role?: "user" | "assistant";
  content?: string;
  session_id?: string;
  metadata?: Record<string, unknown>;
};

export async function executeDialogueSave(
  params: StyxToolExecuteParams<DialogueSaveInput>,
): Promise<AgentToolResult<unknown>> {
  const { client, toolCtx, logger, resolveAgentId, toolParams } = params;

  const openclawAgentId = deriveOpenclawAgentIdFromTool(toolCtx);
  if (openclawAgentId === null) {
    return styxDisabledResult(TOOL_NAME, "no styx agent context");
  }
  const role = toolParams.role;
  const content = (toolParams.content ?? "").trim();
  if (role !== "user" && role !== "assistant") {
    return jsonResult({
      error: `[styx error] ${TOOL_NAME}: role must be 'user' or 'assistant'`,
    });
  }
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

  // Current-turn tool: LLM session param > toolCtx.sessionId > null.
  // UUID-validation silent-drop (Fix 1).
  const sessionId =
    coerceSessionUuid(toolParams.session_id) ??
    coerceSessionUuid(toolCtx.sessionId);

  try {
    const resp = await client.dialogueSave({
      agent_id: agentId,
      role,
      content,
      session_id: sessionId,
      metadata: toolParams.metadata ?? {},
    });
    return styxLlmToolResult(resp);
  } catch (err) {
    logger.warn?.(`[styx] ${TOOL_NAME}: HTTP failure: ${fmtErr(err)}`);
    return styxErrorResult(TOOL_NAME, err);
  }
}
