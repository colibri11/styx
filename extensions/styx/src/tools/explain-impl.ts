// Реализация styx_explain: маршрутизирует kind → 3 HTTP endpoints.

import { jsonResult } from "openclaw/plugin-sdk/core";
import type { AgentToolResult } from "@mariozechner/pi-agent-core";

import { deriveOpenclawAgentIdFromTool } from "./agent-id.js";
import { fmtErr } from "../client.js";
import { styxDisabledResult, styxErrorResult } from "./errors.js";
import type { StyxToolExecuteParams } from "./factory-types.js";
import { styxLlmToolResult } from "./styx-result.js";
import { isUuid } from "./uuid.js";

const TOOL_NAME = "styx_explain";

type ExplainInput = {
  kind?: "decompose" | "lifetime" | "topK";
  memory_id?: string;
  query?: string;
  top_k_limit?: number;
  min_score?: number | null;
  limit?: number;
  kinds?: string[];
  after?: string;
  before?: string;
  include_factors?: boolean;
  include_recall_history?: boolean;
  recall_history_limit?: number;
  prune_min_relevance?: number;
};

export async function executeExplain(
  params: StyxToolExecuteParams<ExplainInput>,
): Promise<AgentToolResult<unknown>> {
  const { client, toolCtx, logger, resolveAgentId, toolParams } = params;

  const openclawAgentId = deriveOpenclawAgentIdFromTool(toolCtx);
  if (openclawAgentId === null) {
    return styxDisabledResult(TOOL_NAME, "no styx agent context");
  }

  const kind = toolParams.kind;
  if (kind !== "decompose" && kind !== "lifetime" && kind !== "topK") {
    return jsonResult({
      error: `[styx error] ${TOOL_NAME}: kind must be 'decompose' | 'lifetime' | 'topK'`,
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
    if (kind === "decompose") {
      const memoryId = (toolParams.memory_id ?? "").trim();
      const query = (toolParams.query ?? "").trim();
      if (!memoryId || !query) {
        return jsonResult({
          error: `[styx error] ${TOOL_NAME}: decompose requires memory_id + query`,
        });
      }
      // UUID validation (Fix 4): memory_id для decompose — UUID-shaped.
      if (!isUuid(memoryId)) {
        return jsonResult({
          error: `[styx error] ${TOOL_NAME}: memory_id must be a valid UUID`,
        });
      }
      const resp = await client.explainDecompose({
        agent_id: agentId,
        memory_id: memoryId,
        query,
        // Hardcoded TS-default снят (Fix 5) — server применит Pydantic
        // default (ExplainDecomposeRequest.top_k_limit=10).
        ...(typeof toolParams.top_k_limit === "number"
          ? { top_k_limit: toolParams.top_k_limit }
          : {}),
        min_score:
          typeof toolParams.min_score === "number"
            ? toolParams.min_score
            : null,
      });
      return styxLlmToolResult(resp);
    }
    if (kind === "lifetime") {
      const memoryId = (toolParams.memory_id ?? "").trim();
      if (!memoryId) {
        return jsonResult({
          error: `[styx error] ${TOOL_NAME}: lifetime requires memory_id`,
        });
      }
      // UUID validation (Fix 4): memory_id для lifetime — UUID-shaped.
      if (!isUuid(memoryId)) {
        return jsonResult({
          error: `[styx error] ${TOOL_NAME}: memory_id must be a valid UUID`,
        });
      }
      const resp = await client.explainLifetime({
        agent_id: agentId,
        memory_id: memoryId,
        include_recall_history:
          toolParams.include_recall_history === undefined
            ? true
            : Boolean(toolParams.include_recall_history),
        // Hardcoded TS-default снят (Fix 5) — server применит Pydantic
        // default (ExplainLifetimeRequest.recall_history_limit=10).
        ...(typeof toolParams.recall_history_limit === "number"
          ? { recall_history_limit: toolParams.recall_history_limit }
          : {}),
        prune_min_relevance:
          typeof toolParams.prune_min_relevance === "number"
            ? toolParams.prune_min_relevance
            : null,
      });
      return styxLlmToolResult(resp);
    }
    // topK
    const query = (toolParams.query ?? "").trim();
    if (!query) {
      return jsonResult({
        error: `[styx error] ${TOOL_NAME}: topK requires query`,
      });
    }
    const resp = await client.explainTopK({
      agent_id: agentId,
      query,
      // Hardcoded TS-default снят (Fix 5) — server применит Pydantic
      // default (ExplainTopKRequest.limit=10).
      ...(typeof toolParams.limit === "number"
        ? { limit: toolParams.limit }
        : {}),
      kinds:
        Array.isArray(toolParams.kinds) && toolParams.kinds.length > 0
          ? toolParams.kinds
          : null,
      after: typeof toolParams.after === "string" ? toolParams.after : null,
      before:
        typeof toolParams.before === "string" ? toolParams.before : null,
      min_score:
        typeof toolParams.min_score === "number"
          ? toolParams.min_score
          : null,
      include_factors:
        toolParams.include_factors === undefined
          ? true
          : Boolean(toolParams.include_factors),
    });
    return styxLlmToolResult(resp);
  } catch (err) {
    logger.warn?.(`[styx] ${TOOL_NAME}: HTTP failure: ${fmtErr(err)}`);
    return styxErrorResult(TOOL_NAME, err);
  }
}
