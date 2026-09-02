import {
  StyxHttpError,
  type CognitionPreturnRequest,
  type CognitionPreturnResponse,
  type StyxClient,
} from "./client.js";

export type CanonicalPreturnResult = CognitionPreturnResponse & {
  legacy: boolean;
};

/**
 * One host-side compatibility seam for pre-generation context.
 * Only a genuine 404 falls back to the legacy endpoint.
 */
export async function fetchCanonicalPreturn(
  client: StyxClient,
  request: CognitionPreturnRequest,
): Promise<CanonicalPreturnResult> {
  try {
    const response = await client.cognitionPreturn(request);
    return { ...response, legacy: false };
  } catch (error) {
    if (!(error instanceof StyxHttpError) || error.status !== 404) throw error;
    const legacy = await client.contextAssemble({
      agent_id: request.agent_id,
      session_id: request.session_id ?? null,
      messages: request.messages,
      token_budget: request.token_budget ?? null,
      available_tools: null,
      citations_mode: null,
      model: request.model ?? null,
      prompt: request.query ?? null,
    });
    return {
      messages: legacy.messages,
      line_version: 0,
      snapshot_token: "",
      will_projection: {
        formed: false,
        technical_projection: true,
        line_version: 0,
        source_count: 0,
        source_hash: "",
        supports: [],
        computation_version: "legacy",
        projection_status: "empty",
        projection_available: false,
        covered_line_version: 0,
        coverage_count: 0,
        coverage_hash: "",
        causal_root_hash: "",
        causal_root_version: 0,
        causal_frontier: [],
        root_coverage_hash: "",
        root_count: 0,
        covered_node_count: 0,
        carrier_text: "",
        carrier_version: null,
        pending_reduction_count: 0,
        reduction_failure_count: 0,
        technical_strength: 0,
        coherence: null,
        diagnostics: { legacy_fallback: true },
      },
      reconstruction: { traces: [], query_used: false, embed_available: false },
      pending_consequences: [],
      continuity_freshness: {
        fresh: true,
        predecessor_found: false,
        reduction_status: "legacy_untracked",
      },
      system_prompt_addition: legacy.system_prompt_addition ?? null,
      legacy: true,
    };
  }
}
