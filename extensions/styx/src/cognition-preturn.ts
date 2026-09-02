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
      },
      reconstruction: { traces: [], query_used: false, embed_available: false },
      pending_consequences: [],
      system_prompt_addition: legacy.system_prompt_addition ?? null,
      legacy: true,
    };
  }
}
