// styx_explain — единый tool, склеивающий 3 HTTP routes
// (/explain/decompose, /explain/lifetime, /explain/topK) под discriminated
// param `kind`. Симметрия с memorybox_explain.

import type { AnyAgentTool } from "openclaw/plugin-sdk/plugin-entry";

import type {
  StyxToolExecuteParams,
  StyxToolFactoryParams,
} from "./factory-types.js";

const ExplainParametersSchema = {
  type: "object",
  properties: {
    kind: {
      type: "string",
      enum: ["decompose", "lifetime", "topK"],
      description:
        "Mode: decompose — 11-факторный breakdown скоринга для (memory_id, query); lifetime — full history конкретной memory; topK — ranked top-K по query с per-item factor decomposition.",
    },
    memory_id: {
      type: "string",
      description: "UUID memory (decompose / lifetime).",
    },
    query: {
      type: "string",
      description: "Query string (decompose / topK).",
    },
    top_k_limit: {
      type: "integer",
      minimum: 1,
      maximum: 200,
      description:
        "decompose: top-K cutoff для outside_top_k diagnostic (default 10).",
    },
    min_score: {
      type: "number",
      description:
        "decompose / topK: опц. порог composite score. null отключает проверку.",
    },
    limit: {
      type: "integer",
      minimum: 1,
      maximum: 50,
      description: "topK: limit (default 10).",
    },
    kinds: {
      type: "array",
      items: { type: "string" },
      description: "topK: фильтр по kinds.",
    },
    after: {
      type: "string",
      format: "date-time",
      description: "topK: ISO-8601 нижняя граница.",
    },
    before: {
      type: "string",
      format: "date-time",
      description: "topK: ISO-8601 верхняя граница.",
    },
    include_factors: {
      type: "boolean",
      description: "topK: include factor blocks per item (default true).",
    },
    include_recall_history: {
      type: "boolean",
      description: "lifetime: include recall_history (default true).",
    },
    recall_history_limit: {
      type: "integer",
      minimum: 1,
      maximum: 100,
      description: "lifetime: max history entries (default 10).",
    },
    prune_min_relevance: {
      type: "number",
      description:
        "lifetime: опц. порог для расчёта decay.estimated_days_to_prune_threshold.",
    },
  },
  required: ["kind"],
  additionalProperties: false,
} as const;

let implPromise: Promise<typeof import("./explain-impl.js")> | undefined;
function loadImpl() {
  implPromise ??= import("./explain-impl.js");
  return implPromise;
}

export function createExplainTool(
  params: StyxToolFactoryParams,
): AnyAgentTool | null {
  return {
    label: "Styx Explain",
    name: "styx_explain",
    description:
      "Observability tool: разбор скоринга и lifecycle. 3 режима через `kind`: decompose (memory_id + query → per-factor breakdown композитного score: relevance × recency × frequency × lifecycle × feedback × importance × diversity × decay × usage × emotional_resonance × baseMatch); lifetime (memory_id → full history: importance, lifecycle transitions, recall_history, decay projections, co-retrieval links); topK (query → ranked top-K с factor decomposition каждого). Применяй для самопроверки «почему этот recall именно так раcположен» или «почему эта memory не возвращается». Под RLS scope caller'а.",
    parameters: ExplainParametersSchema,
    execute: async (toolCallId, toolParams, signal, onUpdate) => {
      const impl = await loadImpl();
      return impl.executeExplain({
        ...params,
        toolCallId,
        toolParams: toolParams as StyxToolExecuteParams["toolParams"],
        signal,
        onUpdate,
      });
    },
  };
}
