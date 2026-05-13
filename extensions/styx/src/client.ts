// HTTP клиент к styx-core daemon.
//
// Тонкий fetch-wrapper. Каждый запрос идёт с per-call таймаутом через
// AbortController; если httpToken задан — добавляется
// `Authorization: Bearer <token>` (auth.py в core).
//
// Phase B методы: agentInitialize, contextBootstrap, contextIngest,
// contextIngestBatch, contextDispose. Phase C добавил contextAssemble,
// contextCompact, contextAfterTurn. Phase D расширил поверхность под
// 16 LLM tools (memory_store, recall, search_archive, reinterpret,
// ingest_experience, dialogue_*, relations/graph/link, analytics,
// explain.{decompose,lifetime,topK}, confirm_usage).
//
// Типы 1:1 матчат Pydantic models в
// packages/styx-core/src/styx/http/models.py.

export type StyxLogger = {
  debug?: (msg: string, ...args: unknown[]) => void;
  info?: (msg: string, ...args: unknown[]) => void;
  warn?: (msg: string, ...args: unknown[]) => void;
  error?: (msg: string, ...args: unknown[]) => void;
};

/**
 * Форматирует ошибку в одну строку. Источник проблемы: OpenClaw
 * `createPluginLogger` принимает только первый строковый аргумент,
 * любой второй (Error, payload) теряется → в Phase E это привело к
 * silent fail ContextEngine.ingest. Все вызовы logger.{warn,error,...}
 * в plugin code обязаны быть одноаргументными; err склеиваем сюда.
 *
 * Для Error предпочитаем `stack` (даёт причину + место), для прочего
 * — `String(err)`. Безопасно для unknown unions с try/catch.
 */
export function fmtErr(err: unknown): string {
  if (err instanceof Error) {
    return err.stack ?? `${err.name}: ${err.message}`;
  }
  return String(err);
}

export type StyxClientOptions = {
  baseUrl: string;
  httpToken?: string;
  timeoutMs?: number;
  logger?: StyxLogger;
  fetchImpl?: typeof fetch;
};

export type StyxMessage = {
  role: string;
  content?: string;
  // OpenClaw присылает дополнительные поля (tool_calls, name, ts...) —
  // core в Phase B игнорирует их (model_config extra=ignore).
};

// ── agent lifecycle ──────────────────────────────────────────────────────

export type AgentInitializeRequest = {
  agent_id: string;
  session_id?: string | null;
  agent_identity?: string | null;
  platform?: string | null;
  model?: string | null;
};

export type AgentInitializeResponse = {
  agent_id: string;
  tools: Array<{ name: string; description: string; parameters: unknown }>;
};

// ── context lifecycle ────────────────────────────────────────────────────

export type ContextBootstrapRequest = {
  agent_id: string;
  session_id?: string | null;
  parent_session_id?: string | null;
};

export type ContextBootstrapResponse = {
  ok: boolean;
  initialized: boolean;
};

export type ContextIngestRequest = {
  agent_id: string;
  session_id?: string | null;
  message: StyxMessage;
  is_heartbeat?: boolean;
};

export type ContextIngestResponse = {
  ok: boolean;
  ingested: boolean;
  memory_id: string | null;
};

export type ContextIngestBatchRequest = {
  agent_id: string;
  session_id?: string | null;
  messages: StyxMessage[];
};

export type ContextIngestBatchResponse = {
  ok: boolean;
  ingested_count: number;
};

export type ContextDisposeRequest = {
  agent_id?: string | null;
  session_id?: string | null;
};

export type ContextDisposeResponse = {
  ok: boolean;
};

export type ContextAssembleRequest = {
  agent_id: string;
  session_id?: string | null;
  messages: Array<Record<string, unknown>>;
  token_budget?: number | null;
  available_tools?: string[] | null;
  citations_mode?: string | null;
  model?: string | null;
  prompt?: string | null;
};

export type ContextAssembleResponse = {
  messages: Array<Record<string, unknown>>;
  estimated_tokens: number;
  system_prompt_addition?: string | null;
  prompt_authority?: "assembled" | "preassembly_may_overflow" | null;
};

export type ContextCompactRequest = {
  agent_id: string;
  session_id?: string | null;
  force?: boolean;
};

export type ContextCompactResponse = {
  ok: boolean;
  compacted: boolean;
  reason?: string | null;
  session_id?: string | null;
  session_file?: string | null;
};

export type ContextAfterTurnRequest = {
  agent_id: string;
  session_id?: string | null;
  messages: Array<Record<string, unknown>>;
};

export type ContextAfterTurnResponse = {
  ok: boolean;
};

// ── memory_store (волна 17) ──────────────────────────────────────────────

export type MemoryStoreRequest = {
  agent_id: string;
  content: string;
  kind?: string;
  kind_src?: string;
  session_id?: string | null;
  metadata?: Record<string, unknown>;
  importance_provisional?: number | null;
};

export type MemoryStoreResponse = {
  action: string;
  memory_id?: string | null;
  existing_id?: string | null;
  similarity?: number | null;
  routed: boolean;
  document_id?: string | null;
  chunks_count?: number | null;
};

// ── recall (волна 7) ─────────────────────────────────────────────────────

export type RecallRequest = {
  agent_id: string;
  query: string;
  limit?: number | null;
  min_score?: number | null;
  session_id?: string | null;
};

export type RecallMemory = {
  id: string;
  content: string;
  score: number;
  role: string;
  created_at?: string | null;
};

export type RecallResponse = {
  memories: RecallMemory[];
  queried_count: number;
  internal_duplicates_removed: number;
  elapsed_ms: number;
  // Волна 30: pre-rendered обёрнутая строка с маркером
  // `<styx-recall>...</styx-recall>` если caller установил
  // `?wrap_for_llm=1` или `X-Wrap-For-LLM: 1`. Plugin клиент
  // устанавливает header автоматически для всех LLM-facing routes
  // (см. createStyxClient ниже).
  llm_text?: string | null;
};

// ── search_archive (волна 20) ────────────────────────────────────────────

export type SearchArchiveRequest = {
  agent_id: string;
  query: string;
  scope?: string;
  limit?: number | null;
  date_from?: string | null;
  date_to?: string | null;
  snapshot_cycle_start?: string | null;
};

export type SearchArchiveResult = {
  scope: string;
  text: string;
  snippet: string;
  score: number;
  document_id?: string | null;
  chunk_position?: number | null;
  chunk_positions?: number[] | null;
  char_start?: number | null;
  char_end?: number | null;
  memory_id?: string | null;
  role?: string | null;
  created_at?: string | null;
};

export type SearchArchiveResponse = {
  results: SearchArchiveResult[];
  total_matched: number;
  llm_text?: string | null;
};

// ── reinterpret (волна 22) ───────────────────────────────────────────────

export type ReinterpretRequest = {
  agent_id: string;
  memory_id: string;
  new_understanding_text: string;
  weight?: number | null;
};

export type ReinterpretResponse = {
  status: string;
  memory_id?: string | null;
  task_id?: string | null;
  application_id?: number | null;
  message?: string | null;
  next_available_at?: string | null;
  last_reinterpreted_at?: string | null;
  pending_application_id?: number | null;
};

// ── ingest_experience (волна 23) ─────────────────────────────────────────

export type IngestExperienceRequest = {
  agent_id: string;
  content: string;
  kind?: string;
  kind_src?: string;
  metadata?: Record<string, unknown>;
  importance_provisional?: number | null;
  content_hash?: string | null;
  pipeline_id?: string | null;
  pipeline_version?: string | null;
  content_ref?: Record<string, unknown> | null;
};

export type IngestExperienceResponse = {
  memory_id: string;
  deduplicated: boolean;
  content_hash?: string | null;
};

// ── ingest_document (волна 28) ───────────────────────────────────────────

export type IngestDocumentRequest = {
  agent_id: string;
  path: string;
  source_ref?: string | null;
  visibility?: string | null;
  metadata?: Record<string, unknown>;
  content_hash?: string | null;
};

export type IngestDocumentResponse = {
  document_id: string;
  deduplicated: boolean;
  chunks_count: number;
  mime_type: string;
  original_name: string;
  size_bytes: number;
  char_count: number;
  content_hash: string;
};

// ── dialogue tools (волна 24) ────────────────────────────────────────────

export type DialogueSaveRequest = {
  agent_id: string;
  role: "user" | "assistant";
  content: string;
  session_id?: string | null;
  metadata?: Record<string, unknown>;
};

export type DialogueSaveResponse = {
  memory_id: string;
  llm_text?: string | null;
};

export type DialogueSearchRequest = {
  agent_id: string;
  query: string;
  session_id?: string | null;
  after?: string | null;
  before?: string | null;
  semantic_only?: boolean;
  // Optional: пропуск (undefined) → server использует Pydantic default.
  // Не отправлять `null` — Pydantic field `int` не принимает None.
  limit?: number;
};

export type DialogueSearchHit = {
  memory_id: string;
  role: string;
  content: string;
  score: number;
  created_at: string;
  session_id?: string | null;
};

export type DialogueSearchResponse = {
  results: DialogueSearchHit[];
  llm_text?: string | null;
};

export type DialogueRecentRequest = {
  agent_id: string;
  session_id?: string | null;
  before?: string | null;
  // Optional: пропуск (undefined) → server использует Pydantic default.
  // Не отправлять `null` — Pydantic field `int` не принимает None.
  limit?: number;
};

export type DialogueRecentRow = {
  memory_id: string;
  role: string;
  content: string;
  created_at: string;
  session_id?: string | null;
};

export type DialogueRecentResponse = {
  rows: DialogueRecentRow[];
  llm_text?: string | null;
};

export type DialogueSessionsRequest = {
  agent_id: string;
  limit?: number;
};

export type DialogueSessionInfo = {
  session_id: string;
  message_count: number;
  first_message_at: string;
  last_message_at: string;
};

export type DialogueSessionsResponse = {
  sessions: DialogueSessionInfo[];
  llm_text?: string | null;
};

export type DialoguePrepareSummaryRequest = {
  agent_id: string;
  session_id: string;
  limit?: number;
};

export type DialoguePrepareSummaryResponse = {
  session_id: string;
  message_count: number;
  first_message_at?: string | null;
  last_message_at?: string | null;
  transcript: string;
  llm_text?: string | null;
};

// ── relations / graph / link (волна 21) ──────────────────────────────────

export type RelationsQueryRequest = {
  agent_id: string;
  source_type?: string | null;
  source_id?: string | null;
  target_type?: string | null;
  target_id?: string | null;
  relation?: string | null;
  limit?: number;
};

export type RelationRow = {
  id: string;
  source_type: string;
  source_id: string;
  target_type: string;
  target_id: string;
  relation: string;
  weight: number;
  metadata: Record<string, unknown>;
  created_at?: string | null;
};

export type RelationsQueryResponse = {
  rows: RelationRow[];
  llm_text?: string | null;
};

export type GraphTraverseRequest = {
  agent_id: string;
  entity_id: string;
  entity_type?: string | null;
  depth?: number;
  relation_filter?: string | null;
  limit?: number;
};

export type GraphNode = {
  id: string;
  type: string;
  relation: string;
  direction: string;
  depth: number;
  weight: number;
  content_preview: string;
};

export type GraphTraverseResponse = {
  root?: GraphNode | null;
  nodes: GraphNode[];
  llm_text?: string | null;
};

export type LinkRequest = {
  agent_id: string;
  source_type: string;
  source_id: string;
  target_type: string;
  target_id: string;
  relation: string;
  weight?: number;
  metadata?: Record<string, unknown>;
};

export type LinkResponse = {
  created: boolean;
};

// ── analytics / explain / confirm_usage (волна 25) ───────────────────────

export type AnalyticsResponse = {
  agents: Array<Record<string, unknown>>;
  global: Record<string, unknown>;
  pending_indexing: Record<string, unknown>;
};

export type ExplainDecomposeRequest = {
  agent_id: string;
  memory_id: string;
  query: string;
  top_k_limit?: number;
  min_score?: number | null;
};

export type ExplainDecomposeResponse = {
  mode: "decompose";
  memory_id: string;
  kind: string;
  query: string;
  final_score: number;
  rank_in_result_set?: number | null;
  top_k_limit: number;
  would_be_returned: boolean;
  return_reason?: "top_k" | "top_k_with_min_score" | null;
  not_returned_because?: Record<string, unknown> | null;
  factors: Record<string, unknown>;
  computed_at: string;
  llm_text?: string | null;
};

export type ExplainLifetimeRequest = {
  agent_id: string;
  memory_id: string;
  include_recall_history?: boolean;
  recall_history_limit?: number;
  prune_min_relevance?: number | null;
};

export type ExplainLifetimeResponse = {
  mode: "lifetime";
  memory_id: string;
  content_preview: string;
  kind: string;
  agent_id: string;
  visibility: string;
  created_at: string;
  updated_at: string;
  age_days: number;
  importance: Record<string, unknown>;
  lifecycle: Record<string, unknown>;
  access: Record<string, unknown>;
  relevance: Record<string, unknown>;
  usefulness: Record<string, unknown>;
  decay: Record<string, unknown>;
  recall_history?: Array<Record<string, unknown>> | null;
  co_retrieval_links: Array<Record<string, unknown>>;
  computed_at: string;
  llm_text?: string | null;
};

export type ExplainTopKRequest = {
  agent_id: string;
  query: string;
  limit?: number;
  kinds?: string[] | null;
  after?: string | null;
  before?: string | null;
  min_score?: number | null;
  include_factors?: boolean;
};

export type ExplainTopKResponse = {
  mode: "top_k";
  query: string;
  limit: number;
  total_candidates_considered: number;
  items: Array<Record<string, unknown>>;
  computed_at: string;
  llm_text?: string | null;
};

export type ConfirmUsageRequest = {
  agent_id: string;
  memory_ids: string[];
};

export type ConfirmUsageResponse = {
  updated: number;
  requested: number;
  missing: string[];
};

// ── error / client ───────────────────────────────────────────────────────

export class StyxHttpError extends Error {
  readonly status: number;
  readonly responseText: string;

  constructor(status: number, message: string, responseText: string) {
    super(`styx http ${status}: ${message}`);
    this.name = "StyxHttpError";
    this.status = status;
    this.responseText = responseText;
  }
}

export type StyxClient = {
  // agent lifecycle
  agentInitialize: (
    body: AgentInitializeRequest,
  ) => Promise<AgentInitializeResponse>;
  // context lifecycle (Phase B/C)
  contextBootstrap: (
    body: ContextBootstrapRequest,
  ) => Promise<ContextBootstrapResponse>;
  contextIngest: (
    body: ContextIngestRequest,
  ) => Promise<ContextIngestResponse>;
  contextIngestBatch: (
    body: ContextIngestBatchRequest,
  ) => Promise<ContextIngestBatchResponse>;
  contextDispose: (
    body: ContextDisposeRequest,
  ) => Promise<ContextDisposeResponse>;
  contextAssemble: (
    body: ContextAssembleRequest,
  ) => Promise<ContextAssembleResponse>;
  contextCompact: (
    body: ContextCompactRequest,
  ) => Promise<ContextCompactResponse>;
  contextAfterTurn: (
    body: ContextAfterTurnRequest,
  ) => Promise<ContextAfterTurnResponse>;
  // tools (Phase D)
  memoryStore: (body: MemoryStoreRequest) => Promise<MemoryStoreResponse>;
  recall: (body: RecallRequest) => Promise<RecallResponse>;
  searchArchive: (
    body: SearchArchiveRequest,
  ) => Promise<SearchArchiveResponse>;
  reinterpret: (body: ReinterpretRequest) => Promise<ReinterpretResponse>;
  ingestExperience: (
    body: IngestExperienceRequest,
  ) => Promise<IngestExperienceResponse>;
  ingestDocument: (
    body: IngestDocumentRequest,
  ) => Promise<IngestDocumentResponse>;
  dialogueSave: (body: DialogueSaveRequest) => Promise<DialogueSaveResponse>;
  dialogueSearch: (
    body: DialogueSearchRequest,
  ) => Promise<DialogueSearchResponse>;
  dialogueRecent: (
    body: DialogueRecentRequest,
  ) => Promise<DialogueRecentResponse>;
  dialogueSessions: (
    body: DialogueSessionsRequest,
  ) => Promise<DialogueSessionsResponse>;
  dialoguePrepareSummary: (
    body: DialoguePrepareSummaryRequest,
  ) => Promise<DialoguePrepareSummaryResponse>;
  relationsQuery: (
    body: RelationsQueryRequest,
  ) => Promise<RelationsQueryResponse>;
  graphTraverse: (
    body: GraphTraverseRequest,
  ) => Promise<GraphTraverseResponse>;
  link: (body: LinkRequest) => Promise<LinkResponse>;
  analytics: (params: { agent_id: string }) => Promise<AnalyticsResponse>;
  explainDecompose: (
    body: ExplainDecomposeRequest,
  ) => Promise<ExplainDecomposeResponse>;
  explainLifetime: (
    body: ExplainLifetimeRequest,
  ) => Promise<ExplainLifetimeResponse>;
  explainTopK: (body: ExplainTopKRequest) => Promise<ExplainTopKResponse>;
  confirmUsage: (
    body: ConfirmUsageRequest,
  ) => Promise<ConfirmUsageResponse>;
};

// Волна 30: paths которые core маршрутизирует под opt-in LLM wrap
// (см. http/_wrap.py). Plugin клиент устанавливает
// `X-Wrap-For-LLM: 1` для них автоматически — все эти tools
// возвращают результаты, попадающие напрямую в LLM input. Остальные
// endpoint'ы (lifecycle, observability, write-only) идут raw.
const LLM_FACING_PATHS: ReadonlySet<string> = new Set([
  "/recall",
  "/search_archive",
  "/dialogue/save",
  "/dialogue/search",
  "/dialogue/recent",
  "/dialogue/sessions",
  "/dialogue/prepare_summary",
  "/relations/query",
  "/graph/traverse",
  "/explain/decompose",
  "/explain/lifetime",
  "/explain/topK",
]);

export function createStyxClient(options: StyxClientOptions): StyxClient {
  const baseUrl = options.baseUrl.replace(/\/+$/, "");
  const timeoutMs = options.timeoutMs ?? 30_000;
  const fetchImpl = options.fetchImpl ?? globalThis.fetch.bind(globalThis);

  function buildHeaders(path: string): Record<string, string> {
    const headers: Record<string, string> = {
      "content-type": "application/json",
    };
    if (options.httpToken) {
      headers["authorization"] = `Bearer ${options.httpToken}`;
    }
    if (LLM_FACING_PATHS.has(path)) {
      headers["x-wrap-for-llm"] = "1";
    }
    return headers;
  }

  async function postCall<TResp>(path: string, body: unknown): Promise<TResp> {
    const url = `${baseUrl}${path}`;
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs);

    try {
      const resp = await fetchImpl(url, {
        method: "POST",
        headers: buildHeaders(path),
        body: JSON.stringify(body),
        signal: controller.signal,
      });
      const text = await resp.text();
      if (!resp.ok) {
        options.logger?.warn?.(
          `[styx-client] POST ${path} → ${resp.status} ${resp.statusText}`,
        );
        throw new StyxHttpError(resp.status, resp.statusText, text);
      }
      return text ? (JSON.parse(text) as TResp) : ({} as TResp);
    } finally {
      clearTimeout(timer);
    }
  }

  async function getCall<TResp>(
    path: string,
    query?: Record<string, string | number | undefined | null>,
  ): Promise<TResp> {
    const url = new URL(`${baseUrl}${path}`);
    if (query) {
      for (const [k, v] of Object.entries(query)) {
        if (v === undefined || v === null) continue;
        url.searchParams.set(k, String(v));
      }
    }
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs);

    try {
      const resp = await fetchImpl(url.toString(), {
        method: "GET",
        headers: buildHeaders(path),
        signal: controller.signal,
      });
      const text = await resp.text();
      if (!resp.ok) {
        options.logger?.warn?.(
          `[styx-client] GET ${path} → ${resp.status} ${resp.statusText}`,
        );
        throw new StyxHttpError(resp.status, resp.statusText, text);
      }
      return text ? (JSON.parse(text) as TResp) : ({} as TResp);
    } finally {
      clearTimeout(timer);
    }
  }

  return {
    // agent lifecycle
    agentInitialize: (body) => postCall("/agent/initialize", body),
    // context lifecycle
    contextBootstrap: (body) => postCall("/context/bootstrap", body),
    contextIngest: (body) => postCall("/context/ingest", body),
    contextIngestBatch: (body) => postCall("/context/ingest_batch", body),
    contextDispose: (body) => postCall("/context/dispose", body),
    contextAssemble: (body) => postCall("/context/assemble", body),
    contextCompact: (body) => postCall("/context/compact", body),
    contextAfterTurn: (body) => postCall("/context/after_turn", body),
    // tools
    memoryStore: (body) => postCall("/memory_store", body),
    recall: (body) => postCall("/recall", body),
    searchArchive: (body) => postCall("/search_archive", body),
    reinterpret: (body) => postCall("/reinterpret", body),
    ingestExperience: (body) => postCall("/ingest_experience", body),
    ingestDocument: (body) => postCall("/ingest_document", body),
    dialogueSave: (body) => postCall("/dialogue/save", body),
    dialogueSearch: (body) => postCall("/dialogue/search", body),
    dialogueRecent: (body) => postCall("/dialogue/recent", body),
    dialogueSessions: (body) => postCall("/dialogue/sessions", body),
    dialoguePrepareSummary: (body) =>
      postCall("/dialogue/prepare_summary", body),
    relationsQuery: (body) => postCall("/relations/query", body),
    graphTraverse: (body) => postCall("/graph/traverse", body),
    link: (body) => postCall("/link", body),
    analytics: (params) =>
      getCall("/analytics", { agent_id: params.agent_id }),
    explainDecompose: (body) => postCall("/explain/decompose", body),
    explainLifetime: (body) => postCall("/explain/lifetime", body),
    explainTopK: (body) => postCall("/explain/topK", body),
    confirmUsage: (body) => postCall("/confirm_usage", body),
  };
}
