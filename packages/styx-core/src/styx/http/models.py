"""Pydantic request/response models для Styx HTTP API.

Соответствует контракту в ``.design/host-agnostic-split-v1.md`` § 6.
"""

from __future__ import annotations

import datetime as _dt
from typing import Any, Literal

from pydantic import BaseModel, Field


class _LlmWrappableResponse(BaseModel):
    """Mixin для LLM-facing response моделей (волна 30, Phase B/C).

    Опциональное поле ``llm_text`` — pre-rendered обёрнутая строка с
    маркером таксономии волны 30 (``<styx-{channel}>...</styx-{channel}>``).
    Заполняется в endpoint'е только если caller установил
    ``?wrap_for_llm=1`` query-param или ``X-Wrap-For-LLM: 1`` header.
    Default — None; raw caller'ы (CLI, тесты, не-LLM consumers)
    игнорируют поле; plugin'ы (OpenClaw / Hermes) с opt-in флагом
    берут уже-готовую строку и кладут как content tool result'а
    для LLM, минуя необходимость рендера на их стороне.

    Все 11 LLM-facing response models (recall, search_archive,
    dialogue × 5, relations.{query,graph_traverse}, explain × 3)
    наследуют этот mixin. Не-LLM-facing endpoint'ы (healthz, /context/*,
    /pre_llm, /sync_turn, /confirm_usage, /agent/*, /memory_store,
    /reinterpret/*, /ingest, /relations/link, /analytics) ничего не
    наследуют — их output не попадает в LLM input как «воспоминание».
    """

    llm_text: str | None = None


# ── healthz ───────────────────────────────────────────────────────────────


class HealthzResponse(BaseModel):
    status: str
    uptime_s: float
    postgres: str
    version: str


class ReadyzResponse(BaseModel):
    status: str
    uptime_s: float
    last_drain_progress_age_s: float | None = None
    ollama: str
    queue: dict[str, int] = Field(default_factory=dict)
    version: str


# ── agent lifecycle ───────────────────────────────────────────────────────


class InitializeRequest(BaseModel):
    agent_id: str
    session_id: str | None = None
    agent_identity: str | None = None
    platform: str | None = None
    model: str | None = None
    config_overrides: dict[str, Any] = Field(default_factory=dict)


class ToolSchema(BaseModel):
    name: str
    description: str
    parameters: dict[str, Any]


class InitializeResponse(BaseModel):
    agent_id: str
    tools: list[ToolSchema] = Field(default_factory=list)


class ShutdownRequest(BaseModel):
    agent_id: str


# ── sync_turn ─────────────────────────────────────────────────────────────


class SyncTurnRequest(BaseModel):
    agent_id: str
    session_id: str | None = None
    user_content: str = ""
    assistant_content: str = ""
    tool_calls: list[dict[str, Any]] | None = None
    ts: _dt.datetime | None = None


class SyncTurnResponse(BaseModel):
    memory_ids: list[str] = Field(default_factory=list)
    recall_event_ids: list[str] = Field(default_factory=list)


# ── recall ────────────────────────────────────────────────────────────────


class RecallRequest(BaseModel):
    agent_id: str
    query: str
    limit: int | None = None
    min_score: float | None = None
    session_id: str | None = None


class RecallMemory(BaseModel):
    id: str
    content: str
    score: float
    role: str
    created_at: _dt.datetime | None = None


class RecallResponse(_LlmWrappableResponse):
    memories: list[RecallMemory] = Field(default_factory=list)
    queried_count: int
    internal_duplicates_removed: int
    elapsed_ms: int


# ── context build ─────────────────────────────────────────────────────────


class ContextBuildRequest(BaseModel):
    agent_id: str
    messages: list[dict[str, Any]]
    current_tokens: int | None = None
    focus_topic: str | None = None


class ContextBuildResponse(BaseModel):
    messages: list[dict[str, Any]]
    compression_count: int
    salient_injected: bool


# ── pre_llm_inject ────────────────────────────────────────────────────────


class PreLlmInjectRequest(BaseModel):
    agent_id: str
    session_id: str | None = None
    user_message: str | None = None
    is_first_turn: bool = False
    model: str | None = None
    platform: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class PreLlmInjectResponse(BaseModel):
    context: str | None = None


# ── agent state ───────────────────────────────────────────────────────────


class VAD(BaseModel):
    valence: float
    arousal: float
    dominance: float


class AgentStateResponse(BaseModel):
    agent_id: str
    instant: VAD | None = None
    baseline: VAD | None = None
    mood: str | None = None


# ── memory_store (волна 17) ───────────────────────────────────────────────


class MemoryStoreRequest(BaseModel):
    """Subjective write через selective gatekeeper.

    ``content`` — короткий (≤ store_routing_limit, default 2400) идёт
    через gatekeeper в memories с CHECK ≤ 2400. Длинный (> limit)
    routит'ся в documents+chunks с tail-memory ≤ summary_chars (D5
    в waves/19); upper-bound 500_000 mirror memorybox.
    ``kind`` — одно из {fact, episode, decision, concept, note};
    ``kind_src`` — одно из {subjective, subjective_tail, experience_intake,
    dialogue_consolidation_daily, dialogue_batch_consolidation}.
    """

    agent_id: str
    content: str = Field(min_length=1, max_length=500_000)
    kind: str = "note"
    kind_src: str = "subjective"
    session_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    importance_provisional: float | None = None


class MemoryStoreResponse(BaseModel):
    """Результат selective gatekeeper'а на subjective write.

    - ``action='store'``     — memory создан, ``memory_id`` populated.
    - ``action='merge'``     — поглощён существующим, ``memory_id=None``;
                               ``existing_id`` — сохранившийся ряд.
    - ``action='supersede'`` — новый создан, ``existing_id`` — старый
                               (получил ``superseded_by``).
    - ``action='skip'``      — отсечено noise filter'ом.

    Store-routing fields (волна 19, populated только когда
    ``routed=True``):

    - ``routed=True`` означает content > store_routing_limit;
      ``memory_id`` указывает на tail-memory, gatekeeper не применялся.
    - ``document_id`` — id virtual document'а в таблице documents.
    - ``chunks_count`` — сколько chunks (с embedding'ами) лежит в
      таблице chunks.
    """

    action: str
    memory_id: str | None = None
    existing_id: str | None = None
    similarity: float | None = None
    routed: bool = False
    document_id: str | None = None
    chunks_count: int | None = None


# ── relations / graph / link (волна 21) ───────────────────────────────────


class RelationsQueryRequest(BaseModel):
    """Плоский фильтр-запрос по `relations`. Cross-agent (без agent_id)."""

    agent_id: str
    source_type: str | None = None
    source_id: str | None = None
    target_type: str | None = None
    target_id: str | None = None
    relation: str | None = None
    limit: int = Field(default=50, ge=1, le=500)


class RelationRow(BaseModel):
    id: str
    source_type: str
    source_id: str
    target_type: str
    target_id: str
    relation: str
    weight: float
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: _dt.datetime | None = None


class RelationsQueryResponse(_LlmWrappableResponse):
    rows: list[RelationRow] = Field(default_factory=list)


class GraphTraverseRequest(BaseModel):
    """Recursive CTE traversal от entity_id, depth ≤ 3, limit ≤ 20."""

    agent_id: str
    entity_id: str
    entity_type: str | None = None  # 'memory' / 'document' / 'dialogue'
    depth: int = Field(default=1, ge=1, le=3)
    relation_filter: str | None = None
    limit: int = Field(default=20, ge=1, le=20)


class GraphNode(BaseModel):
    id: str
    type: str
    relation: str
    direction: str  # 'outgoing' / 'incoming'
    depth: int
    weight: float
    content_preview: str = ""


class GraphTraverseResponse(_LlmWrappableResponse):
    root: GraphNode | None = None
    nodes: list[GraphNode] = Field(default_factory=list)


class LinkRequest(BaseModel):
    """Manual edge insert. Идемпотентен через UNIQUE constraint
    (миграция 0004)."""

    agent_id: str
    source_type: str
    source_id: str
    target_type: str
    target_id: str
    relation: str
    weight: float = 1.0
    metadata: dict[str, Any] = Field(default_factory=dict)


# ── search_archive (волна 20) ─────────────────────────────────────────────


class SearchArchiveRequest(BaseModel):
    """Pull-канал к архиву. См. wave-doc 20 § «D11 Response shape»."""

    agent_id: str
    query: str
    scope: str = "all"  # validated в route (4 enum'а)
    limit: int | None = None
    date_from: _dt.datetime | None = None
    date_to: _dt.datetime | None = None
    snapshot_cycle_start: _dt.datetime | None = None


class SearchArchiveResultModel(BaseModel):
    """Heterogeneous union — `scope` discriminator. Поля, не relevant
    для данного scope, остаются None."""

    scope: str  # 'document' / 'chunk' / 'dialogue'
    text: str
    snippet: str
    score: float
    document_id: str | None = None
    chunk_position: int | None = None
    chunk_positions: list[int] | None = None
    char_start: int | None = None
    char_end: int | None = None
    memory_id: str | None = None
    role: str | None = None
    created_at: str | None = None


class SearchArchiveResponse(_LlmWrappableResponse):
    results: list[SearchArchiveResultModel] = Field(default_factory=list)
    total_matched: int = 0


class LinkResponse(BaseModel):
    created: bool


# ── reinterpret (волна 22) ───────────────────────────────────────────────


class ReinterpretRequest(BaseModel):
    """Explicit reinterpret memory с новым пониманием.

    `weight` — опц. blend weight в [0, 1]. None → default из
    `STYX_REINTERPRET_BLEND_WEIGHT` (0.5).
    """

    agent_id: str
    memory_id: str
    new_understanding_text: str = Field(min_length=1, max_length=2400)
    weight: float | None = Field(default=None, ge=0.0, le=1.0)


class ReinterpretResponse(BaseModel):
    """Heterogeneous union по `status`. Apply happens later через
    `reinterpret_apply_sweeper` под write-gate'ом — HTTP route
    enqueue-only (D3 в waves/22).

    Возможные status'ы:
    - ``queued`` — task поставлен; ``task_id``/``application_id`` populated;
      apply через ~30-90s (после tick'а sweeper'а + close turn'а).
    - ``cooldown`` — последняя revision < 24h; ``next_available_at`` /
      ``last_reinterpreted_at`` populated.
    - ``already_pending`` — есть pending_sleep application для memory;
      ``pending_application_id`` populated.
    - ``memory_not_found`` — memory нет под этим agent_id.
    """

    status: str
    memory_id: str | None = None
    task_id: str | None = None
    application_id: int | None = None
    message: str | None = None
    next_available_at: str | None = None
    last_reinterpreted_at: str | None = None
    pending_application_id: int | None = None


# ── ingest_experience (волна 23) ─────────────────────────────────────────


class IngestExperienceRequest(BaseModel):
    """Pipeline ingest entry. Идемпотентен по ``content_hash``.

    Hash priority (D3 в waves/23):
      1. Явный ``content_hash`` — pipeline сам контролирует.
      2. Auto-compute из ``(pipeline_id, pipeline_version, content_ref)``
         если все три заданы и ``content_ref`` не пуст.
      3. Иначе ``None`` — partial UNIQUE индекс игнорирует, idempotency
         не применяется.

    ``content`` ≤ 2400 chars (CHECK constraint
    ``memories_content_length_check``). Длинные документы — отдельный
    канал (OpenClaw plugin track).
    """

    agent_id: str
    content: str = Field(min_length=1, max_length=2400)
    kind: str = "note"
    kind_src: str = "experience_intake"
    metadata: dict[str, Any] = Field(default_factory=dict)
    importance_provisional: float | None = None
    content_hash: str | None = Field(default=None, max_length=256)
    pipeline_id: str | None = Field(default=None, max_length=64)
    pipeline_version: str | None = Field(default=None, max_length=64)
    content_ref: dict[str, Any] | None = None


class IngestDocumentRequest(BaseModel):
    """File-ingest entry (волна 28).

    Plugin шлёт абсолютный path; core читает диск, парсит, режет на
    chunks, embed'ит, INSERT'ит document + chunks. tail-memory НЕ
    создаётся (pull-only архив, D5 в waves/28).

    Поддерживаемые форматы: PDF, DOCX, XLSX, Markdown, plain text.

    ``content_hash`` — опц. SHA256 override; обычно core вычисляет сам.
    Partial UNIQUE на (agent_id, content_hash) обеспечивает
    идемпотентность повторного ingest того же файла.
    """

    agent_id: str
    path: str = Field(min_length=1)
    source_ref: str | None = Field(default=None, max_length=512)
    visibility: str | None = Field(default=None, max_length=32)
    metadata: dict[str, Any] = Field(default_factory=dict)
    content_hash: str | None = Field(default=None, max_length=256)


class IngestDocumentResponse(BaseModel):
    """Результат /ingest_document.

    - ``deduplicated=False`` — новый document создан, ``chunks_count``
      реальное число chunks.
    - ``deduplicated=True``  — повторный ingest того же файла; existing
      ``document_id`` возвращён, chunks_count=0 (нет новых INSERT'ов).

    ``content_hash`` — что использовалось при INSERT'е (SHA256 file
    bytes либо explicit override).

    ``act_marker_memory_id`` (Defect-fix A) — id tail-memory с
    маркером акта архивации («я положил в архив документ»; IAmBook
    §V). ``None`` при dedup.

    ``chunks_embedded_inline`` (Defect-fix A) — False означает, что
    документ большой: chunks записаны без embedding'а, embedding
    выполняется async в worker pool. ``search_archive`` найдёт
    документ после того как async-задача отработает.
    """

    document_id: str
    deduplicated: bool
    chunks_count: int
    mime_type: str
    original_name: str
    size_bytes: int
    char_count: int
    content_hash: str
    act_marker_memory_id: str | None = None
    chunks_embedded_inline: bool = True


class IngestExperienceResponse(BaseModel):
    """Результат ingest'а.

    - ``deduplicated=False`` — новый ряд создан, ``memory_id`` свежий.
    - ``deduplicated=True`` — повторный ingest того же payload'а от
      того же агента; ``memory_id`` указывает на existing ряд, никаких
      побочных эффектов (embedding/metadata из второго вызова ignored).
    - ``content_hash`` — что использовалось (explicit | auto-computed |
      ``None``); полезно pipeline'у для логирования.
    """

    memory_id: str
    deduplicated: bool
    content_hash: str | None = None


# ── dialogue tools (волна 24) ────────────────────────────────────────────


class DialogueSaveRequest(BaseModel):
    """Explicit ad-hoc save одной реплики (D5 в waves/24).

    Не триггерит auto-link / classifier / sentiment — pipeline-канал,
    не natural turn (как ``ingest_experience``). Для полного pipeline
    с побочными эффектами — ``POST /sync_turn``.

    ``role`` — только 'user'/'assistant'. ``content`` ≤ 2400 (CHECK
    constraint). ``session_id`` опц.: если задан — ``upsert_session``
    идемпотентно, иначе FK NULL.
    """

    agent_id: str
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=2400)
    session_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class DialogueSaveResponse(_LlmWrappableResponse):
    memory_id: str


class DialogueSearchRequest(BaseModel):
    """Hybrid (FTS+vector) либо pure-vector search.

    ``semantic_only=True`` отключает BM25 component — pure cosine.
    Default — hybrid (D6 в waves/24). Cross-agent НЕТ (D13).
    """

    agent_id: str
    query: str = Field(min_length=1, max_length=2000)
    session_id: str | None = None
    after: _dt.datetime | None = None
    before: _dt.datetime | None = None
    semantic_only: bool = False
    limit: int = Field(default=10, ge=1, le=50)


class DialogueSearchHitModel(BaseModel):
    memory_id: str
    role: str
    content: str
    score: float
    created_at: _dt.datetime
    session_id: str | None = None


class DialogueSearchResponse(_LlmWrappableResponse):
    results: list[DialogueSearchHitModel] = Field(default_factory=list)


class DialogueRecentRequest(BaseModel):
    """Pure chronological retrieval. Caller получает oldest-first."""

    agent_id: str
    session_id: str | None = None
    before: _dt.datetime | None = None
    limit: int = Field(default=20, ge=1, le=200)


class DialogueRecentRowModel(BaseModel):
    memory_id: str
    role: str
    content: str
    created_at: _dt.datetime
    session_id: str | None = None


class DialogueRecentResponse(_LlmWrappableResponse):
    rows: list[DialogueRecentRowModel] = Field(default_factory=list)


class DialogueSessionsRequest(BaseModel):
    agent_id: str
    limit: int = Field(default=10, ge=1, le=100)


class DialogueSessionInfoModel(BaseModel):
    session_id: str
    message_count: int
    first_message_at: _dt.datetime
    last_message_at: _dt.datetime


class DialogueSessionsResponse(_LlmWrappableResponse):
    sessions: list[DialogueSessionInfoModel] = Field(default_factory=list)


class DialoguePrepareSummaryRequest(BaseModel):
    """Готовит transcript конкретной session для summarizer-агента.

    ``session_id`` обязателен (D9). Пустая session → empty transcript,
    не 404.
    """

    agent_id: str
    session_id: str = Field(min_length=1)
    limit: int = Field(default=200, ge=1, le=1000)


class DialoguePrepareSummaryResponse(_LlmWrappableResponse):
    session_id: str
    message_count: int
    first_message_at: _dt.datetime | None = None
    last_message_at: _dt.datetime | None = None
    transcript: str


# ── Explain / analytics / confirm_usage (волна 25) ───────────────────────


class ExplainDecomposeRequest(BaseModel):
    """11-факторный breakdown скоринга для (memory_id, query).

    `top_k_limit` — границы для `not_returned_because='outside_top_k'`.
    `min_score` — опц. порог для `not_returned_because='below_min_score'`
    (только из input, без env/config-fallback'а — § 40).
    """

    agent_id: str
    memory_id: str
    query: str = Field(min_length=1, max_length=2000)
    top_k_limit: int = Field(default=10, ge=1, le=200)
    min_score: float | None = None


class ExplainDecomposeResponse(_LlmWrappableResponse):
    mode: Literal["decompose"] = "decompose"
    memory_id: str
    kind: str
    query: str
    final_score: float
    rank_in_result_set: int | None = None
    top_k_limit: int
    would_be_returned: bool
    return_reason: Literal["top_k", "top_k_with_min_score"] | None = None
    not_returned_because: dict[str, Any] | None = None
    factors: dict[str, Any] = Field(default_factory=dict)
    computed_at: str


class ExplainLifetimeRequest(BaseModel):
    """Lifecycle trace: importance, lifecycle, recall_history, decay
    projections, co-retrieval links.

    `prune_min_relevance` — опц. порог для расчёта
    `decay.estimated_days_to_prune_threshold`. Без него поле = None.
    """

    agent_id: str
    memory_id: str
    include_recall_history: bool = True
    recall_history_limit: int = Field(default=10, ge=1, le=100)
    prune_min_relevance: float | None = None


class ExplainLifetimeResponse(_LlmWrappableResponse):
    mode: Literal["lifetime"] = "lifetime"
    memory_id: str
    content_preview: str
    kind: str
    agent_id: str
    visibility: str
    created_at: str
    updated_at: str
    age_days: float
    importance: dict[str, Any] = Field(default_factory=dict)
    lifecycle: dict[str, Any] = Field(default_factory=dict)
    access: dict[str, Any] = Field(default_factory=dict)
    relevance: dict[str, Any] = Field(default_factory=dict)
    usefulness: dict[str, Any] = Field(default_factory=dict)
    decay: dict[str, Any] = Field(default_factory=dict)
    recall_history: list[dict[str, Any]] | None = None
    co_retrieval_links: list[dict[str, Any]] = Field(default_factory=list)
    computed_at: str


class ExplainTopKRequest(BaseModel):
    """Top-K с factor breakdown'ом каждого. `include_factors=False`
    отключает factor blocks (быстрее)."""

    agent_id: str
    query: str = Field(min_length=1, max_length=2000)
    limit: int = Field(default=10, ge=1, le=50)
    kinds: list[str] | None = None
    after: _dt.datetime | None = None
    before: _dt.datetime | None = None
    min_score: float | None = None
    include_factors: bool = True


class ExplainTopKResponse(_LlmWrappableResponse):
    mode: Literal["top_k"] = "top_k"
    query: str
    limit: int
    total_candidates_considered: int
    items: list[dict[str, Any]] = Field(default_factory=list)
    computed_at: str


class AgentCacheStatsRequest(BaseModel):
    """Push cache stats от Hermes-side ``StyxAnthropicTransport``
    (волна 29 Phase E).

    Tokens должны быть ≥ 0. ``cache_read_tokens`` высокий → cache hit
    rate хороший; ``cache_creation_tokens`` высокий после cold start
    или cache invalidation. Кумулятив per-agent в memory только (не
    persistent — обнуляется при daemon рестарте).
    """

    agent_id: str = Field(min_length=1)
    cache_read_tokens: int = Field(ge=0)
    cache_creation_tokens: int = Field(ge=0)


class AnalyticsResponse(BaseModel):
    """Per-agent counts + global totals + pending indexing.

    `agents` — массив длиной 1 (caller-only); поле сохранено для
    parity-формы memorybox client'ов. `global_` сериализуется как
    `global` (reserved word в Python).

    `styx_sanitized_blocks_total` (волна 26.5, Fix 6) — cumulative
    счётчик styx-блоков, вырезанных из input messages с момента
    старта daemon'а (агрегат по всем тегам).

    `styx_sanitized_blocks_by_tag` (волна 30, Phase F) — breakdown
    того же счётчика по суффиксам тегов: ``salient``, ``recall``,
    ``archive``, ``dialogue``, ``relations``, ``explain``,
    ``working-set`` (family) + ``legacy`` для голого
    ``<styx>...</styx>`` эпохи 26.5. Должен оставаться пустым в
    стабильном runtime; ненулевые значения → утечка assembled view
    или tool result (transcript echo / snapshot replay / native
    memory dump) — диагностический сигнал для оператора с указанием
    источника leak'а.

    `cache_stats` (волна 29 Phase E) — per-agent cumulative cache
    hit/miss tokens, push'нутые от Hermes ``StyxAnthropicTransport``
    через ``POST /agent/cache_stats``. Поля: ``cache_read_tokens``
    (cumulative из cache), ``cache_creation_tokens`` (cumulative
    written to cache), ``samples`` (количество turn'ов). Operator
    видит cache hit rate per agent_id; ratio
    ``cache_read / (cache_read + cache_creation)`` стремится к 1.0
    при stable cache prefix.
    """

    agents: list[dict[str, Any]] = Field(default_factory=list)
    global_: dict[str, Any] = Field(
        default_factory=dict, serialization_alias="global"
    )
    pending_indexing: dict[str, Any] = Field(default_factory=dict)
    styx_sanitized_blocks_total: int = 0
    styx_sanitized_blocks_by_tag: dict[str, int] = Field(default_factory=dict)
    cache_stats: dict[str, int] = Field(default_factory=dict)

    model_config = {"populate_by_name": True}


class ConfirmUsageRequest(BaseModel):
    """Explicit `used_in_output=true` для recall_event'ов.

    Cross-agent guard: memory_ids чужого агента не апдейтятся, попадают
    в `missing` response'а.
    """

    agent_id: str
    memory_ids: list[str] = Field(min_length=1, max_length=100)


class ConfirmUsageResponse(BaseModel):
    updated: int
    requested: int
    missing: list[str] = Field(default_factory=list)


# ── maintenance / reembed (волна 31) ──────────────────────────────────────


class MaintenanceReembedRequest(BaseModel):
    """HTTP-параметры backfill'а `memories.embedding` (паритет CLI 7e).

    Все поля опциональны. ``mode='null_only'`` (default) добивает только
    ``embedding IS NULL``; ``'all'`` — полный re-embed (например после смены
    модели). ``agent_id=None`` — все агенты. ``limit`` ограничивает число
    обработанных. ``dry_run=True`` возвращает ``would_process`` без UPDATE.
    """

    mode: Literal["null_only", "all"] = "null_only"
    agent_id: str | None = None
    limit: int | None = Field(default=None, ge=0)
    dry_run: bool = False
    batch_size: int = Field(default=50, ge=1)
    rate_per_second: float = Field(default=5.0, gt=0)


class MaintenanceReembedResponse(BaseModel):
    """Результат прогона.

    ``skipped=True`` — advisory lock (key 9876543211) занят другим
    instance'ом; backfill не запускался (``processed=failed=would_process=0``).
    ``elapsed_ms`` — wall-clock хендлера (включая connect + embed-loop).
    """

    processed: int
    failed: int
    would_process: int
    dry_run: bool
    elapsed_ms: int
    skipped: bool = False


# ── context lifecycle (волна 26 Phase B) ──────────────────────────────────


class ContextMessage(BaseModel):
    """Один message для /context/ingest и /context/ingest_batch.

    Mapping OpenClaw Message → styx insert_message: role + content. Допустимы
    {user, assistant, system, tool}. Дополнительные поля игнорируются —
    OpenClaw присылает богатый message shape (tool_calls, name, ts), но
    core хранит только role+content в memories.
    """

    role: str
    content: str = ""

    model_config = {"extra": "ignore"}


class ContextBootstrapRequest(BaseModel):
    """`POST /context/bootstrap` — engine first sees the session.

    Idempotent: если agent_id уже зарегистрирован — возвращает
    initialized=False. Иначе — выполняет agent/initialize. parent_session_id
    зарезервировано под subagent сценарии (Phase C+).
    """

    agent_id: str
    session_id: str | None = None
    parent_session_id: str | None = None


class ContextBootstrapResponse(BaseModel):
    ok: bool
    initialized: bool


class ContextIngestRequest(BaseModel):
    """`POST /context/ingest` — single-message ingest.

    Phase B-семантика (raw insert): upsert_session + insert_message +
    embed-after-commit. Без gatekeeper / auto-link / classifier-enqueue /
    sentiment — полный stack применяется в `/context/ingest_batch` через
    `sync_turn` для каждой смежной (user, assistant) пары.

    is_heartbeat=True — heartbeat tick без message; ingested=False.
    """

    agent_id: str
    session_id: str | None = None
    message: ContextMessage
    is_heartbeat: bool = False


class ContextIngestResponse(BaseModel):
    ok: bool
    ingested: bool
    memory_id: str | None = None


class ContextIngestBatchRequest(BaseModel):
    """`POST /context/ingest_batch` — целый turn после run'а.

    Phase B-семантика: route группирует messages в смежные (user, assistant)
    пары и для каждой зовёт `sync_turn(user_content, assistant_content)`.
    Хвостовая user-реплика без assistant'а — `sync_turn(user, "")`. Полный
    stack (auto-link / classifier / sentiment) применяется per-pair.
    """

    agent_id: str
    session_id: str | None = None
    messages: list[ContextMessage] = Field(default_factory=list)


class ContextIngestBatchResponse(BaseModel):
    ok: bool
    ingested_count: int


class ContextDisposeRequest(BaseModel):
    """`POST /context/dispose` — release engine resources.

    Если задан `agent_id` — сводится к `agent/shutdown`. Если оба null —
    no-op на текущий момент (per-engine dispose, plugin reload / gateway
    stop). Future: cancel pending work для всех зарегистрированных агентов.
    """

    agent_id: str | None = None
    session_id: str | None = None


class ContextDisposeResponse(BaseModel):
    ok: bool


# ── context lifecycle (волна 26 Phase C) ──────────────────────────────────


class ContextAssembleRequest(BaseModel):
    """`POST /context/assemble` — сборка геометрии входа на model run.

    Wrapper над ``StyxComposer.compress``: head+tail+salient inject поверх
    runtime'ом передаваемых ``messages``. ``token_budget`` маппится в
    ``current_tokens`` — composer eviction'ит по threshold (~80% от
    ``StyxConfig.context_length``). ``prompt`` передаётся в composer как
    focus_topic hint (waves-v1 v1: игнорируется, оставлено для совместимости).
    ``available_tools``/``citations_mode``/``model`` зарезервированы под
    Phase D system_prompt_addition (через buildMemorySystemPromptAddition
    в plugin TS).
    """

    agent_id: str
    session_id: str | None = None
    messages: list[dict[str, Any]] = Field(default_factory=list)
    token_budget: int | None = None
    available_tools: list[str] | None = None
    citations_mode: str | None = None
    model: str | None = None
    prompt: str | None = None


class ContextAssembleResponse(BaseModel):
    """OpenClaw AssembleResult shape (camelCase в TS, snake_case в HTTP)."""

    messages: list[dict[str, Any]]
    estimated_tokens: int
    system_prompt_addition: str | None = None
    prompt_authority: str | None = None


class ContextCompactRequest(BaseModel):
    """`POST /context/compact` — slash /compact или overflow recovery.

    Phase C minimal: возвращает ``compacted=False`` ('no change') и не
    выполняет блокирующего LLM compaction. Реальная семантическая
    компрессия (memory_consolidation, ADR § 37) уже идёт async через
    workers/handlers/memory_daily_consolidation на интервалах. Pi /compact
    — это user gesture; нашему engine'у нет смысла мгновенно блокировать
    turn ради ускорения sweep'а. Phase D / future-волна добавит trigger
    на immediate consolidation tick если оператор реально начнёт
    использовать /compact.
    """

    agent_id: str
    session_id: str | None = None
    force: bool = False


class ContextCompactResponse(BaseModel):
    ok: bool
    compacted: bool
    reason: str | None = None
    session_id: str | None = None
    session_file: str | None = None


class ContextAfterTurnRequest(BaseModel):
    """`POST /context/after_turn` — fire-and-forget post-run hook.

    Phase C minimal: возвращает ``{ok: true}``. Реальные post-turn
    операции (drift recompute, salient cache refresh, sweep ticks) уже
    выполняются автоматически через workers/sweepers (волны 10/11/22).
    Endpoint существует чтобы plugin TS мог корректно implement'ить
    ContextEngine.afterTurn — runtime ждёт async-метод.
    """

    agent_id: str
    session_id: str | None = None
    messages: list[dict[str, Any]] = Field(default_factory=list)


class ContextAfterTurnResponse(BaseModel):
    ok: bool
