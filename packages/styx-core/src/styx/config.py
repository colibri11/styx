"""Конфиг Styx — DSN и параметры рантайма.

Источники в порядке приоритета (override снизу вверх):

1. Hardcoded defaults
2. ``$HERMES_HOME/styx.json`` (если файл есть)
3. Env vars (``STYX_*``)
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


CONFIG_FILENAME = "styx.json"


@dataclass(frozen=True)
class StyxConfig:
    database_url: str
    ollama_url: str = "http://ollama:11434"
    embedding_model: str = "embeddinggemma:300m-qat-q8_0"
    embedding_dim: int = 768
    embedding_timeout_s: float = 30.0
    # LLM chat (qwen3:4b-local) — отдельный URL/модель от embedding'а.
    # По умолчанию используется тот же endpoint что embedding (qwen3 и
    # embeddinggemma могут жить на одной Ollama).
    llm_url: str = "http://ollama:11434"
    llm_model: str = "qwen3:4b-local"
    llm_timeout_s: float = 60.0
    llm_max_attempts: int = 2
    llm_rate_limit_capacity: int = 4
    llm_rate_limit_refill_per_s: float = 1.0
    # Periodic sweep (волна 7b). Default — раз в час; в memorybox 1/день.
    # Styx — активный диалог, нужен короче.
    sweep_interval_s: float = 3600.0
    sweep_lock_timeout_s: float = 1800.0
    # Sentiment hot-path (волна 7d). Inline в sync_turn после embed.
    sentiment_enabled: bool = True
    sentiment_timeout_s: float = 0.8
    # Baseline tick (волна 7d). EMA α=0.98 над окном 60min, периодически.
    emotional_tick_interval_s: float = 60.0
    # Recall classifier (волна 7c). Минимальная длина assistant-ответа,
    # ниже которой classifier не enqueue'ится (короткие ответы — LLM
    # всё равно skip'нет, экономим один call).
    classifier_min_assistant_length: int = 50
    # Сколько recall_event_ids максимально в одном payload'е.
    classifier_max_recall_events_per_turn: int = 20
    # Recall threshold (волна 8). None = «использовать встроенный
    # дефолт из storage.recall_config» (DEFAULT_RECALL_CONFIG.full.min_score).
    # Override через STYX_RECALL_MIN_SCORE / STYX_RECALL_DIALOGUE_MIN_SCORE
    # либо styx.json. Применяется через resolve_recall_config(partial)
    # в StyxMemoryProvider.initialize.
    recall_min_score: float | None = None
    recall_dialogue_min_score: float | None = None
    # None = использовать встроенный дефолт FullRecallConfig.memory_limit
    # (6). Override через STYX_RECALL_MEMORY_LIMIT либо styx.json.
    recall_memory_limit: int | None = None
    # Salient memories injection (волна 9). enabled=False полностью
    # отключает inject в compress() (для отладки и тестов которые не
    # должны ходить в Ollama). timeout_s — общий лимит на recall'е,
    # превышение → skip salient (fail-open). min_query_len — минимум
    # длины last user message чтобы embed не шумел на коротких строках.
    salient_enabled: bool = True
    salient_timeout_s: float = 1.0
    salient_min_query_len: int = 20
    # Drift detection (волна 10). enabled=False → focus_tracker не configure'ится,
    # build_salient_block fallback'ит на волна-9 поведение (fresh recall каждый
    # turn). drift_threshold — cosine, ниже которого считаем сменой темы;
    # рассчитан из bench-suite волны 8 (paraphrase p10=0.439, unrelated p90=0.207).
    # focus_window_size — K последних user-embed'ов в скользящем centroid'е.
    drift_enabled: bool = True
    drift_threshold: float = 0.4
    focus_window_size: int = 3
    # Pre-LLM focus inject (волна 15). Multi-channel framework для
    # инжекта в user message через Hermes pre_llm_call hook. Канал
    # self_state (волна 35, заменяет прежний peer_vad) — отметка о
    # накопленном эмоциональном состоянии самого агента (read_last_state,
    # не raw peer VAD). min_norm — тот же gating что был у peer_vad;
    # max_age_s — safety net на случай мёртвого styx-worker (не «окно
    # свежести реакции»), см. .design/waves/35-self-state-expression.md D3.
    pre_llm_inject_enabled: bool = True
    self_state_enabled: bool = True
    self_state_min_norm: float = 0.2
    self_state_max_age_s: float = 900.0
    # Selective gatekeeper (волна 17). На subjective writes (insert_memory
    # / batch consolidation) выбирает skip / merge / supersede / store по
    # similarity + Levenshtein. enabled=False → каждый subjective write
    # идёт как store; thresholds — port memorybox `selective.ts`.
    # noise_min_length режет короткий шум («да», «ок») до проверки соседей.
    selective_enabled: bool = True
    selective_merge_threshold: float = 0.92
    selective_supersede_threshold: float = 0.85
    selective_levenshtein_threshold: float = 0.3
    selective_noise_filter: bool = True
    selective_noise_min_length: int = 10
    # Auto-link при INSERT (волна 18). После gatekeeper store/supersede
    # и для каждого dialogue ряда в sync_turn — находит ближайших
    # соседей по embedding'у и пишет related_to рёбра в `relations`.
    # Cross-agent: SELECT соседей без agent_id фильтра (общий пул
    # знаний между агентами в одном PG). Дефолты — port memorybox
    # `auto-linking.ts` (max_distance=0.25 → similarity ≥ 0.75).
    auto_link_enabled: bool = True
    auto_link_max_distance: float = 0.25
    auto_link_max_links: int = 3
    # Hebbian co-retrieval reinforcement (волна 21). При recall'е N≥2
    # memories — bump weight ребра 'co_retrieved' между всеми pairs.
    # Initial weight 1.1 (не 1.0) чтобы decay'нувшие cold links
    # отличались от никогда-не-reinforced baseline. Bump 0.1 →
    # насыщение за 10 совместных recall'ов; cap 2.0.
    hebbian_enabled: bool = True
    hebbian_weight_bump: float = 0.1
    hebbian_initial_weight: float = 1.1
    hebbian_weight_max: float = 2.0
    # Store-routing для длинного content'а (волна 19). При len(content)
    # > limit memory_store / insert_batch_memory разделяет content на
    # chunks (engine.chunker) и пишет в documents+chunks; в memories
    # остаётся tail-memory с archive_ref. Дефолты — port memorybox
    # `tools/memory.ts:STORE_ROUTING_THRESHOLD` (2400) +
    # `chunker.ts` (1600/320). summary_chars=1500 — отступ от CHECK
    # constraint'а memories.content ≤ 2400.
    store_routing_enabled: bool = True
    store_routing_limit: int = 2400
    store_routing_chunk_size: int = 1600
    store_routing_chunk_overlap: int = 320
    store_routing_summary_chars: int = 1500
    # Сплиттер больших реплик дневника (Defect-fix B). Реплику
    # (user/assistant) длиннее store_routing_limit режем на ряды
    # ≤ message_split_part_chars — каждый остаётся в memories (дневник
    # = речь, не архив; IAmBook §V). message_split_inline_embed_cap —
    # сколько частей embed'ить inline в hot-path; остаток embed'ит
    # re-embed CLI / воркер (embedding NULL, как при EmbeddingError).
    message_split_part_chars: int = 2000
    message_split_inline_embed_cap: int = 4
    # Async-порог file-ingest'а (Defect-fix A). Документ, который
    # chunker делит больше чем на document_ingest_async_chunk_threshold
    # chunks, ingest'ится через worker pool (endpoint возвращается
    # быстро). Документ меньше — inline.
    document_ingest_async_chunk_threshold: int = 12
    # File-ingest pipeline (волна 28). STYX_INGEST_DOC_ROOTS —
    # colon-separated whitelist абсолютных директорий (как PATH); empty
    # → no whitelist (lab mode). max_bytes — отказ при превышении.
    # ingest_doc_enabled — toggle для всего endpoint'а; при False
    # /ingest_document отвечает 503.
    ingest_doc_enabled: bool = True
    ingest_doc_roots: list[str] = field(default_factory=list)
    ingest_doc_max_bytes: int = 50 * 1024 * 1024
    # Search archive (волна 20). Pull-канал к архиву documents/chunks +
    # dialogue (memories WHERE role IN ('user','assistant')). Hybrid
    # FTS+vector через те же compute_weights что и search_similar.
    # k_candidates_factor × limit (clamp k_candidates_min) — сколько
    # raw chunks тянуть для documents-scope чтобы stitching имел
    # соседей. Port memorybox `Math.min(limit * 8, 80)`.
    search_archive_default_limit: int = 10
    search_archive_max_limit: int = 50
    search_archive_k_candidates_factor: int = 8
    search_archive_k_candidates_min: int = 80
    # Relation decay periodic-task (волна 21). Раз в час уменьшает
    # weight 'co_retrieved' рёбер у которых last_reinforced > idle_days.
    # Floor 1.0. Дефолты — port memorybox `relation-decay.ts`.
    relation_decay_enabled: bool = True
    relation_decay_interval_s: float = 3600.0
    relation_decay_rate: float = 0.05
    relation_decay_idle_days: int = 14
    # Temporal isolation (волна 14, 10a). TTL inactivity для турn'а —
    # если close() не вызвался по какой-то причине, через этот интервал
    # turn автоматически закроется при следующем observe().
    turn_state_ttl_s: float = 60.0
    # Memory-over-memory daily consolidation (волна 22). Scheduler раз
    # в час кластеризует «соседей» по cosine ≥ 0.88 в окне
    # [now-7d..now-24h], INSERT memory_daily_consolidation task'и.
    # Apply-sweeper раз в 30s применяет результат под write-gate'ом.
    # Cooldown 23h на agent — RUN_COOLDOWN_HOURS из memorybox.
    memory_consolidation_enabled: bool = True
    memory_consolidation_tick_s: float = 3600.0
    memory_consolidation_apply_tick_s: float = 30.0
    memory_consolidation_cooldown_h: int = 23
    memory_consolidation_window_days: int = 7
    memory_consolidation_window_tail_h: int = 24
    memory_consolidation_cosine: float = 0.88
    memory_consolidation_min_size: int = 3
    memory_consolidation_max_size: int = 8
    # Reinterpret (волна 22). Explicit caller-side tool: HTTP route
    # /reinterpret + Hermes wrapper styx_reinterpret. Apply-sweeper
    # раз в 30s применяет результат под write-gate'ом. Cooldown 24h
    # на memory.
    reinterpret_enabled: bool = True
    reinterpret_apply_tick_s: float = 30.0
    reinterpret_cooldown_s: int = 86400  # 24h
    reinterpret_blend_weight: float = 0.5
    # Ingest API (волна 23). Внешний канал для pipelines (AudioBox,
    # future ingest workers): POST /ingest_experience с идемпотентностью
    # по content_hash. Когда False — endpoint отвечает 503.
    ingest_api_enabled: bool = True
    # Dialogue tools API (волна 24). 5 routes под /dialogue/* —
    # explicit save / hybrid search / chronological recent / sessions
    # listing / transcript prepare для summarizer'а. Источник —
    # `memories WHERE role IN ('user','assistant')`. Когда False —
    # все 5 routes отвечают 503 (один toggle на всю surface).
    dialogue_api_enabled: bool = True
    # Explain / analytics / confirm_usage API (волна 25). Observability
    # surface: 3 explain routes (decompose/lifetime/topK) поверх
    # build_factor_exprs SQL, GET /analytics с per-agent counts +
    # global totals + pending indexing, POST /confirm_usage для
    # explicit `used_in_output=true`. Когда False — все 5 routes
    # отвечают 503 (один toggle на всю observability surface).
    explain_api_enabled: bool = True
    # Batch consolidation (волна 14). Periodic-task scheduler в
    # styx-worker'е enqueue'ит llm_tasks для batch handler'а при
    # выполнении триггеров (≥20 реплик ∧ ≥20 мин с прошлого batch'а).
    batch_consolidation_enabled: bool = True
    batch_tick_interval_s: float = 60.0
    batch_trigger_messages: int = 20
    batch_trigger_interval_s: int = 1200  # 20 мин
    batch_period_gap_s: int = 21600  # 6 ч
    # Batch sentiment piggyback (волна 14, memorybox 26b). Тот же
    # LLM-вызов возвращает интегральный VAD; handler пишет в
    # emotional_state с source='sentiment:batch' и K_BATCH=0.4.
    batch_sentiment_enabled: bool = True
    # Hot-tier (волна 11). In-process store memory items, прошедших
    # через recall_full недавно (TTL). Используется как supplement
    # в recall_full — items добавляются как extra candidates до
    # filter+dedup+slice. enabled=False → state не configure'ится,
    # supplement пуст, put no-op.
    hot_tier_enabled: bool = True
    hot_tier_ttl_s: float = 300.0
    hot_tier_lru_bound: int = 100
    # Eviction relevance-aware (волна 12). При переполнении окна compress
    # выбирает top-K pair-групп из middle по cosine к focus centroid'у
    # и keep'ит между head и tail. enabled=False → handle не configure'ится,
    # apply_relevance_eviction возвращает []; eviction идентичен волне 11.
    eviction_relevance_enabled: bool = True
    eviction_relevance_keep_k: int = 2
    eviction_relevance_threshold: float = 0.4
    # Working set state persistence (волна 13). Persistence focus_tracker
    # (window + cached_salient + epoch_id) и hot_tier (entries) между
    # restart'ами процесса. enabled=False → load skip, save-thread не
    # стартует, shutdown не flush'ит. save_interval_s — период между
    # save tick'ами; ttl_s — past TTL state'а на load дропается (cold
    # start). Подробности в ADR § 29.
    working_set_persistence_enabled: bool = True
    working_set_save_interval_s: float = 30.0
    working_set_ttl_s: float = 86400.0
    # Healthz endpoint в worker-процессе (волна 16). port=0 отключает
    # HTTP-сервер целиком (для тестов где порт не нужен). bind=127.0.0.1
    # — безопасный дефолт; в Docker ставится 0.0.0.0 через ENV.
    # liveness_threshold_s — `process_one`/`_main_loop` итерация старше
    # threshold'а считается зависшей (/healthz → 503).
    # readiness_threshold_s — drain progress старше threshold'а →
    # /readyz → 503.
    healthz_port: int = 8788
    healthz_bind: str = "127.0.0.1"
    healthz_liveness_threshold_s: float = 120.0
    healthz_readiness_threshold_s: float = 300.0
    # HTTP API daemon (Phase C). FastAPI app + worker pool в одном
    # процессе. http_bind=127.0.0.1 — loopback-only по умолчанию;
    # http_token=None допустим только на loopback (на 0.0.0.0
    # обязателен, иначе daemon не стартует).
    http_bind: str = "127.0.0.1"
    http_port: int = 8788
    http_token: str | None = None
    # Postgres connection pool (D20). pg_pool_min — минимум живых
    # connections, pg_pool_max — верхняя граница. FastAPI handler
    # acquires conn per request, release back to pool на exit.
    pg_pool_min: int = 2
    pg_pool_max: int = 10
    # CLI healthcheck — куда стучать в `styx daemon healthcheck`.
    daemon_url: str = "http://127.0.0.1:8788"
    # Logging format (волна 16). "text" для dev/тестов, "json" для
    # production deployment. JSON-line формат описан в
    # styx.observability.logging.
    log_format: str = "text"
    extra: dict[str, Any] = field(default_factory=dict)

    def gatekeeper_config(self) -> "GatekeeperConfig":
        """Derived getter — собирает GatekeeperConfig для волны 17.

        Lazy import чтобы избежать circular между config и engine.
        """
        from styx.engine.selective_gatekeeper import GatekeeperConfig
        return GatekeeperConfig(
            enabled=self.selective_enabled,
            merge_threshold=self.selective_merge_threshold,
            supersede_threshold=self.selective_supersede_threshold,
            levenshtein_threshold=self.selective_levenshtein_threshold,
            noise_filter=self.selective_noise_filter,
            noise_min_length=self.selective_noise_min_length,
        )

    def auto_link_config(self) -> "AutoLinkConfig":
        """Derived getter — собирает AutoLinkConfig для волны 18."""
        from styx.engine.auto_link import AutoLinkConfig
        return AutoLinkConfig(
            enabled=self.auto_link_enabled,
            max_distance=self.auto_link_max_distance,
            max_links=self.auto_link_max_links,
        )

    def store_routing_config(self) -> "StoreRoutingConfig":
        """Derived getter — собирает StoreRoutingConfig для волны 19."""
        from styx.engine.store_routing import StoreRoutingConfig
        return StoreRoutingConfig(
            enabled=self.store_routing_enabled,
            limit=self.store_routing_limit,
            chunk_size=self.store_routing_chunk_size,
            chunk_overlap=self.store_routing_chunk_overlap,
            summary_chars=self.store_routing_summary_chars,
        )

    def document_ingest_config(self) -> "DocumentIngestConfig":
        """Derived getter — собирает DocumentIngestConfig для волны 28.

        ``allowed_roots`` парсится из списка строк в list[Path]. Все
        ряды должны быть absolute — иначе ValueError при сборке.
        """
        from pathlib import Path

        from styx.engine.document_ingest import DocumentIngestConfig

        roots: list[Path] = []
        for raw in self.ingest_doc_roots:
            p = Path(raw)
            if not p.is_absolute():
                raise ValueError(
                    f"STYX_INGEST_DOC_ROOTS entry must be absolute: {raw!r}"
                )
            roots.append(p.resolve())
        return DocumentIngestConfig(
            allowed_roots=roots,
            max_bytes=self.ingest_doc_max_bytes,
        )

    def hebbian_config(self) -> "HebbianConfig":
        """Derived getter — собирает HebbianConfig для волны 21."""
        from styx.engine.hebbian import HebbianConfig
        return HebbianConfig(
            enabled=self.hebbian_enabled,
            weight_bump=self.hebbian_weight_bump,
            initial_weight=self.hebbian_initial_weight,
            weight_max=self.hebbian_weight_max,
        )

    def search_archive_config(self) -> "SearchArchiveConfig":
        """Derived getter — собирает SearchArchiveConfig для волны 20."""
        from styx.engine.search_archive import SearchArchiveConfig
        return SearchArchiveConfig(
            default_limit=self.search_archive_default_limit,
            max_limit=self.search_archive_max_limit,
            k_candidates_factor=self.search_archive_k_candidates_factor,
            k_candidates_min=self.search_archive_k_candidates_min,
        )

    def memory_consolidation_config(self) -> "MemoryConsolidationConfig":
        """Derived getter — MemoryConsolidationConfig для волны 22."""
        from styx.engine.memory_consolidation import MemoryConsolidationConfig
        return MemoryConsolidationConfig(
            enabled=self.memory_consolidation_enabled,
            tick_s=self.memory_consolidation_tick_s,
            apply_tick_s=self.memory_consolidation_apply_tick_s,
            cooldown_hours=self.memory_consolidation_cooldown_h,
            window_days=self.memory_consolidation_window_days,
            window_tail_hours=self.memory_consolidation_window_tail_h,
            cosine_threshold=self.memory_consolidation_cosine,
            min_cluster_size=self.memory_consolidation_min_size,
            max_cluster_size=self.memory_consolidation_max_size,
        )

    def reinterpret_config(self) -> "ReinterpretConfig":
        """Derived getter — ReinterpretConfig для волны 22."""
        from styx.engine.reinterpret import ReinterpretConfig
        return ReinterpretConfig(
            enabled=self.reinterpret_enabled,
            apply_tick_s=self.reinterpret_apply_tick_s,
            cooldown_s=self.reinterpret_cooldown_s,
            blend_weight=self.reinterpret_blend_weight,
        )


def load(hermes_home: str | os.PathLike[str] | None = None) -> StyxConfig:
    """Собрать конфиг из json + env. Кидает ValueError при отсутствии DSN."""
    file_values = _read_json(hermes_home)
    env_values = _read_env()

    merged: dict[str, Any] = {}
    merged.update(file_values)
    merged.update({k: v for k, v in env_values.items() if v is not None})

    dsn = merged.get("database_url")
    if not dsn:
        raise ValueError(
            "DSN для Styx не задан. Проставь STYX_DATABASE_URL либо "
            "положи database_url в $HERMES_HOME/styx.json."
        )

    known = {
        "database_url",
        "ollama_url",
        "embedding_model",
        "embedding_dim",
        "embedding_timeout_s",
        "llm_url",
        "llm_model",
        "llm_timeout_s",
        "llm_max_attempts",
        "llm_rate_limit_capacity",
        "llm_rate_limit_refill_per_s",
        "sweep_interval_s",
        "sweep_lock_timeout_s",
        "sentiment_enabled",
        "sentiment_timeout_s",
        "emotional_tick_interval_s",
        "classifier_min_assistant_length",
        "classifier_max_recall_events_per_turn",
        "recall_min_score",
        "recall_dialogue_min_score",
        "recall_memory_limit",
        "salient_enabled",
        "salient_timeout_s",
        "salient_min_query_len",
        "drift_enabled",
        "drift_threshold",
        "focus_window_size",
        "pre_llm_inject_enabled",
        "self_state_enabled",
        "self_state_min_norm",
        "self_state_max_age_s",
        "selective_enabled",
        "selective_merge_threshold",
        "selective_supersede_threshold",
        "selective_levenshtein_threshold",
        "selective_noise_filter",
        "selective_noise_min_length",
        "auto_link_enabled",
        "auto_link_max_distance",
        "auto_link_max_links",
        "store_routing_enabled",
        "store_routing_limit",
        "store_routing_chunk_size",
        "store_routing_chunk_overlap",
        "store_routing_summary_chars",
        "message_split_part_chars",
        "message_split_inline_embed_cap",
        "document_ingest_async_chunk_threshold",
        "ingest_doc_enabled",
        "ingest_doc_roots",
        "ingest_doc_max_bytes",
        "search_archive_default_limit",
        "search_archive_max_limit",
        "search_archive_k_candidates_factor",
        "search_archive_k_candidates_min",
        "hebbian_enabled",
        "hebbian_weight_bump",
        "hebbian_initial_weight",
        "hebbian_weight_max",
        "relation_decay_enabled",
        "relation_decay_interval_s",
        "relation_decay_rate",
        "relation_decay_idle_days",
        "turn_state_ttl_s",
        "batch_consolidation_enabled",
        "batch_tick_interval_s",
        "batch_trigger_messages",
        "batch_trigger_interval_s",
        "batch_period_gap_s",
        "batch_sentiment_enabled",
        "memory_consolidation_enabled",
        "memory_consolidation_tick_s",
        "memory_consolidation_apply_tick_s",
        "memory_consolidation_cooldown_h",
        "memory_consolidation_window_days",
        "memory_consolidation_window_tail_h",
        "memory_consolidation_cosine",
        "memory_consolidation_min_size",
        "memory_consolidation_max_size",
        "reinterpret_enabled",
        "reinterpret_apply_tick_s",
        "reinterpret_cooldown_s",
        "reinterpret_blend_weight",
        "ingest_api_enabled",
        "dialogue_api_enabled",
        "explain_api_enabled",
        "hot_tier_enabled",
        "hot_tier_ttl_s",
        "hot_tier_lru_bound",
        "eviction_relevance_enabled",
        "eviction_relevance_keep_k",
        "eviction_relevance_threshold",
        "working_set_persistence_enabled",
        "working_set_save_interval_s",
        "working_set_ttl_s",
        "healthz_port",
        "healthz_bind",
        "healthz_liveness_threshold_s",
        "healthz_readiness_threshold_s",
        "http_bind",
        "http_port",
        "http_token",
        "pg_pool_min",
        "pg_pool_max",
        "daemon_url",
        "log_format",
    }
    extra = {k: v for k, v in merged.items() if k not in known}

    # Валидация message_split_part_chars против CHECK constraint'а
    # memories_content_length_check (MEMORIES_CONTENT_LIMIT = 2400):
    # сплиттер режет реплику на части ≤ part_chars, каждая остаётся
    # одним рядом memories. Если оператор выставит part_chars ≥ лимита
    # — каждый длинный turn будет валиться с ContentTooLongError.
    # Fail-fast на старте, не молчаливый clamp. Lazy import — чтобы не
    # тянуть storage в config на уровне модуля.
    from styx.storage.queries import MEMORIES_CONTENT_LIMIT

    split_part_chars = int(merged.get("message_split_part_chars", 2000))
    if split_part_chars >= MEMORIES_CONTENT_LIMIT:
        raise ValueError(
            f"message_split_part_chars ({split_part_chars}) должен быть "
            f"строго меньше MEMORIES_CONTENT_LIMIT ({MEMORIES_CONTENT_LIMIT}): "
            "иначе нарезанные части превысят CHECK constraint memories.content "
            "и каждый длинный turn упадёт с ContentTooLongError. "
            "Снизь STYX_MESSAGE_SPLIT_PART_CHARS / styx.json."
        )

    # Валидация recall_memory_limit: тот же диапазон 1-20 что и
    # per-call override styx_recall.limit в providers/memory.py
    # (JSON-схема тула объявляет "minimum": 1, "maximum": 20). Fail-fast
    # на старте, не молчаливый clamp.
    recall_memory_limit = _optional_int(merged.get("recall_memory_limit"))
    if recall_memory_limit is not None and not (1 <= recall_memory_limit <= 20):
        raise ValueError(
            f"recall_memory_limit ({recall_memory_limit}) вне допустимого "
            "диапазона 1-20 (тот же диапазон что per-call override "
            "styx_recall.limit). Поправь STYX_RECALL_MEMORY_LIMIT / styx.json."
        )

    ollama_default = merged.get("ollama_url", "http://ollama:11434")
    return StyxConfig(
        database_url=dsn,
        ollama_url=ollama_default,
        embedding_model=merged.get("embedding_model", "embeddinggemma:300m-qat-q8_0"),
        embedding_dim=int(merged.get("embedding_dim", 768)),
        embedding_timeout_s=float(merged.get("embedding_timeout_s", 30.0)),
        # LLM-defaults: если llm_url не задан — используем тот же что
        # ollama_url (qwen3 на той же Ollama что embeddinggemma).
        llm_url=merged.get("llm_url", ollama_default),
        llm_model=merged.get("llm_model", "qwen3:4b-local"),
        llm_timeout_s=float(merged.get("llm_timeout_s", 60.0)),
        llm_max_attempts=int(merged.get("llm_max_attempts", 2)),
        llm_rate_limit_capacity=int(merged.get("llm_rate_limit_capacity", 4)),
        llm_rate_limit_refill_per_s=float(
            merged.get("llm_rate_limit_refill_per_s", 1.0)
        ),
        sweep_interval_s=float(merged.get("sweep_interval_s", 3600.0)),
        sweep_lock_timeout_s=float(merged.get("sweep_lock_timeout_s", 1800.0)),
        sentiment_enabled=bool(merged.get("sentiment_enabled", True)),
        sentiment_timeout_s=float(merged.get("sentiment_timeout_s", 0.8)),
        emotional_tick_interval_s=float(
            merged.get("emotional_tick_interval_s", 60.0)
        ),
        classifier_min_assistant_length=int(
            merged.get("classifier_min_assistant_length", 50)
        ),
        classifier_max_recall_events_per_turn=int(
            merged.get("classifier_max_recall_events_per_turn", 20)
        ),
        recall_min_score=_optional_float(merged.get("recall_min_score")),
        recall_dialogue_min_score=_optional_float(
            merged.get("recall_dialogue_min_score")
        ),
        recall_memory_limit=recall_memory_limit,
        salient_enabled=bool(merged.get("salient_enabled", True)),
        salient_timeout_s=float(merged.get("salient_timeout_s", 1.0)),
        salient_min_query_len=int(merged.get("salient_min_query_len", 20)),
        drift_enabled=bool(merged.get("drift_enabled", True)),
        drift_threshold=float(merged.get("drift_threshold", 0.4)),
        focus_window_size=int(merged.get("focus_window_size", 3)),
        pre_llm_inject_enabled=bool(merged.get("pre_llm_inject_enabled", True)),
        self_state_enabled=bool(merged.get("self_state_enabled", True)),
        self_state_min_norm=float(merged.get("self_state_min_norm", 0.2)),
        self_state_max_age_s=float(merged.get("self_state_max_age_s", 900.0)),
        selective_enabled=bool(merged.get("selective_enabled", True)),
        selective_merge_threshold=float(
            merged.get("selective_merge_threshold", 0.92)
        ),
        selective_supersede_threshold=float(
            merged.get("selective_supersede_threshold", 0.85)
        ),
        selective_levenshtein_threshold=float(
            merged.get("selective_levenshtein_threshold", 0.3)
        ),
        selective_noise_filter=bool(
            merged.get("selective_noise_filter", True)
        ),
        selective_noise_min_length=int(
            merged.get("selective_noise_min_length", 10)
        ),
        auto_link_enabled=bool(merged.get("auto_link_enabled", True)),
        auto_link_max_distance=float(
            merged.get("auto_link_max_distance", 0.25)
        ),
        auto_link_max_links=int(merged.get("auto_link_max_links", 3)),
        store_routing_enabled=bool(merged.get("store_routing_enabled", True)),
        store_routing_limit=int(merged.get("store_routing_limit", 2400)),
        store_routing_chunk_size=int(merged.get("store_routing_chunk_size", 1600)),
        store_routing_chunk_overlap=int(merged.get("store_routing_chunk_overlap", 320)),
        store_routing_summary_chars=int(
            merged.get("store_routing_summary_chars", 1500)
        ),
        # split_part_chars вычислен и провалидирован выше — переиспользуем.
        message_split_part_chars=split_part_chars,
        message_split_inline_embed_cap=int(
            merged.get("message_split_inline_embed_cap", 4)
        ),
        document_ingest_async_chunk_threshold=int(
            merged.get("document_ingest_async_chunk_threshold", 12)
        ),
        ingest_doc_enabled=bool(merged.get("ingest_doc_enabled", True)),
        ingest_doc_roots=list(merged.get("ingest_doc_roots", []) or []),
        ingest_doc_max_bytes=int(
            merged.get("ingest_doc_max_bytes", 50 * 1024 * 1024)
        ),
        search_archive_default_limit=int(
            merged.get("search_archive_default_limit", 10)
        ),
        search_archive_max_limit=int(
            merged.get("search_archive_max_limit", 50)
        ),
        search_archive_k_candidates_factor=int(
            merged.get("search_archive_k_candidates_factor", 8)
        ),
        search_archive_k_candidates_min=int(
            merged.get("search_archive_k_candidates_min", 80)
        ),
        hebbian_enabled=bool(merged.get("hebbian_enabled", True)),
        hebbian_weight_bump=float(merged.get("hebbian_weight_bump", 0.1)),
        hebbian_initial_weight=float(merged.get("hebbian_initial_weight", 1.1)),
        hebbian_weight_max=float(merged.get("hebbian_weight_max", 2.0)),
        relation_decay_enabled=bool(merged.get("relation_decay_enabled", True)),
        relation_decay_interval_s=float(
            merged.get("relation_decay_interval_s", 3600.0)
        ),
        relation_decay_rate=float(merged.get("relation_decay_rate", 0.05)),
        relation_decay_idle_days=int(
            merged.get("relation_decay_idle_days", 14)
        ),
        turn_state_ttl_s=float(merged.get("turn_state_ttl_s", 60.0)),
        batch_consolidation_enabled=bool(
            merged.get("batch_consolidation_enabled", True)
        ),
        batch_tick_interval_s=float(
            merged.get("batch_tick_interval_s", 60.0)
        ),
        batch_trigger_messages=int(
            merged.get("batch_trigger_messages", 20)
        ),
        batch_trigger_interval_s=int(
            merged.get("batch_trigger_interval_s", 1200)
        ),
        batch_period_gap_s=int(merged.get("batch_period_gap_s", 21600)),
        batch_sentiment_enabled=bool(
            merged.get("batch_sentiment_enabled", True)
        ),
        memory_consolidation_enabled=bool(
            merged.get("memory_consolidation_enabled", True)
        ),
        memory_consolidation_tick_s=float(
            merged.get("memory_consolidation_tick_s", 3600.0)
        ),
        memory_consolidation_apply_tick_s=float(
            merged.get("memory_consolidation_apply_tick_s", 30.0)
        ),
        memory_consolidation_cooldown_h=int(
            merged.get("memory_consolidation_cooldown_h", 23)
        ),
        memory_consolidation_window_days=int(
            merged.get("memory_consolidation_window_days", 7)
        ),
        memory_consolidation_window_tail_h=int(
            merged.get("memory_consolidation_window_tail_h", 24)
        ),
        memory_consolidation_cosine=float(
            merged.get("memory_consolidation_cosine", 0.88)
        ),
        memory_consolidation_min_size=int(
            merged.get("memory_consolidation_min_size", 3)
        ),
        memory_consolidation_max_size=int(
            merged.get("memory_consolidation_max_size", 8)
        ),
        reinterpret_enabled=bool(merged.get("reinterpret_enabled", True)),
        reinterpret_apply_tick_s=float(
            merged.get("reinterpret_apply_tick_s", 30.0)
        ),
        reinterpret_cooldown_s=int(
            merged.get("reinterpret_cooldown_s", 86400)
        ),
        reinterpret_blend_weight=float(
            merged.get("reinterpret_blend_weight", 0.5)
        ),
        ingest_api_enabled=bool(merged.get("ingest_api_enabled", True)),
        dialogue_api_enabled=bool(merged.get("dialogue_api_enabled", True)),
        explain_api_enabled=bool(merged.get("explain_api_enabled", True)),
        hot_tier_enabled=bool(merged.get("hot_tier_enabled", True)),
        hot_tier_ttl_s=float(merged.get("hot_tier_ttl_s", 300.0)),
        hot_tier_lru_bound=int(merged.get("hot_tier_lru_bound", 100)),
        eviction_relevance_enabled=bool(
            merged.get("eviction_relevance_enabled", True)
        ),
        eviction_relevance_keep_k=int(
            merged.get("eviction_relevance_keep_k", 2)
        ),
        eviction_relevance_threshold=float(
            merged.get("eviction_relevance_threshold", 0.4)
        ),
        working_set_persistence_enabled=bool(
            merged.get("working_set_persistence_enabled", True)
        ),
        working_set_save_interval_s=float(
            merged.get("working_set_save_interval_s", 30.0)
        ),
        working_set_ttl_s=float(
            merged.get("working_set_ttl_s", 86400.0)
        ),
        healthz_port=int(merged.get("healthz_port", 8788)),
        healthz_bind=str(merged.get("healthz_bind", "127.0.0.1")),
        healthz_liveness_threshold_s=float(
            merged.get("healthz_liveness_threshold_s", 120.0)
        ),
        healthz_readiness_threshold_s=float(
            merged.get("healthz_readiness_threshold_s", 300.0)
        ),
        http_bind=str(merged.get("http_bind", "127.0.0.1")),
        http_port=int(merged.get("http_port", 8788)),
        http_token=_optional_str(merged.get("http_token")),
        pg_pool_min=int(merged.get("pg_pool_min", 2)),
        pg_pool_max=int(merged.get("pg_pool_max", 10)),
        daemon_url=str(merged.get("daemon_url", "http://127.0.0.1:8788")),
        log_format=str(merged.get("log_format", "text")),
        extra=extra,
    )


def _optional_str(value: Any) -> str | None:
    """None / пустое → None; иначе str()."""
    if value is None:
        return None
    s = str(value).strip()
    return s or None


def _optional_float(value: Any) -> float | None:
    """None / отсутствие → None; иначе float()."""
    if value is None:
        return None
    return float(value)


def _optional_int(value: Any) -> int | None:
    """None / отсутствие → None; иначе int()."""
    if value is None:
        return None
    return int(value)


def is_available(hermes_home: str | os.PathLike[str] | None = None) -> bool:
    """Быстрая проверка без подключения к БД (для MemoryProvider.is_available)."""
    if os.environ.get("STYX_DATABASE_URL") or os.environ.get("DATABASE_URL"):
        return True
    return bool(_read_json(hermes_home).get("database_url"))


def _read_json(hermes_home: str | os.PathLike[str] | None) -> dict[str, Any]:
    if hermes_home is None:
        return {}
    path = Path(hermes_home) / CONFIG_FILENAME
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _read_env() -> dict[str, Any]:
    return {
        "database_url": (
            os.environ.get("STYX_DATABASE_URL")
            or os.environ.get("DATABASE_URL")
        ),
        "ollama_url": os.environ.get("STYX_OLLAMA_URL"),
        "embedding_model": os.environ.get("STYX_EMBEDDING_MODEL"),
        "embedding_dim": (
            int(os.environ["STYX_EMBEDDING_DIM"])
            if os.environ.get("STYX_EMBEDDING_DIM")
            else None
        ),
        "embedding_timeout_s": (
            float(os.environ["STYX_EMBEDDING_TIMEOUT_S"])
            if os.environ.get("STYX_EMBEDDING_TIMEOUT_S")
            else None
        ),
        "llm_url": os.environ.get("STYX_LLM_URL"),
        "llm_model": os.environ.get("STYX_LLM_MODEL"),
        "llm_timeout_s": (
            float(os.environ["STYX_LLM_TIMEOUT_S"])
            if os.environ.get("STYX_LLM_TIMEOUT_S")
            else None
        ),
        "llm_max_attempts": (
            int(os.environ["STYX_LLM_MAX_ATTEMPTS"])
            if os.environ.get("STYX_LLM_MAX_ATTEMPTS")
            else None
        ),
        "llm_rate_limit_capacity": (
            int(os.environ["STYX_LLM_RATE_LIMIT_CAPACITY"])
            if os.environ.get("STYX_LLM_RATE_LIMIT_CAPACITY")
            else None
        ),
        "llm_rate_limit_refill_per_s": (
            float(os.environ["STYX_LLM_RATE_LIMIT_REFILL_PER_S"])
            if os.environ.get("STYX_LLM_RATE_LIMIT_REFILL_PER_S")
            else None
        ),
        "sweep_interval_s": (
            float(os.environ["STYX_SWEEP_INTERVAL_S"])
            if os.environ.get("STYX_SWEEP_INTERVAL_S")
            else None
        ),
        "sweep_lock_timeout_s": (
            float(os.environ["STYX_SWEEP_LOCK_TIMEOUT_S"])
            if os.environ.get("STYX_SWEEP_LOCK_TIMEOUT_S")
            else None
        ),
        "sentiment_enabled": (
            os.environ["STYX_SENTIMENT_ENABLED"].lower() not in ("0", "false", "no")
            if os.environ.get("STYX_SENTIMENT_ENABLED") is not None
            else None
        ),
        "sentiment_timeout_s": (
            float(os.environ["STYX_SENTIMENT_TIMEOUT_S"])
            if os.environ.get("STYX_SENTIMENT_TIMEOUT_S")
            else None
        ),
        "emotional_tick_interval_s": (
            float(os.environ["STYX_EMOTIONAL_TICK_INTERVAL_S"])
            if os.environ.get("STYX_EMOTIONAL_TICK_INTERVAL_S")
            else None
        ),
        "classifier_min_assistant_length": (
            int(os.environ["STYX_CLASSIFIER_MIN_ASSISTANT_LENGTH"])
            if os.environ.get("STYX_CLASSIFIER_MIN_ASSISTANT_LENGTH")
            else None
        ),
        "classifier_max_recall_events_per_turn": (
            int(os.environ["STYX_CLASSIFIER_MAX_RECALL_EVENTS_PER_TURN"])
            if os.environ.get("STYX_CLASSIFIER_MAX_RECALL_EVENTS_PER_TURN")
            else None
        ),
        "recall_min_score": (
            float(os.environ["STYX_RECALL_MIN_SCORE"])
            if os.environ.get("STYX_RECALL_MIN_SCORE")
            else None
        ),
        "recall_dialogue_min_score": (
            float(os.environ["STYX_RECALL_DIALOGUE_MIN_SCORE"])
            if os.environ.get("STYX_RECALL_DIALOGUE_MIN_SCORE")
            else None
        ),
        "recall_memory_limit": (
            int(os.environ["STYX_RECALL_MEMORY_LIMIT"])
            if os.environ.get("STYX_RECALL_MEMORY_LIMIT")
            else None
        ),
        "salient_enabled": (
            os.environ["STYX_SALIENT_ENABLED"].lower() not in ("0", "false", "no")
            if os.environ.get("STYX_SALIENT_ENABLED") is not None
            else None
        ),
        "salient_timeout_s": (
            float(os.environ["STYX_SALIENT_TIMEOUT_S"])
            if os.environ.get("STYX_SALIENT_TIMEOUT_S")
            else None
        ),
        "salient_min_query_len": (
            int(os.environ["STYX_SALIENT_MIN_QUERY_LEN"])
            if os.environ.get("STYX_SALIENT_MIN_QUERY_LEN")
            else None
        ),
        "drift_enabled": (
            os.environ["STYX_DRIFT_ENABLED"].lower() not in ("0", "false", "no")
            if os.environ.get("STYX_DRIFT_ENABLED") is not None
            else None
        ),
        "drift_threshold": (
            float(os.environ["STYX_DRIFT_THRESHOLD"])
            if os.environ.get("STYX_DRIFT_THRESHOLD")
            else None
        ),
        "focus_window_size": (
            int(os.environ["STYX_FOCUS_WINDOW_SIZE"])
            if os.environ.get("STYX_FOCUS_WINDOW_SIZE")
            else None
        ),
        "pre_llm_inject_enabled": (
            os.environ["STYX_PRE_LLM_INJECT_ENABLED"].lower() not in ("0", "false", "no")
            if os.environ.get("STYX_PRE_LLM_INJECT_ENABLED") is not None
            else None
        ),
        "self_state_enabled": (
            os.environ["STYX_SELF_STATE_ENABLED"].lower() not in ("0", "false", "no")
            if os.environ.get("STYX_SELF_STATE_ENABLED") is not None
            else None
        ),
        "self_state_min_norm": (
            float(os.environ["STYX_SELF_STATE_MIN_NORM"])
            if os.environ.get("STYX_SELF_STATE_MIN_NORM")
            else None
        ),
        "self_state_max_age_s": (
            float(os.environ["STYX_SELF_STATE_MAX_AGE_S"])
            if os.environ.get("STYX_SELF_STATE_MAX_AGE_S")
            else None
        ),
        "selective_enabled": (
            os.environ["STYX_SELECTIVE_ENABLED"].lower() not in ("0", "false", "no")
            if os.environ.get("STYX_SELECTIVE_ENABLED") is not None
            else None
        ),
        "selective_merge_threshold": (
            float(os.environ["STYX_SELECTIVE_MERGE_THRESHOLD"])
            if os.environ.get("STYX_SELECTIVE_MERGE_THRESHOLD")
            else None
        ),
        "selective_supersede_threshold": (
            float(os.environ["STYX_SELECTIVE_SUPERSEDE_THRESHOLD"])
            if os.environ.get("STYX_SELECTIVE_SUPERSEDE_THRESHOLD")
            else None
        ),
        "selective_levenshtein_threshold": (
            float(os.environ["STYX_SELECTIVE_LEVENSHTEIN_THRESHOLD"])
            if os.environ.get("STYX_SELECTIVE_LEVENSHTEIN_THRESHOLD")
            else None
        ),
        "selective_noise_filter": (
            os.environ["STYX_SELECTIVE_NOISE_FILTER"].lower() not in ("0", "false", "no")
            if os.environ.get("STYX_SELECTIVE_NOISE_FILTER") is not None
            else None
        ),
        "selective_noise_min_length": (
            int(os.environ["STYX_SELECTIVE_NOISE_MIN_LENGTH"])
            if os.environ.get("STYX_SELECTIVE_NOISE_MIN_LENGTH")
            else None
        ),
        "auto_link_enabled": (
            os.environ["STYX_AUTO_LINK_ENABLED"].lower() not in ("0", "false", "no")
            if os.environ.get("STYX_AUTO_LINK_ENABLED") is not None
            else None
        ),
        "auto_link_max_distance": (
            float(os.environ["STYX_AUTO_LINK_MAX_DISTANCE"])
            if os.environ.get("STYX_AUTO_LINK_MAX_DISTANCE")
            else None
        ),
        "auto_link_max_links": (
            int(os.environ["STYX_AUTO_LINK_MAX_LINKS"])
            if os.environ.get("STYX_AUTO_LINK_MAX_LINKS")
            else None
        ),
        "store_routing_enabled": (
            os.environ["STYX_STORE_ROUTING_ENABLED"].lower() not in ("0", "false", "no")
            if os.environ.get("STYX_STORE_ROUTING_ENABLED") is not None
            else None
        ),
        "store_routing_limit": (
            int(os.environ["STYX_STORE_ROUTING_LIMIT"])
            if os.environ.get("STYX_STORE_ROUTING_LIMIT")
            else None
        ),
        "store_routing_chunk_size": (
            int(os.environ["STYX_CHUNK_SIZE"])
            if os.environ.get("STYX_CHUNK_SIZE")
            else None
        ),
        "store_routing_chunk_overlap": (
            int(os.environ["STYX_CHUNK_OVERLAP"])
            if os.environ.get("STYX_CHUNK_OVERLAP")
            else None
        ),
        "store_routing_summary_chars": (
            int(os.environ["STYX_STORE_ROUTING_SUMMARY_CHARS"])
            if os.environ.get("STYX_STORE_ROUTING_SUMMARY_CHARS")
            else None
        ),
        "message_split_part_chars": (
            int(os.environ["STYX_MESSAGE_SPLIT_PART_CHARS"])
            if os.environ.get("STYX_MESSAGE_SPLIT_PART_CHARS")
            else None
        ),
        "message_split_inline_embed_cap": (
            int(os.environ["STYX_MESSAGE_SPLIT_INLINE_EMBED_CAP"])
            if os.environ.get("STYX_MESSAGE_SPLIT_INLINE_EMBED_CAP")
            else None
        ),
        "document_ingest_async_chunk_threshold": (
            int(os.environ["STYX_DOCUMENT_INGEST_ASYNC_CHUNK_THRESHOLD"])
            if os.environ.get("STYX_DOCUMENT_INGEST_ASYNC_CHUNK_THRESHOLD")
            else None
        ),
        "ingest_doc_enabled": (
            os.environ["STYX_INGEST_DOC_ENABLED"].lower() not in ("0", "false", "no")
            if os.environ.get("STYX_INGEST_DOC_ENABLED") is not None
            else None
        ),
        "ingest_doc_roots": (
            [p for p in os.environ["STYX_INGEST_DOC_ROOTS"].split(":") if p]
            if os.environ.get("STYX_INGEST_DOC_ROOTS")
            else None
        ),
        "ingest_doc_max_bytes": (
            int(os.environ["STYX_INGEST_DOC_MAX_BYTES"])
            if os.environ.get("STYX_INGEST_DOC_MAX_BYTES")
            else None
        ),
        "search_archive_default_limit": (
            int(os.environ["STYX_SEARCH_ARCHIVE_DEFAULT_LIMIT"])
            if os.environ.get("STYX_SEARCH_ARCHIVE_DEFAULT_LIMIT")
            else None
        ),
        "search_archive_max_limit": (
            int(os.environ["STYX_SEARCH_ARCHIVE_MAX_LIMIT"])
            if os.environ.get("STYX_SEARCH_ARCHIVE_MAX_LIMIT")
            else None
        ),
        "search_archive_k_candidates_factor": (
            int(os.environ["STYX_SEARCH_ARCHIVE_K_CANDIDATES_FACTOR"])
            if os.environ.get("STYX_SEARCH_ARCHIVE_K_CANDIDATES_FACTOR")
            else None
        ),
        "search_archive_k_candidates_min": (
            int(os.environ["STYX_SEARCH_ARCHIVE_K_CANDIDATES_MIN"])
            if os.environ.get("STYX_SEARCH_ARCHIVE_K_CANDIDATES_MIN")
            else None
        ),
        "hebbian_enabled": (
            os.environ["STYX_HEBBIAN_ENABLED"].lower() not in ("0", "false", "no")
            if os.environ.get("STYX_HEBBIAN_ENABLED") is not None
            else None
        ),
        "hebbian_weight_bump": (
            float(os.environ["STYX_HEBBIAN_WEIGHT_BUMP"])
            if os.environ.get("STYX_HEBBIAN_WEIGHT_BUMP")
            else None
        ),
        "hebbian_initial_weight": (
            float(os.environ["STYX_HEBBIAN_INITIAL_WEIGHT"])
            if os.environ.get("STYX_HEBBIAN_INITIAL_WEIGHT")
            else None
        ),
        "hebbian_weight_max": (
            float(os.environ["STYX_HEBBIAN_WEIGHT_MAX"])
            if os.environ.get("STYX_HEBBIAN_WEIGHT_MAX")
            else None
        ),
        "relation_decay_enabled": (
            os.environ["STYX_RELATION_DECAY_ENABLED"].lower() not in ("0", "false", "no")
            if os.environ.get("STYX_RELATION_DECAY_ENABLED") is not None
            else None
        ),
        "relation_decay_interval_s": (
            float(os.environ["STYX_RELATION_DECAY_INTERVAL_S"])
            if os.environ.get("STYX_RELATION_DECAY_INTERVAL_S")
            else None
        ),
        "relation_decay_rate": (
            float(os.environ["STYX_RELATION_DECAY_RATE"])
            if os.environ.get("STYX_RELATION_DECAY_RATE")
            else None
        ),
        "relation_decay_idle_days": (
            int(os.environ["STYX_RELATION_DECAY_IDLE_DAYS"])
            if os.environ.get("STYX_RELATION_DECAY_IDLE_DAYS")
            else None
        ),
        "turn_state_ttl_s": (
            float(os.environ["STYX_TURN_STATE_TTL_S"])
            if os.environ.get("STYX_TURN_STATE_TTL_S")
            else None
        ),
        "batch_consolidation_enabled": (
            os.environ["STYX_BATCH_CONSOLIDATION_ENABLED"].lower() not in ("0", "false", "no")
            if os.environ.get("STYX_BATCH_CONSOLIDATION_ENABLED") is not None
            else None
        ),
        "batch_tick_interval_s": (
            float(os.environ["STYX_BATCH_TICK_INTERVAL_S"])
            if os.environ.get("STYX_BATCH_TICK_INTERVAL_S")
            else None
        ),
        "batch_trigger_messages": (
            int(os.environ["STYX_BATCH_TRIGGER_MESSAGES"])
            if os.environ.get("STYX_BATCH_TRIGGER_MESSAGES")
            else None
        ),
        "batch_trigger_interval_s": (
            int(os.environ["STYX_BATCH_TRIGGER_INTERVAL_S"])
            if os.environ.get("STYX_BATCH_TRIGGER_INTERVAL_S")
            else None
        ),
        "batch_period_gap_s": (
            int(os.environ["STYX_BATCH_PERIOD_GAP_S"])
            if os.environ.get("STYX_BATCH_PERIOD_GAP_S")
            else None
        ),
        "batch_sentiment_enabled": (
            os.environ["STYX_BATCH_SENTIMENT_ENABLED"].lower() not in ("0", "false", "no")
            if os.environ.get("STYX_BATCH_SENTIMENT_ENABLED") is not None
            else None
        ),
        "memory_consolidation_enabled": (
            os.environ["STYX_MEMORY_CONSOLIDATION_ENABLED"].lower() not in ("0", "false", "no")
            if os.environ.get("STYX_MEMORY_CONSOLIDATION_ENABLED") is not None
            else None
        ),
        "memory_consolidation_tick_s": (
            float(os.environ["STYX_MEMORY_CONSOLIDATION_TICK_S"])
            if os.environ.get("STYX_MEMORY_CONSOLIDATION_TICK_S")
            else None
        ),
        "memory_consolidation_apply_tick_s": (
            float(os.environ["STYX_MEMORY_CONSOLIDATION_APPLY_TICK_S"])
            if os.environ.get("STYX_MEMORY_CONSOLIDATION_APPLY_TICK_S")
            else None
        ),
        "memory_consolidation_cooldown_h": (
            int(os.environ["STYX_MEMORY_CONSOLIDATION_COOLDOWN_H"])
            if os.environ.get("STYX_MEMORY_CONSOLIDATION_COOLDOWN_H")
            else None
        ),
        "memory_consolidation_window_days": (
            int(os.environ["STYX_MEMORY_CONSOLIDATION_WINDOW_DAYS"])
            if os.environ.get("STYX_MEMORY_CONSOLIDATION_WINDOW_DAYS")
            else None
        ),
        "memory_consolidation_window_tail_h": (
            int(os.environ["STYX_MEMORY_CONSOLIDATION_WINDOW_TAIL_H"])
            if os.environ.get("STYX_MEMORY_CONSOLIDATION_WINDOW_TAIL_H")
            else None
        ),
        "memory_consolidation_cosine": (
            float(os.environ["STYX_MEMORY_CONSOLIDATION_COSINE"])
            if os.environ.get("STYX_MEMORY_CONSOLIDATION_COSINE")
            else None
        ),
        "memory_consolidation_min_size": (
            int(os.environ["STYX_MEMORY_CONSOLIDATION_MIN_SIZE"])
            if os.environ.get("STYX_MEMORY_CONSOLIDATION_MIN_SIZE")
            else None
        ),
        "memory_consolidation_max_size": (
            int(os.environ["STYX_MEMORY_CONSOLIDATION_MAX_SIZE"])
            if os.environ.get("STYX_MEMORY_CONSOLIDATION_MAX_SIZE")
            else None
        ),
        "reinterpret_enabled": (
            os.environ["STYX_REINTERPRET_ENABLED"].lower() not in ("0", "false", "no")
            if os.environ.get("STYX_REINTERPRET_ENABLED") is not None
            else None
        ),
        "reinterpret_apply_tick_s": (
            float(os.environ["STYX_REINTERPRET_APPLY_TICK_S"])
            if os.environ.get("STYX_REINTERPRET_APPLY_TICK_S")
            else None
        ),
        "reinterpret_cooldown_s": (
            int(os.environ["STYX_REINTERPRET_COOLDOWN_S"])
            if os.environ.get("STYX_REINTERPRET_COOLDOWN_S")
            else None
        ),
        "reinterpret_blend_weight": (
            float(os.environ["STYX_REINTERPRET_BLEND_WEIGHT"])
            if os.environ.get("STYX_REINTERPRET_BLEND_WEIGHT")
            else None
        ),
        "ingest_api_enabled": (
            os.environ["STYX_INGEST_API_ENABLED"].lower() not in ("0", "false", "no")
            if os.environ.get("STYX_INGEST_API_ENABLED") is not None
            else None
        ),
        "dialogue_api_enabled": (
            os.environ["STYX_DIALOGUE_API_ENABLED"].lower() not in ("0", "false", "no")
            if os.environ.get("STYX_DIALOGUE_API_ENABLED") is not None
            else None
        ),
        "explain_api_enabled": (
            os.environ["STYX_EXPLAIN_API_ENABLED"].lower() not in ("0", "false", "no")
            if os.environ.get("STYX_EXPLAIN_API_ENABLED") is not None
            else None
        ),
        "hot_tier_enabled": (
            os.environ["STYX_HOT_TIER_ENABLED"].lower() not in ("0", "false", "no")
            if os.environ.get("STYX_HOT_TIER_ENABLED") is not None
            else None
        ),
        "hot_tier_ttl_s": (
            float(os.environ["STYX_HOT_TIER_TTL_S"])
            if os.environ.get("STYX_HOT_TIER_TTL_S")
            else None
        ),
        "hot_tier_lru_bound": (
            int(os.environ["STYX_HOT_TIER_LRU_BOUND"])
            if os.environ.get("STYX_HOT_TIER_LRU_BOUND")
            else None
        ),
        "eviction_relevance_enabled": (
            os.environ["STYX_EVICTION_RELEVANCE_ENABLED"].lower()
            not in ("0", "false", "no")
            if os.environ.get("STYX_EVICTION_RELEVANCE_ENABLED") is not None
            else None
        ),
        "eviction_relevance_keep_k": (
            int(os.environ["STYX_EVICTION_RELEVANCE_KEEP_K"])
            if os.environ.get("STYX_EVICTION_RELEVANCE_KEEP_K")
            else None
        ),
        "eviction_relevance_threshold": (
            float(os.environ["STYX_EVICTION_RELEVANCE_THRESHOLD"])
            if os.environ.get("STYX_EVICTION_RELEVANCE_THRESHOLD")
            else None
        ),
        "working_set_persistence_enabled": (
            os.environ["STYX_WORKING_SET_PERSISTENCE_ENABLED"].lower()
            not in ("0", "false", "no")
            if os.environ.get("STYX_WORKING_SET_PERSISTENCE_ENABLED") is not None
            else None
        ),
        "working_set_save_interval_s": (
            float(os.environ["STYX_WORKING_SET_SAVE_INTERVAL_S"])
            if os.environ.get("STYX_WORKING_SET_SAVE_INTERVAL_S")
            else None
        ),
        "working_set_ttl_s": (
            float(os.environ["STYX_WORKING_SET_TTL_S"])
            if os.environ.get("STYX_WORKING_SET_TTL_S")
            else None
        ),
        "healthz_port": (
            int(os.environ["STYX_HEALTHZ_PORT"])
            if os.environ.get("STYX_HEALTHZ_PORT")
            else None
        ),
        "healthz_bind": os.environ.get("STYX_HEALTHZ_BIND"),
        "healthz_liveness_threshold_s": (
            float(os.environ["STYX_HEALTHZ_LIVENESS_THRESHOLD_S"])
            if os.environ.get("STYX_HEALTHZ_LIVENESS_THRESHOLD_S")
            else None
        ),
        "healthz_readiness_threshold_s": (
            float(os.environ["STYX_HEALTHZ_READINESS_THRESHOLD_S"])
            if os.environ.get("STYX_HEALTHZ_READINESS_THRESHOLD_S")
            else None
        ),
        "log_format": os.environ.get("STYX_LOG_FORMAT"),
        "http_bind": os.environ.get("STYX_HTTP_BIND"),
        "http_port": (
            int(os.environ["STYX_HTTP_PORT"])
            if os.environ.get("STYX_HTTP_PORT")
            else None
        ),
        "http_token": os.environ.get("STYX_HTTP_TOKEN"),
        "pg_pool_min": (
            int(os.environ["STYX_PG_POOL_MIN"])
            if os.environ.get("STYX_PG_POOL_MIN")
            else None
        ),
        "pg_pool_max": (
            int(os.environ["STYX_PG_POOL_MAX"])
            if os.environ.get("STYX_PG_POOL_MAX")
            else None
        ),
        "daemon_url": os.environ.get("STYX_DAEMON_URL"),
    }
