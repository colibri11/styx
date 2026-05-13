"""Agent-scoped query wrapper для Styx storage.

Каждый StyxMemoryProvider знает свой ``agent_id`` ровно один — это
process-level константа от Hermes. AgentScopedQueries инкапсулирует
это знание и автоматически инжектит ``agent_id`` во все запросы
к ``memories`` / ``sessions``.

Это единственная точка доступа к storage из остального кода Styx.
Прямые ``cur.execute`` на ``memories`` / ``sessions`` запрещены —
тесты ловят это в ``test_queries_have_no_raw_sql``.
"""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Iterator

if TYPE_CHECKING:
    from styx.engine.auto_link import AutoLinkNeighbor
    from styx.engine.selective_gatekeeper import Candidate
    from styx.turn_state import RecallSnapshot

import psycopg
from psycopg.rows import dict_row

from .recall_config import DEFAULT_RECALL_CONFIG, FullRecallConfig
from .scoring import (
    BuildFactorExprsOptions,
    DecayConfig,
    EmotionalBaseline,
    build_factor_exprs,
)
from .search_weights import SearchConfig


@dataclass(frozen=True)
class StoredMessage:
    id: uuid.UUID
    session_id: uuid.UUID | None
    role: str
    content: str
    metadata: dict[str, Any]
    created_at: Any  # datetime — typing-проблема psycopg, оставляем Any


@dataclass(frozen=True)
class MemoryHit:
    """Результат vector/hybrid search'а — одна memory с её scoring-полями.

    ``score`` — composite final (см. scoring.build_factor_exprs).
    ``match_score`` — только base_match (vector cosine sim или hybrid),
    отдельно для INSERT в recall_events (memorybox-семантика).
    ``embedding`` populated только когда search_similar вызван с
    ``include_embedding=True`` (для internal_dedup).
    """

    id: uuid.UUID
    agent_id: str
    kind: str
    role: str
    content: str
    metadata: dict[str, Any]
    created_at: Any
    score: float
    match_score: float
    embedding: list[float] | None = field(default=None)
    recall_event_id: int | None = field(default=None)
    kind_src: str | None = field(default=None)


@dataclass(frozen=True)
class ChunkHit:
    """Результат `search_chunks_for_archive` (волна 20).

    Hybrid score: ``vector_weight * (1 - cosine) + bm25_weight * ts_rank``,
    веса через ``search_weights.compute_weights(query_text)``.
    """

    document_id: uuid.UUID
    position: int
    content: str
    char_start: int
    char_end: int
    score: float


@dataclass(frozen=True)
class DialogueHit:
    """Результат `search_dialogue_for_archive` (волна 20).

    Источник — `memories WHERE role IN ('user','assistant')` (Styx-
    семантика реплик после волны 14). Hybrid score через те же веса.
    """

    memory_id: uuid.UUID
    role: str
    content: str
    created_at: Any
    score: float


@dataclass(frozen=True)
class DialogueSearchHit:
    """Результат `dialogue_search` (волна 24).

    Hybrid (FTS+vector) либо pure-vector в зависимости от наличия
    ``query_text``. ``score`` — итоговый: для hybrid — взвешенная сумма
    cosine_sim+ts_rank; для pure-vector — `1 - distance` ∈ [0..1].
    """

    memory_id: uuid.UUID
    role: str
    content: str
    created_at: Any
    score: float
    session_id: uuid.UUID | None = None


@dataclass(frozen=True)
class DialogueRecentRow:
    """Строка `dialogue_recent` (волна 24)."""

    memory_id: uuid.UUID
    role: str
    content: str
    created_at: Any
    session_id: uuid.UUID | None = None


@dataclass(frozen=True)
class DialogueSessionInfo:
    """Строка `dialogue_list_sessions` (волна 24)."""

    session_id: uuid.UUID
    message_count: int
    first_message_at: Any
    last_message_at: Any


@dataclass(frozen=True)
class DialogueTranscriptRow:
    """Строка `dialogue_prepare_summary` (волна 24)."""

    role: str
    content: str
    created_at: Any


class AgentScopedQueries:
    """Все CRUD-операции для одного агента.

    Создаётся StyxMemoryProvider'ом сразу после ``initialize`` и живёт
    столько же сколько процесс. ``agent_id`` нельзя поменять.
    """

    def __init__(self, conn: psycopg.Connection, agent_id: str) -> None:
        if not agent_id:
            raise ValueError("agent_id обязателен и не может быть пустым")
        self._conn = conn
        self._agent_id = agent_id

    @property
    def agent_id(self) -> str:
        return self._agent_id

    @property
    def conn(self) -> psycopg.Connection:
        """Низкоуровневый доступ к connection — для слоёв, которым нужен
        прямой psycopg API (recall→read_baseline_for_scoring и т.д.).
        Внешний код через ``queries.conn`` обязан помнить про agent_id
        scope."""
        return self._conn

    # -- sessions ---------------------------------------------------------

    def upsert_session(self, session_id: uuid.UUID | str) -> None:
        """Регистрирует session_id. Идемпотентен — повторный старт не падает.

        Не делает commit — вызывающий код управляет транзакционной границей.
        """
        sid = _as_uuid(session_id)
        with self._conn.cursor() as cur:
            cur.execute(
                "INSERT INTO sessions (id, agent_id) VALUES (%s, %s) "
                "ON CONFLICT (id) DO NOTHING",
                (sid, self._agent_id),
            )

    def end_session(self, session_id: uuid.UUID | str) -> None:
        """Не делает commit — вызывающий код управляет транзакционной границей."""
        sid = _as_uuid(session_id)
        with self._conn.cursor() as cur:
            cur.execute(
                "UPDATE sessions SET ended_at = now() "
                "WHERE id = %s AND agent_id = %s AND ended_at IS NULL",
                (sid, self._agent_id),
            )

    # -- memories (write) -------------------------------------------------

    def insert_message(
        self,
        *,
        role: str,
        content: str,
        session_id: uuid.UUID | str | None = None,
        embedding: list[float] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> uuid.UUID:
        sid = _as_uuid(session_id) if session_id is not None else None
        meta = metadata or {}
        with self._conn.cursor() as cur:
            cur.execute(
                "INSERT INTO memories "
                "(agent_id, session_id, role, content, embedding, metadata) "
                "VALUES (%s, %s, %s, %s, %s, %s) "
                "RETURNING id",
                (self._agent_id, sid, role, content, _vector_literal(embedding), psycopg.types.json.Jsonb(meta)),
            )
            row = cur.fetchone()
        # Не делаем commit — вызывающий код управляет транзакционной границей.
        if row is None:
            raise RuntimeError(
                "INSERT INTO memories не вернул id — возможно constraint violation"
            )
        return row[0]

    def lookup_embeddings_by_content(
        self, contents: list[str]
    ) -> dict[str, list[float]]:
        """Batch lookup embedding'ов в ``memories`` по тексту content'а.

        Используется волной 12 (eviction relevance-aware) для получения
        embed'ов body-сообщений compress'а без re-embed'а в Ollama.
        ``embed-after-commit`` hook'а ``sync_turn`` (волна 7) гарантирует,
        что user/assistant turn'ы имеют записанный embedding в БД.

        Возвращает map content → embedding только для рядов, у которых
        ``embedding IS NOT NULL`` и matching ``agent_id`` (scope).
        Multiple ряды с одинаковым content (могут быть при повторении
        реплики) сворачиваются через ``DISTINCT ON (content)`` —
        выбирается самая свежая (``seq DESC``); embedding для одного
        и того же текста при стабильной модели одинаков.

        Пустой ``contents`` → пустой dict без БД-запроса.
        """
        if not contents:
            return {}
        with self._conn.cursor() as cur:
            cur.execute(
                "SELECT DISTINCT ON (content) content, embedding "
                "FROM memories "
                "WHERE agent_id = %s "
                "  AND content = ANY(%s) "
                "  AND embedding IS NOT NULL "
                "ORDER BY content, seq DESC",
                (self._agent_id, contents),
            )
            rows = cur.fetchall()
        out: dict[str, list[float]] = {}
        for content, raw_vec in rows:
            vec = _parse_vector(raw_vec)
            if vec is not None:
                out[content] = vec
        return out

    # -- memories (read) --------------------------------------------------

    def recent_messages(
        self,
        *,
        limit: int,
        session_id: uuid.UUID | str | None = None,
    ) -> list[StoredMessage]:
        params: list[Any] = [self._agent_id]
        sql = (
            "SELECT id, session_id, role, content, metadata, created_at "
            "FROM memories WHERE agent_id = %s"
        )
        if session_id is not None:
            sql += " AND session_id = %s"
            params.append(_as_uuid(session_id))
        # ORDER BY seq — монотонный bigserial, не зависит от microsecond-
        # совпадений created_at внутри одной транзакции (sync_turn пишет
        # user+assistant одной tx). DESC — последние сверху.
        sql += " ORDER BY seq DESC LIMIT %s"
        params.append(limit)

        with self._conn.cursor(row_factory=dict_row) as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()

        return [
            StoredMessage(
                id=r["id"],
                session_id=r["session_id"],
                role=r["role"],
                content=r["content"],
                metadata=r["metadata"],
                created_at=r["created_at"],
            )
            for r in rows
        ]

    def count_messages(self, *, session_id: uuid.UUID | str | None = None) -> int:
        params: list[Any] = [self._agent_id]
        sql = "SELECT count(*) FROM memories WHERE agent_id = %s"
        if session_id is not None:
            sql += " AND session_id = %s"
            params.append(_as_uuid(session_id))
        with self._conn.cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchone()[0]

    # -- recall (search + tracking) ---------------------------------------

    def search_similar(
        self,
        *,
        query_vector: list[float],
        query_text: str | None = None,
        limit: int,
        full_config: FullRecallConfig = DEFAULT_RECALL_CONFIG.full,
        search_config: SearchConfig | None = None,
        decay_config: DecayConfig | None = None,
        usage_norm_p75: float = 0.0,
        emotional_baseline: EmotionalBaseline | None = None,
        include_embedding: bool = False,
        snapshot: "RecallSnapshot | None" = None,
    ) -> list[MemoryHit]:
        """Composite-scored top-K по cosine similarity (+ опц. BM25 hybrid).

        SQL формула — port ``buildFactorExprs`` (memorybox memory.ts:454).
        Фильтр ``min_score`` применяется на Python-стороне после получения
        результата (как в memorybox ``recall/full.ts``); этот метод
        возвращает все limit рядов в score-DESC, caller отфильтровывает.

        ``query_text`` не None → активируется hybrid режим (vector + BM25
        через content_tsv GIN). Adaptive weights — внутри build_factor_exprs.

        ``include_embedding=True`` тащит embedding в результаты для
        последующего internal_dedup. По умолчанию выключено — лишние
        768 floats на запись.

        Memories без embedding (NULL) исключаются. ``superseded_by IS NULL``
        — только живые.
        """
        text_query_param_index = 2 if query_text else None
        opts = BuildFactorExprsOptions(
            text_query_param_index=text_query_param_index,
            search_config=search_config,
            decay_config=decay_config,
            table_alias="m",
            usage_norm_p75=usage_norm_p75,
            emotional_baseline=emotional_baseline,
        )
        inp: dict[str, object] = {"text_query": query_text} if query_text else {}
        factors = build_factor_exprs(inp, opts)

        embedding_select = "m.embedding" if include_embedding else "NULL::vector"

        # Волна 14 (10a): snapshot fence — отсекаем non-subjective memories
        # появившиеся после cycle_start. Subjective sync_turn записи видны
        # сразу («я положил, я помню»). Без snapshot'а — полный фильтр
        # как до волны 14.
        snapshot_clause = ""
        if snapshot is not None:
            snapshot_clause = (
                " AND (m.created_at <= %(cycle_start)s "
                "OR (m.kind_src IN ('subjective','subjective_tail') "
                "AND m.agent_id = %(snapshot_agent_id)s))"
            )

        sql = f"""
            SELECT
                m.id,
                m.agent_id,
                m.kind,
                m.kind_src,
                m.role,
                m.content,
                m.metadata,
                m.created_at,
                ({factors.score_expr}) AS score,
                ({factors.base_match_expr}) AS match_score,
                {embedding_select} AS embedding
            FROM memories m
            {factors.usage_lateral_from}
            WHERE m.agent_id = %(agent_id)s
              AND m.embedding IS NOT NULL
              AND m.superseded_by IS NULL
              {snapshot_clause}
            ORDER BY score DESC
            LIMIT %(lim)s
        """
        # build_factor_exprs использует pg-style $1/$2 (TS-семантика).
        # psycopg не поддерживает $N — конвертируем в named, чтобы один
        # литерал привязывался к нескольким вхождениям ($1 встречается
        # как минимум дважды: внутри score_expr и в base_match_expr).
        sql_pg = sql.replace("$1", "%(qvec)s").replace("$2", "%(qtext)s")

        params: dict[str, Any] = {
            "qvec": _vector_literal(query_vector),
            "agent_id": self._agent_id,
            "lim": limit,
        }
        if query_text:
            params["qtext"] = query_text
        if snapshot is not None:
            params["cycle_start"] = snapshot.cycle_start
            params["snapshot_agent_id"] = snapshot.agent_id

        with self._conn.cursor(row_factory=dict_row) as cur:
            cur.execute(sql_pg, params)
            rows = cur.fetchall()

        return [
            MemoryHit(
                id=r["id"],
                agent_id=r["agent_id"],
                kind=r["kind"],
                kind_src=r["kind_src"],
                role=r["role"],
                content=r["content"],
                metadata=r["metadata"],
                created_at=r["created_at"],
                score=float(r["score"]),
                match_score=float(r["match_score"]),
                embedding=_parse_vector(r["embedding"]) if include_embedding else None,
            )
            for r in rows
        ]

    def record_recall_event(
        self,
        *,
        memory_id: uuid.UUID,
        query_hash: bytes,
        match_score: float,
        session_id: uuid.UUID | str | None = None,
    ) -> int:
        """UPSERT recall_events. Дедуп по UNIQUE (memory_id, query_hash)
        partial WHERE query_hash IS NOT NULL.

        На повторный вызов с тем же (memory_id, query_hash) — обновляем
        ``matched_at = now()`` и пишем актуальный ``match_score``.

        Возвращает bigserial ``recall_events.id`` — нужен для волны 7c
        (RecallTracker → классификатор).

        Не делает commit — caller управляет транзакцией.
        """
        sid = _as_uuid(session_id) if session_id is not None else None
        with self._conn.cursor() as cur:
            cur.execute(
                "INSERT INTO recall_events "
                "(memory_id, session_id, query_hash, match_score) "
                "VALUES (%s, %s, %s, %s) "
                "ON CONFLICT (memory_id, query_hash) "
                "WHERE query_hash IS NOT NULL "
                "DO UPDATE SET matched_at = now(), match_score = EXCLUDED.match_score "
                "RETURNING id",
                (memory_id, sid, query_hash, match_score),
            )
            row = cur.fetchone()
        if row is None:
            raise RuntimeError(
                "INSERT INTO recall_events не вернул id — необъяснимо"
            )
        return int(row[0])

    def enqueue_classification(
        self,
        *,
        recall_event_ids: list[int],
        llm_output_text: str,
    ) -> None:
        """Атомарно: UPDATE classifier_run_at = now() для recall_events
        + INSERT новой задачи в llm_tasks (task_type='usage_classification').

        Idempotency: UPDATE guard'ит ``classifier_run_at IS NULL``, так
        что повторный enqueue для уже classified ids не пропускает их в
        payload — но в payload они всё равно идут (handler найдёт что
        они уже classified и no-op'нет в reconcile-фазе). См. ADR § 20.

        Не делает commit.
        """
        if not recall_event_ids:
            return
        from psycopg.types.json import Jsonb

        with self._conn.cursor() as cur:
            cur.execute(
                "UPDATE recall_events SET classifier_run_at = now() "
                " WHERE id = ANY(%s::bigint[]) AND classifier_run_at IS NULL",
                (recall_event_ids,),
            )
            cur.execute(
                "INSERT INTO llm_tasks (task_type, payload) "
                "VALUES ('usage_classification', %s)",
                (
                    Jsonb(
                        {
                            "recall_event_ids": recall_event_ids,
                            "llm_output_text": llm_output_text,
                            "agent_id": self._agent_id,
                        }
                    ),
                ),
            )

    def update_last_accessed_at(self, memory_ids: list[uuid.UUID]) -> int:
        """Batch UPDATE last_accessed_at = now() для возвращённых из recall'а
        memories. Используется ``recall_full`` чтобы lifecycle sweep
        видел реальную idleness — без этого settled memories никогда не
        переходили бы в dormant.

        Возвращает rowcount. Не делает commit.
        """
        if not memory_ids:
            return 0
        with self._conn.cursor() as cur:
            cur.execute(
                "UPDATE memories SET last_accessed_at = now() "
                " WHERE id = ANY(%s::uuid[]) AND agent_id = %s",
                (memory_ids, self._agent_id),
            )
            return cur.rowcount or 0

    def update_embedding(
        self, memory_id: uuid.UUID, embedding: list[float]
    ) -> None:
        """Записывает вектор в memories.embedding. Используется
        embed-after-commit hook'ом sync_turn'а.

        Не делает commit — caller управляет транзакцией.
        """
        with self._conn.cursor() as cur:
            cur.execute(
                "UPDATE memories SET embedding = %s "
                "WHERE id = %s AND agent_id = %s",
                (_vector_literal(embedding), memory_id, self._agent_id),
            )

    # -- subjective writes + selective gatekeeper (волна 17) ------------

    def insert_memory(
        self,
        *,
        role: str,
        content: str,
        kind: str,
        kind_src: str,
        session_id: uuid.UUID | str | None = None,
        embedding: list[float] | None = None,
        metadata: dict[str, Any] | None = None,
        importance_provisional: float | None = None,
        archive_ref: dict[str, Any] | None = None,
    ) -> uuid.UUID:
        """INSERT subjective memory с явными ``kind`` / ``kind_src``.

        Отделено от ``insert_message`` (dialogue capture) — у этих write
        path'ов разные семантика и call sites. ``insert_message``
        пишет с DEFAULT'ами kind='episode', kind_src='subjective'; здесь
        кaller указывает их явно (для /memory_store, future
        consolidation handlers, etc).

        ``archive_ref`` (волна 19) — указатель на ``documents.id`` для
        tail-memory'и при store-routing. Schema:
        ``{"kind": "document", "id": "<uuid>", "locator":
        "styx://store/<uuid>", "snippet": "<до 1000 chars>"}``. Без
        archive_ref ряд записан как полный subjective (≤ 2400 chars).

        Не делает commit — caller управляет транзакцией.
        """
        sid = _as_uuid(session_id) if session_id is not None else None
        meta = metadata or {}
        cols = [
            "agent_id", "session_id", "role", "kind", "kind_src",
            "content", "embedding", "metadata",
        ]
        params: list[Any] = [
            self._agent_id, sid, role, kind, kind_src,
            content, _vector_literal(embedding),
            psycopg.types.json.Jsonb(meta),
        ]
        if importance_provisional is not None:
            cols.append("importance_provisional")
            params.append(float(importance_provisional))
        if archive_ref is not None:
            cols.append("archive_ref")
            params.append(psycopg.types.json.Jsonb(archive_ref))
        placeholders = ", ".join(["%s"] * len(cols))
        sql = (
            f"INSERT INTO memories ({', '.join(cols)}) "
            f"VALUES ({placeholders}) RETURNING id"
        )
        with self._conn.cursor() as cur:
            cur.execute(sql, tuple(params))
            row = cur.fetchone()
        if row is None:
            raise RuntimeError(
                "insert_memory не вернул id — возможно constraint violation"
            )
        return row[0]

    def find_gatekeeper_candidates(
        self,
        embedding: list[float],
        *,
        top_k: int = 5,
        max_cosine_distance: float,
        exclude_id: uuid.UUID | None = None,
    ) -> list["Candidate"]:
        """Top-K соседей текущего агента по pgvector cosine distance.

        Фильтр: ``superseded_by IS NULL AND embedding IS NOT NULL``.
        ORDER BY ``embedding <=> qvec ASC, created_at ASC`` —
        детерминированный tie-break (D6: старший ряд побеждает на равных
        расстояниях).

        ``exclude_id`` — исключить конкретный memory_id из кандидатов.
        Используется в gatekeeper apply: новый ряд уже INSERT'нут (но
        не commit'нут — виден в той же транзакции), его нельзя
        предлагать как «существующего соседа» себе же.

        Возвращает только candidates с ``cosine_distance <
        max_cosine_distance`` (т.е. similarity > supersede_threshold).
        Если в зоне нет никого — пустой список.
        """
        from styx.engine.selective_gatekeeper import Candidate as _Candidate
        qvec = _vector_literal(embedding)
        sql = (
            "SELECT id, content, "
            "       (embedding <=> %s::vector) AS cosine_distance "
            "  FROM memories "
            " WHERE agent_id = %s "
            "   AND superseded_by IS NULL "
            "   AND embedding IS NOT NULL "
        )
        params: list[Any] = [qvec, self._agent_id]
        if exclude_id is not None:
            sql += "   AND id <> %s "
            params.append(exclude_id)
        sql += (
            " ORDER BY embedding <=> %s::vector ASC, created_at ASC "
            " LIMIT %s"
        )
        params.extend([qvec, top_k])
        with self._conn.cursor() as cur:
            cur.execute(sql, tuple(params))
            rows = cur.fetchall()
        out: list[_Candidate] = []
        for mid, content, distance in rows:
            d = float(distance)
            if d >= max_cosine_distance:
                continue
            out.append(_Candidate(id=mid, content=content, cosine_distance=d))
        return out

    def apply_gatekeeper_skip(self, memory_id: uuid.UUID) -> None:
        """Skip-action: DELETE relations с этим memory + DELETE memory.

        Не делает commit — caller управляет транзакцией.
        """
        with self._conn.cursor() as cur:
            cur.execute(
                "DELETE FROM relations "
                " WHERE (source_type = 'memory' AND source_id = %s) "
                "    OR (target_type = 'memory' AND target_id = %s)",
                (memory_id, memory_id),
            )
            cur.execute(
                "DELETE FROM memories "
                " WHERE id = %s AND agent_id = %s",
                (memory_id, self._agent_id),
            )

    def apply_gatekeeper_merge(
        self,
        *,
        new_id: uuid.UUID,
        existing_id: uuid.UUID,
        new_content: str,
        new_embedding: list[float],
    ) -> None:
        """Merge-action: redirect relations new→existing, обновить existing
        если new длиннее, удалить new.

        Memorybox condition `length(new) > length(existing)` сохранён —
        предпочтение более полного текста. Если new короче — existing
        остаётся как был, но relations всё равно перенаправляются.

        Не делает commit — caller управляет транзакцией.
        """
        with self._conn.cursor() as cur:
            cur.execute(
                "UPDATE memories "
                "   SET content = %s, embedding = %s, updated_at = now() "
                " WHERE id = %s AND agent_id = %s "
                "   AND length(%s) > length(content)",
                (
                    new_content, _vector_literal(new_embedding),
                    existing_id, self._agent_id, new_content,
                ),
            )
            cur.execute(
                "UPDATE relations SET source_id = %s "
                " WHERE source_type = 'memory' AND source_id = %s",
                (existing_id, new_id),
            )
            cur.execute(
                "UPDATE relations SET target_id = %s "
                " WHERE target_type = 'memory' AND target_id = %s",
                (existing_id, new_id),
            )
            cur.execute(
                "DELETE FROM memories "
                " WHERE id = %s AND agent_id = %s",
                (new_id, self._agent_id),
            )

    def apply_gatekeeper_supersede(
        self,
        *,
        new_id: uuid.UUID,
        existing_id: uuid.UUID,
        new_embedding: list[float],
    ) -> None:
        """Supersede-action: новый получает embedding (если ещё не было),
        existing.superseded_by = new_id, INSERT relation 'supersedes'.

        Не делает commit — caller управляет транзакцией.
        """
        with self._conn.cursor() as cur:
            cur.execute(
                "UPDATE memories SET embedding = %s "
                " WHERE id = %s AND agent_id = %s",
                (_vector_literal(new_embedding), new_id, self._agent_id),
            )
            cur.execute(
                "UPDATE memories SET superseded_by = %s "
                " WHERE id = %s AND agent_id = %s",
                (new_id, existing_id, self._agent_id),
            )
            cur.execute(
                "INSERT INTO relations "
                "  (source_type, source_id, target_type, target_id, relation, weight) "
                "VALUES ('memory', %s, 'memory', %s, 'supersedes', 1.0) "
                "ON CONFLICT DO NOTHING",
                (new_id, existing_id),
            )

    # -- auto-link при INSERT (волна 18) ---------------------------------

    def find_auto_link_candidates(
        self,
        embedding: list[float],
        *,
        max_distance: float,
        max_links: int,
        exclude_id: uuid.UUID,
    ) -> list["AutoLinkNeighbor"]:
        """Top-K соседей по cosine distance для auto-link рёбер.

        **Cross-agent** (D2 в waves/18 wave-doc): нет фильтра по
        ``agent_id`` — рёбра auto-link связывают memories разных
        агентов (общий пул знаний). Это единственное cross-agent
        место в Styx.

        Фильтры:
        - ``id <> exclude_id`` — не связываем ряд с самим собой
          (только-что INSERT'нутый виден в той же транзакции).
        - ``superseded_by IS NULL`` — superseded ряды не candidates.
        - ``embedding IS NOT NULL`` — без embedding'а нет similarity.
        - ``cosine_distance < max_distance`` — порог из config'а
          (default 0.25, т.е. similarity ≥ 0.75).

        ORDER BY ``embedding <=> qvec ASC, created_at ASC`` — tie-break
        как в волне 17 (D8).
        """
        from styx.engine.auto_link import AutoLinkNeighbor as _N
        qvec = _vector_literal(embedding)
        with self._conn.cursor() as cur:
            cur.execute(
                "SELECT id, (embedding <=> %s::vector) AS cosine_distance "
                "  FROM memories "
                " WHERE id <> %s "
                "   AND superseded_by IS NULL "
                "   AND embedding IS NOT NULL "
                "   AND (embedding <=> %s::vector) < %s "
                " ORDER BY embedding <=> %s::vector ASC, created_at ASC "
                " LIMIT %s",
                (qvec, exclude_id, qvec, max_distance, qvec, max_links),
            )
            rows = cur.fetchall()
        return [_N(id=mid, cosine_distance=float(d)) for mid, d in rows]

    def insert_auto_link_relations(
        self,
        memory_id: uuid.UUID,
        neighbors: list["AutoLinkNeighbor"],
    ) -> None:
        """Batch INSERT ``related_to`` рёбер ``memory_id → neighbor.id``.

        Идемпотентно через UNIQUE constraint ``relations_unique``
        (миграция 0004) + ``ON CONFLICT DO NOTHING``. Двойной call с
        одинаковыми (memory_id, neighbor_id) не создаёт дубль.

        Не делает commit — caller управляет транзакцией.
        """
        if not neighbors:
            return
        with self._conn.cursor() as cur:
            for n in neighbors:
                cur.execute(
                    "INSERT INTO relations "
                    "  (source_type, source_id, target_type, target_id, relation) "
                    "VALUES ('memory', %s, 'memory', %s, 'related_to') "
                    "ON CONFLICT DO NOTHING",
                    (memory_id, n.id),
                )

    # -- relations API + Hebbian (волна 21) ------------------------------

    def upsert_co_retrieved_pair(
        self,
        *,
        source_id: uuid.UUID,
        target_id: uuid.UUID,
        initial_weight: float,
        weight_bump: float,
        weight_max: float,
    ) -> None:
        """UPSERT ``co_retrieved`` ребра между двумя memories.

        Если ребра нет — INSERT с ``weight = initial_weight`` (1.1 в
        дефолте) и ``metadata.last_reinforced = now()``. Если есть —
        UPDATE: ``weight = LEAST(weight + bump, weight_max)``,
        ``metadata.last_reinforced = now()``.

        Идемпотентность через UNIQUE constraint ``relations_unique``
        (миграция 0004 от волны 18).

        Не делает commit — caller управляет транзакцией.
        """
        with self._conn.cursor() as cur:
            cur.execute(
                "INSERT INTO relations "
                "  (source_type, source_id, target_type, target_id, "
                "   relation, weight, metadata) "
                "VALUES ('memory', %s, 'memory', %s, 'co_retrieved', %s, "
                "        jsonb_build_object("
                "            'last_reinforced', now()::text)) "
                "ON CONFLICT ON CONSTRAINT relations_unique DO UPDATE SET "
                "  weight = LEAST(relations.weight + %s, %s), "
                "  metadata = jsonb_set("
                "    relations.metadata, '{last_reinforced}', "
                "    to_jsonb(now()::text))",
                (source_id, target_id, initial_weight,
                 weight_bump, weight_max),
            )

    def query_relations(
        self,
        *,
        source_type: str | None = None,
        source_id: uuid.UUID | None = None,
        target_type: str | None = None,
        target_id: uuid.UUID | None = None,
        relation: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Плоский фильтр-SELECT по ``relations``.

        Cross-agent (D7): без фильтра по agent_id. Согласовано с § 33.2
        (волна 18) — auto-link cross-agent, traversal/query тоже.

        Возвращает list of dict'ов с ключами: id, source_type,
        source_id, target_type, target_id, relation, weight, metadata,
        created_at.
        """
        clauses: list[str] = []
        params: list[Any] = []
        if source_type is not None:
            clauses.append("source_type = %s")
            params.append(source_type)
        if source_id is not None:
            clauses.append("source_id = %s")
            params.append(source_id)
        if target_type is not None:
            clauses.append("target_type = %s")
            params.append(target_type)
        if target_id is not None:
            clauses.append("target_id = %s")
            params.append(target_id)
        if relation is not None:
            clauses.append("relation = %s")
            params.append(relation)
        where = "WHERE " + " AND ".join(clauses) if clauses else ""
        sql = (
            "SELECT id, source_type, source_id, target_type, target_id, "
            "       relation, weight, metadata, created_at "
            "  FROM relations " + where +
            " ORDER BY created_at DESC LIMIT %s"
        )
        params.append(limit)
        with self._conn.cursor(row_factory=dict_row) as cur:
            cur.execute(sql, tuple(params))
            return [dict(r) for r in cur.fetchall()]

    def traverse_graph(
        self,
        *,
        root_id: uuid.UUID,
        depth: int = 1,
        relation_filter: str | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Recursive CTE traversal от root_id, depth ≤ 3, limit ≤ 20.

        Cross-agent (D7) — без фильтра по agent_id. Возвращает список
        соседей: id, type, relation, direction (outgoing/incoming),
        depth, weight, content_preview (первые 100 chars из memories.content).

        В Styx все entities — type='memory' (одна таблица). После волны
        19 (documents+chunks) traversal расширится.

        Если ``relation_filter`` задан — применяется в каждой ветке
        recursive CTE (а не в outer query): иначе recursive step
        проходит через любые рёбра, включая нежелательные (например,
        supersedes), пachuna depth calculations.
        """
        capped_depth = max(1, min(int(depth), 3))
        capped_limit = max(1, min(int(limit), 20))

        rfilter_clause = ""
        if relation_filter is not None:
            rfilter_clause = "AND r.relation = %s"

        # Параметры идут в порядке появления %s в SQL'е:
        # base out (root, [rfilter]), base in (root, [rfilter]),
        # recursive (depth, [rfilter]), outer WHERE (root), LIMIT.
        sql_params: list[Any] = [root_id]
        if relation_filter is not None:
            sql_params.append(relation_filter)
        sql_params.append(root_id)
        if relation_filter is not None:
            sql_params.append(relation_filter)
        sql_params.append(capped_depth)
        if relation_filter is not None:
            sql_params.append(relation_filter)
        sql_params.append(root_id)
        sql_params.append(capped_limit)

        sql = f"""
        WITH RECURSIVE graph AS (
            SELECT
                r.target_id AS id, r.target_type AS type, r.relation,
                'outgoing'::text AS direction, 1 AS depth, r.weight
              FROM relations r
             WHERE r.source_id = %s {rfilter_clause}
            UNION ALL
            SELECT
                r.source_id AS id, r.source_type AS type, r.relation,
                'incoming'::text AS direction, 1 AS depth, r.weight
              FROM relations r
             WHERE r.target_id = %s {rfilter_clause}
            UNION ALL
            SELECT
                r.target_id AS id, r.target_type AS type, r.relation,
                g.direction, g.depth + 1 AS depth, r.weight
              FROM graph g
              JOIN relations r ON r.source_id = g.id
             WHERE g.depth < %s {rfilter_clause}
        )
        SELECT DISTINCT ON (id) id, type, relation, direction, depth, weight
          FROM graph
         WHERE id <> %s
         ORDER BY id, depth
         LIMIT %s
        """
        with self._conn.cursor() as cur:
            cur.execute(sql, tuple(sql_params))
            rows = cur.fetchall()

        # Достаём content_preview одним SELECT'ом — пока только из
        # memories (волна 19 расширит).
        ids = [r[0] for r in rows]
        previews: dict[uuid.UUID, str] = {}
        if ids:
            with self._conn.cursor() as cur:
                cur.execute(
                    "SELECT id, LEFT(content, 100) FROM memories "
                    " WHERE id = ANY(%s::uuid[])",
                    (ids,),
                )
                previews = {r[0]: r[1] for r in cur.fetchall()}

        out: list[dict[str, Any]] = []
        for r in rows:
            mid, mtype, rel, direction, dpt, weight = r
            out.append({
                "id": mid,
                "type": mtype,
                "relation": rel,
                "direction": direction,
                "depth": dpt,
                "weight": float(weight),
                "content_preview": previews.get(mid, ""),
            })
        return out

    def insert_link(
        self,
        *,
        source_type: str,
        source_id: uuid.UUID,
        target_type: str,
        target_id: uuid.UUID,
        relation: str,
        weight: float = 1.0,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Manual edge insert. Идемпотентен через ON CONFLICT DO NOTHING.

        Возвращает True если ряд создан, False если уже существовал.

        Не делает commit — caller управляет транзакцией.
        """
        meta = psycopg.types.json.Jsonb(metadata or {})
        with self._conn.cursor() as cur:
            cur.execute(
                "INSERT INTO relations "
                "  (source_type, source_id, target_type, target_id, "
                "   relation, weight, metadata) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s) "
                "ON CONFLICT ON CONSTRAINT relations_unique DO NOTHING "
                "RETURNING id",
                (source_type, source_id, target_type, target_id,
                 relation, weight, meta),
            )
            return cur.fetchone() is not None

    # -- documents + chunks (волна 19) -----------------------------------

    def insert_document(
        self,
        *,
        source: str,
        char_count: int,
        summary: str | None = None,
        content_hash: str | None = None,
        metadata: dict[str, Any] | None = None,
        file_path: str | None = None,
        original_name: str | None = None,
        mime_type: str | None = None,
        source_ref: str | None = None,
        size_bytes: int | None = None,
        visibility: str | None = None,
    ) -> uuid.UUID:
        """INSERT virtual document для store-routing'а длинного content'а
        либо file-ingest'а (волна 28).

        Документ — pointer на агрегат (chunks). ``source`` указывает на
        call-site роутинга: ``'memory_store'`` для StyxMemoryCore.
        ``'insert_batch_memory'`` для batch consolidation handler'а,
        ``'ingest_document'`` для file-ingest pipeline (волна 28).

        File-метаданные (``file_path``, ``original_name``, ``mime_type``,
        ``source_ref``, ``size_bytes``, ``visibility``) — заполняются для
        file-ingest, NULL для store-routed субъектов (волна 19).
        Backfill из metadata JSONB для memorybox-migrated рядов делает
        миграция 0007.

        ``content_hash`` — partial UNIQUE constraint
        ``uq_documents_agent_content_hash`` срабатывает на дубликат для
        того же agent_id; caller должен ловить psycopg UniqueViolation
        либо предварительно вызвать ``find_document_by_content_hash``.

        Не делает commit — caller управляет транзакцией.
        """
        meta = psycopg.types.json.Jsonb(metadata or {})
        with self._conn.cursor() as cur:
            cur.execute(
                "INSERT INTO documents "
                "  (agent_id, source, content_hash, char_count, "
                "   summary, metadata, file_path, original_name, "
                "   mime_type, source_ref, size_bytes, visibility) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) "
                "RETURNING id",
                (
                    self._agent_id,
                    source,
                    content_hash,
                    int(char_count),
                    summary,
                    meta,
                    file_path,
                    original_name,
                    mime_type,
                    source_ref,
                    None if size_bytes is None else int(size_bytes),
                    visibility,
                ),
            )
            row = cur.fetchone()
        if row is None:
            raise RuntimeError("insert_document не вернул id")
        return row[0]

    def find_document_by_content_hash(
        self,
        content_hash: str,
    ) -> uuid.UUID | None:
        """Поиск existing document по (agent_id, content_hash) для
        idempotency повторного file-ingest'а (волна 28 D9).

        Возвращает ``document_id`` если найден, иначе ``None``. Partial
        UNIQUE index ``uq_documents_agent_content_hash`` обеспечивает
        быстрый lookup; matched на конкретный ``self._agent_id``.
        """
        with self._conn.cursor() as cur:
            cur.execute(
                "SELECT id FROM documents "
                "WHERE agent_id = %s AND content_hash = %s "
                "LIMIT 1",
                (self._agent_id, content_hash),
            )
            row = cur.fetchone()
        return row[0] if row else None

    def insert_chunks_batch(
        self,
        document_id: uuid.UUID,
        chunks: list[tuple[int, str, list[float] | None, int, int]],
    ) -> None:
        """Batch INSERT chunks одного document'а.

        ``chunks`` — list of ``(position, content, embedding, char_start,
        char_end)``. Embedding ``None`` → INSERT с NULL (recall не
        найдёт chunk до reembed'а). UNIQUE (document_id, position)
        защитит от двойного INSERT'а одной позиции.

        Не делает commit — caller управляет транзакцией.
        """
        if not chunks:
            return
        sql = (
            "INSERT INTO chunks "
            "  (document_id, position, content, embedding, "
            "   char_start, char_end) "
            "VALUES (%s, %s, %s, %s, %s, %s)"
        )
        with self._conn.cursor() as cur:
            for position, content, embedding, char_start, char_end in chunks:
                cur.execute(
                    sql,
                    (
                        document_id,
                        int(position),
                        content,
                        _vector_literal(embedding),
                        int(char_start),
                        int(char_end),
                    ),
                )

    # -- search_archive (волна 20) ---------------------------------------

    def search_chunks_for_archive(
        self,
        *,
        query_vector: list[float],
        query_text: str,
        limit: int,
        date_from: Any | None = None,
        date_to: Any | None = None,
        snapshot_cycle_start: Any | None = None,
    ) -> list["ChunkHit"]:
        """Hybrid search поверх `chunks` — top-K по weighted vector+ts_rank.

        agent_id isolation через JOIN на `documents` (chunks нет
        собственного agent_id — он у parent документа). content_tsv
        генерируется миграцией 0006; GIN `idx_chunks_fts` — там же.

        Веса adaptive через `search_weights.compute_weights(query_text)`:
        короткий query → vector-heavy 0.8/0.2; длинный → 0.6/0.4.

        Filters:
        - `date_from`/`date_to` — на `c.created_at` (когда chunk
          создан; chunks могут быть переписаны позже document'а).
        - `snapshot_cycle_start` — на `c.created_at <= cs` (опц.
          temporal isolation, см. waves/20 § D6).

        Chunks без embedding (NULL) исключаются.
        """
        from .search_weights import compute_weights

        weights = compute_weights(query_text)

        snapshot_clause = ""
        if snapshot_cycle_start is not None:
            snapshot_clause = " AND c.created_at <= %(cycle_start)s"
        date_clauses: list[str] = []
        if date_from is not None:
            date_clauses.append(" AND c.created_at >= %(date_from)s")
        if date_to is not None:
            date_clauses.append(" AND c.created_at <= %(date_to)s")
        date_sql = "".join(date_clauses)

        sql = f"""
            SELECT c.document_id,
                   c.position,
                   c.content,
                   c.char_start,
                   c.char_end,
                   (
                       {weights.vector_weight} * (1 - (c.embedding <=> %(qvec)s))
                     + {weights.bm25_weight} * ts_rank(
                           c.content_tsv,
                           plainto_tsquery('simple', %(qtext)s),
                           32
                       )
                   ) AS score
              FROM chunks c
              JOIN documents d ON d.id = c.document_id
             WHERE d.agent_id = %(agent_id)s
               AND c.embedding IS NOT NULL
               {snapshot_clause}{date_sql}
             ORDER BY score DESC
             LIMIT %(lim)s
        """
        params: dict[str, Any] = {
            "qvec": _vector_literal(query_vector),
            "qtext": query_text,
            "agent_id": self._agent_id,
            "lim": int(limit),
        }
        if snapshot_cycle_start is not None:
            params["cycle_start"] = snapshot_cycle_start
        if date_from is not None:
            params["date_from"] = date_from
        if date_to is not None:
            params["date_to"] = date_to

        with self._conn.cursor(row_factory=dict_row) as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
        return [
            ChunkHit(
                document_id=r["document_id"],
                position=int(r["position"]),
                content=r["content"],
                char_start=int(r["char_start"]),
                char_end=int(r["char_end"]),
                score=float(r["score"]),
            )
            for r in rows
        ]

    def search_dialogue_for_archive(
        self,
        *,
        query_vector: list[float],
        query_text: str,
        limit: int,
        date_from: Any | None = None,
        date_to: Any | None = None,
        snapshot_cycle_start: Any | None = None,
    ) -> list["DialogueHit"]:
        """Hybrid search поверх `memories WHERE role IN ('user','assistant')`.

        `superseded_by IS NULL` — только живые реплики (selective
        gatekeeper в волне 17 мог suppress'нуть устаревшие).
        Embedding NOT NULL — без embed'а recall не находится.

        Filters / weights — как в `search_chunks_for_archive`.
        """
        from .search_weights import compute_weights

        weights = compute_weights(query_text)

        snapshot_clause = ""
        if snapshot_cycle_start is not None:
            snapshot_clause = " AND m.created_at <= %(cycle_start)s"
        date_clauses: list[str] = []
        if date_from is not None:
            date_clauses.append(" AND m.created_at >= %(date_from)s")
        if date_to is not None:
            date_clauses.append(" AND m.created_at <= %(date_to)s")
        date_sql = "".join(date_clauses)

        sql = f"""
            SELECT m.id,
                   m.role,
                   m.content,
                   m.created_at,
                   (
                       {weights.vector_weight} * (1 - (m.embedding <=> %(qvec)s))
                     + {weights.bm25_weight} * ts_rank(
                           m.content_tsv,
                           plainto_tsquery('simple', %(qtext)s),
                           32
                       )
                   ) AS score
              FROM memories m
             WHERE m.agent_id = %(agent_id)s
               AND m.role IN ('user','assistant')
               AND m.embedding IS NOT NULL
               AND m.superseded_by IS NULL
               {snapshot_clause}{date_sql}
             ORDER BY score DESC
             LIMIT %(lim)s
        """
        params: dict[str, Any] = {
            "qvec": _vector_literal(query_vector),
            "qtext": query_text,
            "agent_id": self._agent_id,
            "lim": int(limit),
        }
        if snapshot_cycle_start is not None:
            params["cycle_start"] = snapshot_cycle_start
        if date_from is not None:
            params["date_from"] = date_from
        if date_to is not None:
            params["date_to"] = date_to

        with self._conn.cursor(row_factory=dict_row) as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
        return [
            DialogueHit(
                memory_id=r["id"],
                role=r["role"],
                content=r["content"],
                created_at=r["created_at"],
                score=float(r["score"]),
            )
            for r in rows
        ]

    def compute_agent_usage_p75(self) -> float:
        """75-й перцентиль 30-day used_in_output count'а для текущего агента.

        В волне 7 ``used_in_output`` не выставляется (classifier sweep —
        волна 7c) → этот метод всегда возвращает 0.0, что в
        scoring.build_factor_exprs трактуется как «нет персонализации»
        → usage_factor = 1.0.

        Port ``computeAgentUsageP75`` (memory.ts:689).
        """
        with self._conn.cursor() as cur:
            cur.execute(
                "SELECT percentile_cont(0.75) WITHIN GROUP (ORDER BY uc)::double precision "
                "AS p75 FROM ("
                "  SELECT count(re.*) AS uc"
                "    FROM memories m"
                "    LEFT JOIN recall_events re"
                "      ON re.memory_id = m.id"
                "     AND re.used_in_output = true"
                "     AND re.matched_at > now() - interval '30 days'"
                "   WHERE m.agent_id = %s"
                "   GROUP BY m.id"
                "  HAVING count(re.*) > 0"
                ") agent_usage",
                (self._agent_id,),
            )
            row = cur.fetchone()
        if row is None or row[0] is None:
            return 0.0
        p75 = float(row[0])
        return p75 if p75 > 0 else 0.0

    # -- low-level хелперы (не для внешнего использования) ---------------

    def get_latest_hot_sentiment(
        self, *, within_seconds: float
    ) -> tuple[tuple[float, float, float], Any] | None:
        """Latest emotional_state entry с ``source='hot_sentiment'`` и
        ``metadata.hot_vad`` (raw VAD peer-реплики) не старее ``within_seconds``.

        Возвращает ``((v, a, d), at)`` либо ``None``. Используется каналом
        peer_vad волны 15 для inject'а через pre_llm_call hook.

        Backward compat: legacy записи без ``metadata`` или с metadata без
        ``hot_vad`` ключа дают None — channel skip'нет canal silently.
        """
        sql = (
            "SELECT metadata, at "
            "FROM emotional_state "
            "WHERE agent_id = %s "
            "  AND source = 'hot_sentiment' "
            "  AND at >= now() - make_interval(secs => %s) "
            "ORDER BY at DESC "
            "LIMIT 1"
        )
        with self._conn.cursor() as cur:
            cur.execute(sql, (self._agent_id, float(within_seconds)))
            row = cur.fetchone()
        if row is None:
            return None
        metadata, at = row
        if not isinstance(metadata, dict):
            return None
        vad = metadata.get("hot_vad")
        if not (
            isinstance(vad, list)
            and len(vad) == 3
            and all(isinstance(x, (int, float)) for x in vad)
        ):
            return None
        return ((float(vad[0]), float(vad[1]), float(vad[2])), at)

    # ── Волна 14: queries для batch consolidation ───────────────────

    def count_dialogue_messages_since(
        self, *, since: Any | None
    ) -> int:
        """Кол-во user/assistant memories у этого агента после ``since``.

        ``since=None`` — все реплики. Используется scheduler'ом для
        проверки триггера ≥20 новых реплик.
        """
        sql = (
            "SELECT count(*)::int FROM memories "
            "WHERE agent_id = %s "
            "  AND role IN ('user','assistant') "
            "  AND created_at > coalesce(%s, '-infinity'::timestamptz)"
        )
        with self._conn.cursor() as cur:
            cur.execute(sql, (self._agent_id, since))
            return cur.fetchone()[0]

    def latest_dialogue_at(self) -> Any | None:
        """``max(created_at)`` среди user/assistant memories. None если
        агент ещё не писал диалога."""
        sql = (
            "SELECT max(created_at) FROM memories "
            "WHERE agent_id = %s "
            "  AND role IN ('user','assistant')"
        )
        with self._conn.cursor() as cur:
            cur.execute(sql, (self._agent_id,))
            return cur.fetchone()[0]

    def select_dialogue_window(
        self,
        *,
        window_from: Any,
        window_to: Any,
        with_overlap_messages: int = 0,
    ) -> list[dict]:
        """Реплики окна для batch consolidation handler'а.

        ``window_from`` exclusive (>), ``window_to`` inclusive (<=).
        ``with_overlap_messages > 0`` → дополнительно последние N
        реплик строго ДО ``window_from`` (для контекстного overlap'а).

        Возвращает list[{id, role, content, created_at}] в
        хронологическом порядке (overlap первым, затем main).
        """
        main_sql = (
            "SELECT id, role, content, created_at "
            "FROM memories "
            "WHERE agent_id = %s "
            "  AND role IN ('user','assistant') "
            "  AND created_at > %s "
            "  AND created_at <= %s "
            "ORDER BY created_at ASC, seq ASC"
        )
        with self._conn.cursor(row_factory=dict_row) as cur:
            cur.execute(main_sql, (self._agent_id, window_from, window_to))
            main_rows = list(cur.fetchall())

        if with_overlap_messages <= 0:
            return main_rows

        overlap_sql = (
            "SELECT id, role, content, created_at FROM ("
            "  SELECT id, role, content, created_at "
            "  FROM memories "
            "  WHERE agent_id = %s "
            "    AND role IN ('user','assistant') "
            "    AND created_at <= %s "
            "  ORDER BY created_at DESC, seq DESC "
            "  LIMIT %s"
            ") t ORDER BY created_at ASC"
        )
        with self._conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                overlap_sql,
                (self._agent_id, window_from, with_overlap_messages),
            )
            overlap_rows = list(cur.fetchall())
        return overlap_rows + main_rows

    def select_recent_memories_for_consolidation(
        self, *, hours: int = 24, limit: int = 30
    ) -> list[dict]:
        """30 последних memories агента за последние ``hours`` часов.

        Используется как memory_context в LLM-prompt'е batch
        consolidation handler'а — что агент уже недавно запомнил.
        Фильтр по `superseded_by IS NULL` — только живые.
        """
        sql = (
            "SELECT kind_src, content "
            "FROM memories "
            "WHERE agent_id = %s "
            "  AND created_at > now() - make_interval(hours => %s) "
            "  AND superseded_by IS NULL "
            "ORDER BY created_at DESC "
            "LIMIT %s"
        )
        with self._conn.cursor(row_factory=dict_row) as cur:
            cur.execute(sql, (self._agent_id, hours, limit))
            return list(cur.fetchall())

    def insert_batch_memory(
        self,
        *,
        content: str,
        archive_ref: dict,
    ) -> uuid.UUID:
        """INSERT batch consolidation memory. kind='episode',
        kind_src='dialogue_batch_consolidation'.

        Не делает commit. embedding=NULL — подхватит reembed CLI или
        following sync_turn'ы.
        """
        from psycopg.types.json import Jsonb
        # role='summary' — batch consolidation = produced summary окна
        # диалога. Совпадает с CHECK constraint (memories_role_check
        # из миграции 0001) и семантически точнее чем user/assistant.
        sql = (
            "INSERT INTO memories ("
            "  agent_id, role, kind, kind_src, content, "
            "  importance_provisional, archive_ref, embedding"
            ") VALUES ("
            "  %s, 'summary', 'episode', 'dialogue_batch_consolidation', "
            "  %s, 0.5, %s, NULL"
            ") RETURNING id"
        )
        with self._conn.cursor() as cur:
            cur.execute(sql, (self._agent_id, content, Jsonb(archive_ref)))
            return cur.fetchone()[0]

    # -- Reinterpret (волна 22) ------------------------------------------

    def find_pending_reinterpret_application(
        self, memory_id: uuid.UUID
    ) -> int | None:
        """Branch 1 cooldown'а — есть ли pending_sleep application для
        memory_id. Возвращает application.id или None.

        partial UNIQUE индекс
        ``uq_reinterpret_applications_one_pending_per_memory`` (schema
        0002) гарантирует что pending_sleep ≤ 1 на memory.
        """
        sql = (
            "SELECT id FROM reinterpret_applications "
            "WHERE memory_id = %s AND status = 'pending_sleep' "
            "  AND agent_id = %s LIMIT 1"
        )
        with self._conn.cursor() as cur:
            cur.execute(sql, (memory_id, self._agent_id))
            row = cur.fetchone()
        return None if row is None else int(row[0])

    def latest_reinterpretation_at(self, memory_id: uuid.UUID) -> Any | None:
        """Branch 2 cooldown'а — `reinterpreted_at` последней revision."""
        sql = (
            "SELECT reinterpreted_at FROM memory_reinterpretations "
            "WHERE memory_id = %s AND agent_id = %s "
            "ORDER BY reinterpreted_at DESC LIMIT 1"
        )
        with self._conn.cursor() as cur:
            cur.execute(sql, (memory_id, self._agent_id))
            row = cur.fetchone()
        return None if row is None else row[0]

    def insert_reinterpret_application(
        self, *, task_id: uuid.UUID, memory_id: uuid.UUID
    ) -> int:
        """INSERT application (status='pending_sleep'). Не делает commit."""
        sql = (
            "INSERT INTO reinterpret_applications "
            "(task_id, memory_id, agent_id, status) "
            "VALUES (%s, %s, %s, 'pending_sleep') RETURNING id"
        )
        with self._conn.cursor() as cur:
            cur.execute(sql, (task_id, memory_id, self._agent_id))
            return int(cur.fetchone()[0])

    def memory_exists(self, memory_id: uuid.UUID) -> bool:
        """Light check — есть ли memory с этим id под `agent_id`."""
        sql = (
            "SELECT 1 FROM memories WHERE id = %s AND agent_id = %s LIMIT 1"
        )
        with self._conn.cursor() as cur:
            cur.execute(sql, (memory_id, self._agent_id))
            return cur.fetchone() is not None

    def load_pending_reinterpret_applications(self) -> list[dict]:
        """SELECT pending_sleep applications + JOIN на llm_tasks для
        apply-sweeper'а. Только этот агент."""
        sql = (
            "SELECT a.id AS application_id, "
            "       a.memory_id::text AS memory_id, "
            "       a.task_id::text AS task_id, "
            "       t.status AS task_status, "
            "       t.result AS task_result, "
            "       t.error  AS task_error "
            "  FROM reinterpret_applications a "
            "  JOIN llm_tasks t ON t.id = a.task_id "
            " WHERE a.status = 'pending_sleep' AND a.agent_id = %s "
            " ORDER BY a.created_at"
        )
        with self._conn.cursor(row_factory=dict_row) as cur:
            cur.execute(sql, (self._agent_id,))
            return list(cur.fetchall())

    def apply_reinterpret_update(
        self,
        *,
        memory_id: uuid.UUID,
        merged_text: str,
        merged_embedding: list[float],
    ) -> int:
        """UPDATE memories.content + embedding. Возвращает rowcount.
        Не делает commit."""
        sql = (
            "UPDATE memories SET content = %s, embedding = %s, "
            "  updated_at = now() "
            "WHERE id = %s AND agent_id = %s"
        )
        with self._conn.cursor() as cur:
            cur.execute(
                sql,
                (
                    merged_text,
                    _vector_literal(merged_embedding),
                    memory_id,
                    self._agent_id,
                ),
            )
            return cur.rowcount or 0

    def insert_memory_reinterpretation(
        self,
        *,
        memory_id: uuid.UUID,
        previous_text: str,
        new_understanding_text: str,
        merged_text: str,
        previous_embedding: list[float],
        merged_embedding: list[float],
        weight_applied: float,
    ) -> int:
        """INSERT audit row в memory_reinterpretations. Возвращает id."""
        if len(previous_embedding) == 0 or len(merged_embedding) == 0:
            raise ValueError(
                "insert_memory_reinterpretation: пустой embedding "
                f"(prev={len(previous_embedding)} merged={len(merged_embedding)})"
            )
        if len(previous_embedding) != len(merged_embedding):
            raise ValueError(
                "insert_memory_reinterpretation: dim mismatch "
                f"(prev={len(previous_embedding)} merged={len(merged_embedding)})"
            )
        sql = (
            "INSERT INTO memory_reinterpretations "
            "(memory_id, agent_id, previous_text, new_understanding_text, "
            " merged_text, previous_embedding, merged_embedding, "
            " weight_applied) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s) RETURNING id"
        )
        with self._conn.cursor() as cur:
            cur.execute(
                sql,
                (
                    memory_id,
                    self._agent_id,
                    previous_text,
                    new_understanding_text,
                    merged_text,
                    _vector_literal(previous_embedding),
                    _vector_literal(merged_embedding),
                    float(weight_applied),
                ),
            )
            return int(cur.fetchone()[0])

    def mark_reinterpret_applied(self, application_id: int) -> None:
        """UPDATE applications status='applied', applied_at=now().
        Не делает commit."""
        sql = (
            "UPDATE reinterpret_applications SET status='applied', "
            "  applied_at = now() "
            "WHERE id = %s AND status = 'pending_sleep' AND agent_id = %s"
        )
        with self._conn.cursor() as cur:
            cur.execute(sql, (application_id, self._agent_id))

    def mark_reinterpret_skipped(self, application_id: int) -> None:
        """UPDATE applications status='skipped', applied_at=now()."""
        sql = (
            "UPDATE reinterpret_applications SET status='skipped', "
            "  applied_at = now() "
            "WHERE id = %s AND status = 'pending_sleep' AND agent_id = %s"
        )
        with self._conn.cursor() as cur:
            cur.execute(sql, (application_id, self._agent_id))

    # -- Memory consolidation (волна 22) ---------------------------------

    def select_consolidation_window(
        self, *, window_from: Any, window_to: Any
    ) -> list[dict]:
        """SELECT memories для cluster discovery. Filter:
        - agent_id = current
        - created_at в [window_from, window_to]
        - superseded_by IS NULL
        - kind_src <> 'dialogue_consolidation_daily' (избегаем рекурсии)
        - embedding IS NOT NULL
        Сортировка по created_at ASC.

        Returns dicts с {id, embedding} (embedding raw — caller парсит
        через parse_vector).
        """
        sql = (
            "SELECT id, embedding FROM memories "
            "WHERE agent_id = %s "
            "  AND created_at >= %s AND created_at <= %s "
            "  AND superseded_by IS NULL "
            "  AND kind_src <> 'dialogue_consolidation_daily' "
            "  AND embedding IS NOT NULL "
            "ORDER BY created_at ASC"
        )
        with self._conn.cursor(row_factory=dict_row) as cur:
            cur.execute(sql, (self._agent_id, window_from, window_to))
            return list(cur.fetchall())

    def insert_memory_consolidation_application(
        self, *, task_id: uuid.UUID, source_ids: list[uuid.UUID]
    ) -> int:
        """INSERT application (status='pending_sleep'). Не делает commit."""
        if len(source_ids) < 2:
            raise ValueError(
                "memory_consolidation_applications.source_ids — минимум 2 "
                f"(получено {len(source_ids)})"
            )
        sql = (
            "INSERT INTO memory_consolidation_applications "
            "(task_id, agent_id, source_ids, status) "
            "VALUES (%s, %s, %s, 'pending_sleep') RETURNING id"
        )
        with self._conn.cursor() as cur:
            cur.execute(sql, (task_id, self._agent_id, source_ids))
            return int(cur.fetchone()[0])

    def load_pending_consolidation_applications(self) -> list[dict]:
        """SELECT pending_sleep applications + JOIN на llm_tasks для
        apply-sweeper'а. Только этот агент."""
        sql = (
            "SELECT a.id AS application_id, "
            "       a.source_ids AS source_ids, "
            "       a.task_id::text AS task_id, "
            "       t.status AS task_status, "
            "       t.result AS task_result, "
            "       t.error  AS task_error "
            "  FROM memory_consolidation_applications a "
            "  JOIN llm_tasks t ON t.id = a.task_id "
            " WHERE a.status = 'pending_sleep' AND a.agent_id = %s "
            " ORDER BY a.created_at"
        )
        with self._conn.cursor(row_factory=dict_row) as cur:
            cur.execute(sql, (self._agent_id,))
            return list(cur.fetchall())

    def load_memories_for_consolidation(
        self, memory_ids: list[uuid.UUID]
    ) -> list[dict]:
        """SELECT N memories по PK для handler'а. Filter agent_id +
        IN list. Возвращает [{id, content, kind, kind_src,
        visibility, superseded_by, embedding}] в порядке payload'а
        (как memorybox)."""
        if not memory_ids:
            return []
        sql = (
            "SELECT id, content, kind, kind_src, visibility, "
            "       superseded_by, embedding "
            "  FROM memories "
            " WHERE id = ANY(%s) AND agent_id = %s"
        )
        with self._conn.cursor(row_factory=dict_row) as cur:
            cur.execute(sql, (memory_ids, self._agent_id))
            rows = list(cur.fetchall())
        # preserve payload order
        by_id = {r["id"]: r for r in rows}
        out: list[dict] = []
        for mid in memory_ids:
            row = by_id.get(mid)
            if row is not None:
                out.append(row)
        return out

    def insert_consolidated_memory(
        self,
        *,
        content: str,
        embedding: list[float],
        kind: str,
        visibility: str,
        source_ids: list[uuid.UUID],
        application_id: int,
    ) -> uuid.UUID:
        """INSERT new consolidated memory.
        kind_src='dialogue_consolidation_daily', importance_provisional=0.7.
        metadata.consolidation хранит source_ids + count + application_id.
        Не делает commit.
        """
        from psycopg.types.json import Jsonb
        metadata = {
            "consolidation": {
                "source_ids": [str(sid) for sid in source_ids],
                "source_count": len(source_ids),
                "llm_task_application_id": application_id,
            }
        }
        sql = (
            "INSERT INTO memories "
            "(agent_id, visibility, kind, kind_src, content, metadata, "
            " embedding, importance_provisional, role) "
            "VALUES (%s, %s, %s, 'dialogue_consolidation_daily', "
            "        %s, %s, %s, 0.7, 'summary') "
            "RETURNING id"
        )
        with self._conn.cursor() as cur:
            cur.execute(
                sql,
                (
                    self._agent_id,
                    visibility,
                    kind,
                    content,
                    Jsonb(metadata),
                    _vector_literal(embedding),
                ),
            )
            return cur.fetchone()[0]

    def mark_consolidation_sources_superseded(
        self, *, new_memory_id: uuid.UUID, source_ids: list[uuid.UUID]
    ) -> int:
        """UPDATE memories SET superseded_by=$new_id WHERE id=ANY($source_ids)
        AND superseded_by IS NULL. Idempotent — если уже superseded
        другим pipeline'ом, пропускаем. Возвращает rowcount.
        Не делает commit."""
        sql = (
            "UPDATE memories SET superseded_by = %s, updated_at = now() "
            "WHERE id = ANY(%s) AND superseded_by IS NULL "
            "  AND agent_id = %s"
        )
        with self._conn.cursor() as cur:
            cur.execute(sql, (new_memory_id, source_ids, self._agent_id))
            return cur.rowcount or 0

    def mark_consolidation_applied(
        self, *, application_id: int, new_memory_id: uuid.UUID
    ) -> None:
        """UPDATE applications status='applied', new_memory_id, applied_at."""
        sql = (
            "UPDATE memory_consolidation_applications "
            "  SET status='applied', new_memory_id=%s, applied_at=now() "
            "WHERE id = %s AND status = 'pending_sleep' AND agent_id = %s"
        )
        with self._conn.cursor() as cur:
            cur.execute(sql, (new_memory_id, application_id, self._agent_id))

    def mark_consolidation_skipped(self, application_id: int) -> None:
        """UPDATE applications status='skipped', applied_at=now()."""
        sql = (
            "UPDATE memory_consolidation_applications "
            "  SET status='skipped', applied_at=now() "
            "WHERE id = %s AND status = 'pending_sleep' AND agent_id = %s"
        )
        with self._conn.cursor() as cur:
            cur.execute(sql, (application_id, self._agent_id))

    # -- Ingest API (волна 23) -------------------------------------------

    def ingest_upsert_memory(
        self,
        *,
        content: str,
        kind: str,
        kind_src: str,
        content_hash: str | None,
        embedding: list[float] | None = None,
        metadata: dict[str, Any] | None = None,
        role: str = "system",
        importance_provisional: float | None = None,
    ) -> tuple[uuid.UUID, bool]:
        """INSERT memory с ON CONFLICT (agent_id, content_hash) DO NOTHING.

        Используется external-pipeline ingest каналом
        (``POST /ingest_experience``). Идемпотентен: повторный вызов с
        тем же ``content_hash`` возвращает существующий ``memory_id``
        с ``deduplicated=True``, без побочных эффектов.

        ``content_hash=None`` → чистый INSERT (partial UNIQUE индекс
        ``memories_agent_content_hash_uniq`` игнорирует NULL),
        ``deduplicated=False`` всегда. ``content_hash != None`` и
        конфликт → SELECT existing по ``(agent_id, content_hash)``,
        ``deduplicated=True``.

        ON CONFLICT vs SELECT+INSERT: параллельные ingest'ы того же
        payload'а проходят SELECT до первого COMMIT'а (READ COMMITTED)
        и оба пытаются INSERT, второй ловит unique_violation 23505.
        ON CONFLICT перекладывает race на БД.

        Не делает commit — caller управляет транзакцией.
        """
        meta = metadata or {}
        cols = [
            "agent_id", "role", "kind", "kind_src",
            "content", "embedding", "metadata", "content_hash",
        ]
        params: list[Any] = [
            self._agent_id, role, kind, kind_src,
            content, _vector_literal(embedding),
            psycopg.types.json.Jsonb(meta), content_hash,
        ]
        if importance_provisional is not None:
            cols.append("importance_provisional")
            params.append(float(importance_provisional))
        placeholders = ", ".join(["%s"] * len(cols))

        if content_hash is None:
            # Чистый INSERT без конфликта — partial UNIQUE индекс
            # игнорирует NULL, идемпотентность не применяется.
            sql = (
                f"INSERT INTO memories ({', '.join(cols)}) "
                f"VALUES ({placeholders}) RETURNING id"
            )
            with self._conn.cursor() as cur:
                cur.execute(sql, tuple(params))
                row = cur.fetchone()
            if row is None:
                raise RuntimeError(
                    "ingest_upsert_memory: INSERT не вернул id"
                )
            return row[0], False

        sql = (
            f"INSERT INTO memories ({', '.join(cols)}) "
            f"VALUES ({placeholders}) "
            f"ON CONFLICT (agent_id, content_hash) "
            f"  WHERE content_hash IS NOT NULL "
            f"  DO NOTHING "
            f"RETURNING id"
        )
        with self._conn.cursor() as cur:
            cur.execute(sql, tuple(params))
            row = cur.fetchone()
            if row is not None:
                return row[0], False
            # Конфликт по (agent_id, content_hash) — достаём existing.
            cur.execute(
                "SELECT id FROM memories "
                "WHERE agent_id = %s AND content_hash = %s LIMIT 1",
                (self._agent_id, content_hash),
            )
            existing = cur.fetchone()
        if existing is None:
            # Не должно случаться: ON CONFLICT сработал, но existing
            # ряд не виден. Честная ошибка лучше молчаливого повтора.
            raise RuntimeError(
                "ingest_upsert_memory: ON CONFLICT triggered but no "
                "existing memory found under same (agent_id, content_hash)"
            )
        return existing[0], True

    # ── Волна 24: dialogue tools ────────────────────────────────────

    def dialogue_search(
        self,
        *,
        query_vector: list[float],
        query_text: str | None = None,
        limit: int = 10,
        session_id: uuid.UUID | str | None = None,
        after: Any | None = None,
        before: Any | None = None,
    ) -> list[DialogueSearchHit]:
        """Hybrid (FTS+vector) либо pure-vector search поверх
        ``memories WHERE role IN ('user','assistant')``.

        ``query_text=None`` → pure-vector (ORDER BY distance ASC,
        score = 1 - distance). Иначе — hybrid через
        ``compute_weights(query_text)`` (D6 в waves/24).

        Optional filters: ``session_id``, ``after`` (>=), ``before`` (<=).
        Cross-agent НЕТ — `agent_id = self._agent_id` зашит (D13).

        ``embedding IS NOT NULL`` — рядов без embed'а нет в выдаче.
        ``superseded_by IS NULL`` — на dialogue реплики обычно никто
        не ставит supersede, но фильтр держим.

        Не делает commit — read-only.
        """
        from .search_weights import compute_weights

        sid = _as_uuid(session_id) if session_id is not None else None

        where_clauses: list[str] = [
            "m.agent_id = %(agent_id)s",
            "m.role IN ('user','assistant')",
            "m.embedding IS NOT NULL",
            "m.superseded_by IS NULL",
        ]
        params: dict[str, Any] = {
            "agent_id": self._agent_id,
            "qvec": _vector_literal(query_vector),
            "lim": int(limit),
        }
        if sid is not None:
            where_clauses.append("m.session_id = %(session_id)s")
            params["session_id"] = sid
        if after is not None:
            where_clauses.append("m.created_at >= %(after)s")
            params["after"] = after
        if before is not None:
            where_clauses.append("m.created_at <= %(before)s")
            params["before"] = before
        where_sql = " AND ".join(where_clauses)

        if query_text:
            weights = compute_weights(query_text)
            score_expr = (
                f"({weights.vector_weight} * (1 - (m.embedding <=> %(qvec)s))"
                f" + {weights.bm25_weight} * ts_rank("
                f"  m.content_tsv,"
                f"  plainto_tsquery('simple', %(qtext)s),"
                f"  32"
                f"))"
            )
            order_sql = "score DESC"
            params["qtext"] = query_text
        else:
            score_expr = "(1 - (m.embedding <=> %(qvec)s))"
            order_sql = "m.embedding <=> %(qvec)s"

        sql = f"""
            SELECT m.id,
                   m.session_id,
                   m.role,
                   m.content,
                   m.created_at,
                   {score_expr} AS score
              FROM memories m
             WHERE {where_sql}
             ORDER BY {order_sql}
             LIMIT %(lim)s
        """
        with self._conn.cursor(row_factory=dict_row) as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
        return [
            DialogueSearchHit(
                memory_id=r["id"],
                session_id=r["session_id"],
                role=r["role"],
                content=r["content"],
                created_at=r["created_at"],
                score=float(r["score"]),
            )
            for r in rows
        ]

    def dialogue_recent(
        self,
        *,
        limit: int = 20,
        session_id: uuid.UUID | str | None = None,
        before: Any | None = None,
    ) -> list[DialogueRecentRow]:
        """Последние ``limit`` реплик (role IN ('user','assistant'))
        в DESC ORDER BY seq, без vector ranking'а.

        Caller должен ``reverse()`` для chronological output (oldest
        first) — pattern волны 24 D7 (consumer-friendly).

        Не делает commit — read-only.
        """
        sid = _as_uuid(session_id) if session_id is not None else None
        clauses = [
            "agent_id = %s",
            "role IN ('user','assistant')",
        ]
        params: list[Any] = [self._agent_id]
        if sid is not None:
            clauses.append("session_id = %s")
            params.append(sid)
        if before is not None:
            clauses.append("created_at <= %s")
            params.append(before)
        params.append(int(limit))
        sql = (
            "SELECT id, session_id, role, content, created_at "
            "FROM memories "
            f"WHERE {' AND '.join(clauses)} "
            "ORDER BY seq DESC LIMIT %s"
        )
        with self._conn.cursor(row_factory=dict_row) as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
        return [
            DialogueRecentRow(
                memory_id=r["id"],
                session_id=r["session_id"],
                role=r["role"],
                content=r["content"],
                created_at=r["created_at"],
            )
            for r in rows
        ]

    def dialogue_list_sessions(
        self, *, limit: int = 10
    ) -> list[DialogueSessionInfo]:
        """List of sessions с counts + first/last_at, ORDER DESC по
        last_message_at.

        Реплики без ``session_id`` (FK NULL) не учитываются (D9).

        Не делает commit — read-only.
        """
        sql = (
            "SELECT session_id, "
            "       count(*)::int AS message_count, "
            "       min(created_at) AS first_message_at, "
            "       max(created_at) AS last_message_at "
            "FROM memories "
            "WHERE agent_id = %s "
            "  AND role IN ('user','assistant') "
            "  AND session_id IS NOT NULL "
            "GROUP BY session_id "
            "ORDER BY max(created_at) DESC "
            "LIMIT %s"
        )
        with self._conn.cursor(row_factory=dict_row) as cur:
            cur.execute(sql, (self._agent_id, int(limit)))
            rows = cur.fetchall()
        return [
            DialogueSessionInfo(
                session_id=r["session_id"],
                message_count=int(r["message_count"]),
                first_message_at=r["first_message_at"],
                last_message_at=r["last_message_at"],
            )
            for r in rows
        ]

    def dialogue_prepare_summary(
        self,
        *,
        session_id: uuid.UUID | str,
        limit: int = 200,
    ) -> list[DialogueTranscriptRow]:
        """Реплики одной session в ASC chronological order для
        transcript-сборщика.

        Фильтр `role IN ('user','assistant')` — system/tool/summary
        не попадают.

        Не делает commit — read-only.
        """
        sid = _as_uuid(session_id)
        sql = (
            "SELECT role, content, created_at "
            "FROM memories "
            "WHERE agent_id = %s "
            "  AND session_id = %s "
            "  AND role IN ('user','assistant') "
            "ORDER BY created_at ASC, seq ASC "
            "LIMIT %s"
        )
        with self._conn.cursor(row_factory=dict_row) as cur:
            cur.execute(sql, (self._agent_id, sid, int(limit)))
            rows = cur.fetchall()
        return [
            DialogueTranscriptRow(
                role=r["role"],
                content=r["content"],
                created_at=r["created_at"],
            )
            for r in rows
        ]

    # -- Explain / analytics / confirm_usage (волна 25) ------------------

    def explain_decompose_target(
        self,
        *,
        memory_id: uuid.UUID,
        query_vector: list[float],
        query_text: str,
        decay_config: DecayConfig | None = None,
        search_config: SearchConfig | None = None,
        usage_norm_p75: float = 0.0,
        emotional_baseline: EmotionalBaseline | None = None,
    ) -> dict[str, Any] | None:
        """SELECT memory row + 11 factor columns + llm_task status для
        decompose-mode'а explain-инструмента (волна 25, port memorybox
        ``explainDecompose`` step 1).

        Возвращает row как dict (с factor-колонками типа ``vector_sim``,
        ``base_match``, ``recency_factor``, ``decay_factor``,
        ``final_score`` и т.п. — список синхронизирован с
        ``factor_select_columns``). Plus поля для importance lifecycle:
        ``llm_task_status``, ``llm_task_created_at``.

        agent_id scope: ``WHERE m.id = %s AND m.agent_id = %s`` —
        memory чужого агента → None (404 на route уровне).

        Не делает commit — read-only.
        """
        opts = BuildFactorExprsOptions(
            text_query_param_index=2,
            search_config=search_config,
            decay_config=decay_config,
            table_alias="m",
            usage_norm_p75=usage_norm_p75,
            emotional_baseline=emotional_baseline,
        )
        factors = build_factor_exprs({"text_query": query_text}, opts)
        cols = factor_select_columns(factors)

        sql = f"""
            SELECT
                m.id, m.kind, m.content, m.created_at, m.updated_at,
                m.agent_id, m.visibility, m.superseded_by,
                m.importance_provisional, m.importance_final,
                m.lifecycle, m.relevance, m.access_count,
                m.last_accessed_at, m.unique_query_count,
                m.recall_score_sum, m.usefulness,
                {cols},
                lt.status     AS llm_task_status,
                lt.created_at AS llm_task_created_at
              FROM memories m
              {factors.usage_lateral_from}
              LEFT JOIN LATERAL (
                SELECT status, created_at
                  FROM llm_tasks
                 WHERE memory_id = m.id
                   AND task_type = 'importance_scoring_from_content'
                 ORDER BY created_at DESC
                 LIMIT 1
              ) lt ON true
             WHERE m.id = %(memory_id)s
               AND m.agent_id = %(agent_id)s
             LIMIT 1
        """
        sql_pg = sql.replace("$1", "%(qvec)s").replace("$2", "%(qtext)s")
        params: dict[str, Any] = {
            "qvec": _vector_literal(query_vector),
            "qtext": query_text,
            "memory_id": memory_id,
            "agent_id": self._agent_id,
        }
        with self._conn.cursor(row_factory=dict_row) as cur:
            cur.execute(sql_pg, params)
            row = cur.fetchone()
        return dict(row) if row is not None else None

    def explain_decompose_rank(
        self,
        *,
        target_score: float,
        query_vector: list[float],
        query_text: str,
        decay_config: DecayConfig | None = None,
        search_config: SearchConfig | None = None,
        usage_norm_p75: float = 0.0,
        emotional_baseline: EmotionalBaseline | None = None,
    ) -> int:
        """count(*)+1 кандидатов (того же агента) с score > target_score.

        Используется decompose'ом для ``rank_in_result_set`` (port
        memorybox explain.ts:280-306). Применяет такие же scoring-
        выражения что target SELECT — fair comparison.
        """
        opts = BuildFactorExprsOptions(
            text_query_param_index=2,
            search_config=search_config,
            decay_config=decay_config,
            table_alias="c",
            usage_norm_p75=usage_norm_p75,
            emotional_baseline=emotional_baseline,
        )
        factors = build_factor_exprs({"text_query": query_text}, opts)
        sql = f"""
            SELECT count(*)::int + 1 AS rank
              FROM memories c
              {factors.usage_lateral_from}
             WHERE c.agent_id = %(agent_id)s
               AND c.embedding IS NOT NULL
               AND c.superseded_by IS NULL
               AND ({factors.score_expr}) > %(target)s
        """
        sql_pg = sql.replace("$1", "%(qvec)s").replace("$2", "%(qtext)s")
        params = {
            "qvec": _vector_literal(query_vector),
            "qtext": query_text,
            "target": float(target_score),
            "agent_id": self._agent_id,
        }
        with self._conn.cursor() as cur:
            cur.execute(sql_pg, params)
            row = cur.fetchone()
        return int(row[0]) if row is not None else 1

    def explain_lifetime_main(
        self, *, memory_id: uuid.UUID
    ) -> dict[str, Any] | None:
        """memory + recall_events aggregate + llm_task для lifetime-режима.

        Port memorybox explain.ts:380-410. `age_days` посчитан в SQL.
        agent_id scope. None если memory не найдено.
        """
        sql = """
            SELECT
                m.id, m.kind, m.content, m.created_at, m.updated_at,
                m.agent_id, m.visibility, m.superseded_by,
                m.importance_provisional, m.importance_final,
                m.lifecycle, m.relevance, m.access_count,
                m.last_accessed_at, m.unique_query_count,
                m.recall_score_sum, m.usefulness,
                EXTRACT(EPOCH FROM (now() - m.created_at)) / 86400.0
                    AS age_days,
                re.total_recall_events, re.avg_match_score,
                lt.id         AS llm_task_id,
                lt.status     AS llm_task_status,
                lt.created_at AS llm_task_created_at,
                lt.task_version AS llm_task_version
              FROM memories m
              LEFT JOIN LATERAL (
                SELECT count(*)::int  AS total_recall_events,
                       avg(match_score)::real AS avg_match_score
                  FROM recall_events
                 WHERE memory_id = m.id
              ) re ON true
              LEFT JOIN LATERAL (
                SELECT id, status, created_at, task_version
                  FROM llm_tasks
                 WHERE memory_id = m.id
                   AND task_type = 'importance_scoring_from_content'
                 ORDER BY created_at DESC
                 LIMIT 1
              ) lt ON true
             WHERE m.id = %s AND m.agent_id = %s
             LIMIT 1
        """
        with self._conn.cursor(row_factory=dict_row) as cur:
            cur.execute(sql, (memory_id, self._agent_id))
            row = cur.fetchone()
        return dict(row) if row is not None else None

    def explain_lifetime_recall_history(
        self, *, memory_id: uuid.UUID, limit: int = 10
    ) -> list[dict[str, Any]]:
        """N последних recall_events для memory_id, DESC by matched_at.

        Не фильтруется по agent_id — caller уже проверил доступность
        memory через ``explain_lifetime_main``. recall_events scope
        наследуется через FK на memories.
        """
        # Tie-break по id DESC: внутри одной транзакции `now()` возвращает
        # start-of-transaction → multiple INSERT'ы дают одинаковый
        # matched_at, и без tie-break'а порядок недетерминирован.
        sql = (
            "SELECT matched_at, query_hash, match_score "
            "  FROM recall_events "
            " WHERE memory_id = %s "
            " ORDER BY matched_at DESC, id DESC "
            " LIMIT %s"
        )
        with self._conn.cursor(row_factory=dict_row) as cur:
            cur.execute(sql, (memory_id, int(limit)))
            return [dict(r) for r in cur.fetchall()]

    def explain_lifetime_co_retrieval(
        self, *, memory_id: uuid.UUID, limit: int = 20
    ) -> list[dict[str, Any]]:
        """co_retrieved relations (DESC by weight) с JOIN на memories
        для target_preview.

        Идёт через relation='co_retrieved' (волна 21 Hebbian). source/
        target — memory side; нормализуем чтобы target_id был
        противоположной стороной от запрошенного memory_id.
        """
        sql = (
            "SELECT "
            "  CASE WHEN r.source_id = %s THEN r.target_id "
            "       ELSE r.source_id END AS target_id, "
            "  r.weight, r.metadata, "
            "  m.content AS target_content "
            "FROM relations r "
            "LEFT JOIN memories m ON m.id = "
            "  CASE WHEN r.source_id = %s THEN r.target_id "
            "       ELSE r.source_id END "
            "WHERE r.relation = 'co_retrieved' "
            "  AND r.source_type = 'memory' AND r.target_type = 'memory' "
            "  AND (r.source_id = %s OR r.target_id = %s) "
            "ORDER BY r.weight DESC "
            "LIMIT %s"
        )
        with self._conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                sql,
                (memory_id, memory_id, memory_id, memory_id, int(limit)),
            )
            return [dict(r) for r in cur.fetchall()]

    def explain_topk(
        self,
        *,
        query_vector: list[float],
        query_text: str,
        limit: int = 10,
        kinds: list[str] | None = None,
        after: Any | None = None,
        before: Any | None = None,
        decay_config: DecayConfig | None = None,
        search_config: SearchConfig | None = None,
        usage_norm_p75: float = 0.0,
        emotional_baseline: EmotionalBaseline | None = None,
    ) -> tuple[list[dict[str, Any]], int]:
        """Top-K с factor breakdown'ом + total_candidates_considered.

        Port memorybox explain.ts:562-697. Возвращает
        ``(rows, total_count)``. ``rows`` — list[dict] с теми же
        factor-колонками что decompose target SELECT (factor_select_columns).
        """
        opts = BuildFactorExprsOptions(
            text_query_param_index=2,
            search_config=search_config,
            decay_config=decay_config,
            table_alias=None,
            usage_norm_p75=usage_norm_p75,
            emotional_baseline=emotional_baseline,
        )
        factors = build_factor_exprs({"text_query": query_text}, opts)
        cols = factor_select_columns(factors)

        where: list[str] = [
            "agent_id = %(agent_id)s",
            "embedding IS NOT NULL",
            "superseded_by IS NULL",
        ]
        params: dict[str, Any] = {
            "qvec": _vector_literal(query_vector),
            "qtext": query_text,
            "agent_id": self._agent_id,
            "lim": int(limit),
        }
        if kinds:
            where.append("kind = ANY(%(kinds)s)")
            params["kinds"] = list(kinds)
        if after is not None:
            where.append("created_at >= %(after)s")
            params["after"] = after
        if before is not None:
            where.append("created_at <= %(before)s")
            params["before"] = before
        where_sql = " AND ".join(where)

        main_sql = f"""
            SELECT
                id, kind, content, created_at, updated_at, agent_id,
                visibility, superseded_by,
                importance_provisional, importance_final,
                lifecycle, relevance, access_count, last_accessed_at,
                unique_query_count, recall_score_sum, usefulness,
                {cols}
              FROM memories
              {factors.usage_lateral_from}
             WHERE {where_sql}
             ORDER BY final_score DESC NULLS LAST
             LIMIT %(lim)s
        """
        main_pg = main_sql.replace("$1", "%(qvec)s").replace("$2", "%(qtext)s")

        # Count: только agent + filters, без embedding'а — count относится
        # ко всем кандидатам пула (как в memorybox: total_candidates_considered).
        count_where = list(where)
        # убираем embedding/superseded — total_candidates это всё в scope'е
        # ну, memorybox также не убирает их (см. tools/explain.ts:626-637).
        # Оставляем те же фильтры — это «считаем сколько кандидатов».
        count_sql = (
            f"SELECT count(*)::int AS total FROM memories WHERE "
            f"{' AND '.join(count_where)}"
        )

        with self._conn.cursor(row_factory=dict_row) as cur:
            cur.execute(main_pg, params)
            main_rows = cur.fetchall()
            cur.execute(count_sql, params)
            count_row = cur.fetchone()

        total = int(count_row["total"]) if count_row is not None else 0
        return [dict(r) for r in main_rows], total

    def explain_topk_llm_tasks(
        self, *, memory_ids: list[uuid.UUID]
    ) -> dict[uuid.UUID, dict[str, Any]]:
        """Batch SELECT importance llm_task для каждого memory_id.

        DISTINCT ON (memory_id) — последняя задача того же task_type'а.
        Используется topK для заполнения llm_task_status / created_at в
        factor block'ах. Cross-agent — caller передаёт уже отфильтрованные
        memory_ids (своего агента).
        """
        if not memory_ids:
            return {}
        sql = (
            "SELECT DISTINCT ON (memory_id) "
            "       memory_id, status, created_at "
            "  FROM llm_tasks "
            " WHERE memory_id = ANY(%s::uuid[]) "
            "   AND task_type = 'importance_scoring_from_content' "
            " ORDER BY memory_id, created_at DESC"
        )
        with self._conn.cursor(row_factory=dict_row) as cur:
            cur.execute(sql, (list(memory_ids),))
            rows = cur.fetchall()
        return {
            r["memory_id"]: {
                "status": r["status"],
                "created_at": r["created_at"],
            }
            for r in rows
        }

    def analytics_for_agent(self) -> dict[str, Any]:
        """Per-agent counts + global totals + pending indexing.

        Port memorybox analytics.ts. Caller-scoped (один агент). Поля:
        ``memories_count``, ``memories_by_kind``, ``documents_count``,
        ``chunks_count``, ``dialogue_messages_count`` (через role
        IN ('user','assistant')), ``total_storage_bytes`` (sum
        ``documents.char_count``), ``total_relations`` (cross-agent
        by design § 33.2), ``database_size_bytes`` (best-effort,
        None при PG permission failure), ``pending_indexing``
        (memories+chunks с embedding IS NULL).
        """
        memories_by_kind: dict[str, int] = {}
        memories_count = 0
        with self._conn.cursor() as cur:
            cur.execute(
                "SELECT kind, count(*)::int FROM memories "
                "WHERE agent_id = %s GROUP BY kind",
                (self._agent_id,),
            )
            for kind, cnt in cur.fetchall():
                c = int(cnt)
                memories_by_kind[kind] = c
                memories_count += c

            cur.execute(
                "SELECT count(*)::int "
                "FROM memories WHERE agent_id = %s "
                "  AND role IN ('user','assistant')",
                (self._agent_id,),
            )
            dialogue_count = int(cur.fetchone()[0] or 0)

            cur.execute(
                "SELECT count(*)::int, "
                "       coalesce(sum(char_count), 0)::bigint "
                "FROM documents WHERE agent_id = %s",
                (self._agent_id,),
            )
            row = cur.fetchone()
            documents_count = int(row[0] or 0)
            total_storage_bytes = int(row[1] or 0)

            cur.execute(
                "SELECT count(c.id)::int "
                "FROM chunks c JOIN documents d ON c.document_id = d.id "
                "WHERE d.agent_id = %s",
                (self._agent_id,),
            )
            chunks_count = int(cur.fetchone()[0] or 0)

            # relations cross-agent by design (§ 33.2). Возвращаем
            # глобальный count как metric «насыщенности shared
            # knowledge graph'а».
            cur.execute("SELECT count(*)::int FROM relations")
            total_relations = int(cur.fetchone()[0] or 0)

            db_size: int | None = None
            try:
                cur.execute(
                    "SELECT pg_database_size(current_database())::bigint"
                )
                db_size = int(cur.fetchone()[0] or 0)
            except psycopg.Error:
                self._conn.rollback()
                db_size = None

            pending_dialogue = 0  # Styx без dialogue_messages-таблицы.
            cur.execute(
                "SELECT "
                "  (SELECT count(*)::int FROM memories WHERE embedding IS NULL) "
                "    AS pending_memories, "
                "  (SELECT count(*)::int FROM chunks WHERE embedding IS NULL) "
                "    AS pending_chunks, "
                "  (SELECT min(created_at) FROM ("
                "      SELECT created_at FROM memories WHERE embedding IS NULL "
                "      UNION ALL "
                "      SELECT created_at FROM chunks WHERE embedding IS NULL"
                "    ) p) AS oldest_pending_at"
            )
            pending_row = cur.fetchone() or (0, 0, None)
            pending_memories = int(pending_row[0] or 0)
            pending_chunks = int(pending_row[1] or 0)
            oldest_pending_at = pending_row[2]

        agent_block = {
            "agent_id": self._agent_id,
            "display_name": None,  # Styx без agents-table.
            "memories_count": memories_count,
            "memories_by_kind": memories_by_kind,
            "documents_count": documents_count,
            "chunks_count": chunks_count,
            "dialogue_messages_count": dialogue_count,
            "total_storage_bytes": total_storage_bytes,
        }
        global_block = {
            "total_memories": memories_count,
            "total_documents": documents_count,
            "total_chunks": chunks_count,
            "total_dialogue_messages": dialogue_count,
            "total_relations": total_relations,
            "total_storage_bytes": total_storage_bytes,
            "database_size_bytes": db_size,
        }
        pending_block = {
            "dialogue_messages": pending_dialogue,
            "memories": pending_memories,
            "chunks": pending_chunks,
            "oldest_pending_at": oldest_pending_at,
        }
        return {
            "agents": [agent_block],
            "global": global_block,
            "pending_indexing": pending_block,
        }

    def confirm_usage_update(
        self, *, memory_ids: list[uuid.UUID]
    ) -> set[uuid.UUID]:
        """UPDATE recall_events.used_in_output=true для DISTINCT-ON
        самых свежих recall_events каждого memory_id.

        Scope guard через JOIN на ``memories.agent_id`` —
        recall_events чужого агента не апдейтятся. RETURNING memory_id
        возвращает фактически matched ids.

        Idempotent: повторный UPDATE того же ряда не падает (RETURNING
        всё равно возвращает row, count'ит как updated).

        Не делает commit — caller управляет транзакцией.
        """
        if not memory_ids:
            return set()
        sql = (
            "WITH latest AS ( "
            "  SELECT DISTINCT ON (re.memory_id) re.id, re.memory_id "
            "    FROM recall_events re "
            "    JOIN memories m ON m.id = re.memory_id "
            "   WHERE re.memory_id = ANY(%s::uuid[]) "
            "     AND m.agent_id = %s "
            "   ORDER BY re.memory_id, re.matched_at DESC "
            ") "
            "UPDATE recall_events re "
            "   SET used_in_output = true "
            "  FROM latest "
            " WHERE re.id = latest.id "
            "RETURNING re.memory_id"
        )
        with self._conn.cursor() as cur:
            cur.execute(sql, (list(memory_ids), self._agent_id))
            rows = cur.fetchall()
        return {r[0] for r in rows}

    def iter_all_for_agent(self) -> Iterator[StoredMessage]:
        """Только для отладки/миграций. Не использовать в горячем пути."""
        with self._conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                "SELECT id, session_id, role, content, metadata, created_at "
                "FROM memories WHERE agent_id = %s ORDER BY seq",
                (self._agent_id,),
            )
            for r in cur:
                yield StoredMessage(
                    id=r["id"],
                    session_id=r["session_id"],
                    role=r["role"],
                    content=r["content"],
                    metadata=r["metadata"],
                    created_at=r["created_at"],
                )


def factor_select_columns(factors: "FactorExprs") -> str:
    """SELECT-list фрагмент со всеми 11 факторами scoring'а как
    отдельными колонками.

    Port memorybox ``factorSelectColumns`` (explain.ts:59-82). Используется
    decompose target SELECT и topK main SELECT — caller затем
    собирает FactorsBlock через ``engine.explain.build_factors_block``.
    """
    f = factors
    bm25_col = (
        f"{f.bm25_expr} AS bm25_rank"
        if f.bm25_expr is not None
        else "NULL::real AS bm25_rank"
    )
    cols = [
        f"{f.vector_sim_expr} AS vector_sim",
        bm25_col,
        f"({f.base_match_expr}) AS base_match",
        f"{f.relevance_ref_expr} AS relevance_factor",
        f"({f.recency_expr}) AS recency_factor",
        f"({f.frequency_expr}) AS frequency_factor",
        f"({f.lifecycle_expr}) AS lifecycle_factor",
        f"({f.feedback_expr}) AS feedback_factor",
        f"({f.importance_expr}) AS importance_factor",
        f"({f.importance_effective_expr}) AS importance_effective",
        f"({f.diversity_expr}) AS diversity_factor",
        f"({f.decay_expr}) AS decay_factor",
        f"({f.lambda_base_expr}) AS lambda_base",
        f"({f.effective_lambda_expr}) AS effective_lambda",
        f"{f.usage_count_expr} AS usage_count_30d",
        f"({f.usage_factor_expr}) AS usage_factor",
        f"({f.emotional_resonance_expr}) AS emotional_resonance_factor",
        f"{f.age_days_expr} AS age_days_sql",
        f"({f.score_expr}) AS final_score",
    ]
    return ",\n       ".join(cols)


# ── Worker-side queries (cross-agent, без agent_id scope) ──────────────
#
# Worker drain'ит llm_tasks по PK и читает memory по PK — agent_id
# фильтрация не нужна, т.к. PK уникален по всей таблице. Хелперы
# намеренно вне AgentScopedQueries — у worker'а нет agent-context'а.


@dataclass(frozen=True)
class _WorkerMemoryRow:
    """Узкая выборка из ``memories`` для LLM-handler'ов."""

    content: str
    kind: str
    created_at: Any
    importance_provisional: float


def worker_load_memory(
    conn: psycopg.Connection, memory_id: uuid.UUID
) -> _WorkerMemoryRow | None:
    """SELECT по memory.id для importance handler'а (волна 7a).

    Возвращает None если row удалён (race с CASCADE).
    """
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            "SELECT content, kind, created_at, importance_provisional "
            "  FROM memories WHERE id = %s",
            (memory_id,),
        )
        row = cur.fetchone()
    if row is None:
        return None
    return _WorkerMemoryRow(
        content=row["content"],
        kind=row["kind"],
        created_at=row["created_at"],
        importance_provisional=float(row["importance_provisional"]),
    )


def worker_load_memory_for_reinterpret(
    conn: psycopg.Connection, memory_id: uuid.UUID
) -> dict | None:
    """SELECT по PK для reinterpret_merge handler'а (волна 22).

    Возвращает {id, agent_id, content, embedding} или None если row
    удалён (race с CASCADE). Cross-agent — caller сам сравнивает
    agent_id с payload'ом для terminal mismatch error.
    """
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            "SELECT id, agent_id, content, embedding "
            "  FROM memories WHERE id = %s",
            (memory_id,),
        )
        row = cur.fetchone()
    return dict(row) if row else None


def worker_update_importance_final(
    conn: psycopg.Connection, memory_id: uuid.UUID, score: float
) -> None:
    """UPDATE memories.importance_final по PK. Не делает commit."""
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE memories SET importance_final = %s WHERE id = %s",
            (score, memory_id),
        )


# ── Reembed helpers (волна 7e, cross-agent admin-tier) ─────────────────


def reembed_count_targets(
    conn: psycopg.Connection,
    *,
    null_only: bool,
    agent_id: str | None = None,
) -> int:
    """SELECT count(*) для прогноза `--dry-run`."""
    where, params = _reembed_where(null_only=null_only, agent_id=agent_id)
    with conn.cursor() as cur:
        cur.execute(f"SELECT count(*)::int FROM memories WHERE {where}", params)
        return int(cur.fetchone()[0])


def reembed_iter_targets(
    conn: psycopg.Connection,
    *,
    null_only: bool,
    agent_id: str | None,
    batch_size: int,
) -> "Iterator[tuple[uuid.UUID, str]]":
    """Cursor-pagination по id. Yields (memory_id, content).

    Не закрывает курсор — caller'у важно итерировать до конца либо
    рано выходить (psycopg сам соберёт мусор). UPDATE'ы embedding'ом не
    влияют на курсор, потому что мы упорядочены по id (PK не мигрирует).
    """
    where, params_base = _reembed_where(null_only=null_only, agent_id=agent_id)
    last_id: uuid.UUID | None = None
    while True:
        if last_id is None:
            cur_where = where
            params = list(params_base)
        else:
            cur_where = where + " AND id > %s"
            params = [*params_base, last_id]
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT id, content FROM memories "
                f" WHERE {cur_where} "
                f" ORDER BY id "
                f" LIMIT %s",
                [*params, batch_size],
            )
            rows = cur.fetchall()
        if not rows:
            return
        for row in rows:
            yield row[0], row[1]
        last_id = rows[-1][0]
        if len(rows) < batch_size:
            return


def reembed_update_embedding(
    conn: psycopg.Connection, memory_id: uuid.UUID, embedding: list[float]
) -> None:
    """UPDATE memories.embedding по PK без agent-scope. Не commit'ит.

    Триггер `enqueue_importance_scoring` сидит на ``UPDATE OF content``
    — этот UPDATE столбца ``embedding`` его НЕ дёргает. Намеренно.
    """
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE memories SET embedding = %s WHERE id = %s",
            (_vector_literal(embedding), memory_id),
        )


def _reembed_where(
    *, null_only: bool, agent_id: str | None
) -> tuple[str, list[Any]]:
    parts = ["superseded_by IS NULL"]
    params: list[Any] = []
    if null_only:
        parts.append("embedding IS NULL")
    if agent_id is not None:
        parts.append("agent_id = %s")
        params.append(agent_id)
    return " AND ".join(parts), params


def _as_uuid(value: uuid.UUID | str) -> uuid.UUID:
    if isinstance(value, uuid.UUID):
        return value
    try:
        return uuid.UUID(value)
    except ValueError:
        raise ValueError(
            f"Некорректный UUID (тип={type(value).__name__}, длина={len(value)})"
        ) from None


def _vector_literal(embedding: list[float] | None) -> str | None:
    """pgvector принимает text-литерал ``[1.0,2.0,...]``.

    psycopg сам не знает тип vector, поэтому мы передаём строку, а в БД
    выполняется неявный cast text → vector.
    """
    if embedding is None:
        return None
    if not all(math.isfinite(x) for x in embedding):
        raise ValueError(
            "embedding содержит non-finite значения (NaN или Inf) — "
            "запись в pgvector невозможна"
        )
    return "[" + ",".join(repr(float(x)) for x in embedding) + "]"


# ── Волна 14: cross-agent helpers (без agent_id scope) ────────────────
#
# Используются scheduler'ом (`workers/sweep/batch_consolidation.py`) и
# handler'ом (`workers/handlers/dialogue_batch_consolidation.py`) — у
# них нет agent_id-scope конкретного провайдера, agent_id передаётся
# параметром.


def list_subject_agents(conn: psycopg.Connection) -> list[str]:
    """``SELECT DISTINCT agent_id FROM memories`` исключая 'system'.

    'system' — bootstrap agent_id (если кем-то использовался); реальные
    subject-агенты берутся отсюда. Сортировка для детерминизма.
    """
    sql = (
        "SELECT DISTINCT agent_id FROM memories "
        "WHERE agent_id <> 'system' ORDER BY agent_id"
    )
    with conn.cursor() as cur:
        cur.execute(sql)
        return [row[0] for row in cur.fetchall()]


def get_batch_state(conn: psycopg.Connection, agent_id: str) -> dict | None:
    """``consolidation_state`` row под ключом
    ``batch_consolidation:<agent_id>``. None если не существует."""
    sql = (
        "SELECT value FROM consolidation_state "
        "WHERE key = %s"
    )
    with conn.cursor() as cur:
        cur.execute(sql, (f"batch_consolidation:{agent_id}",))
        row = cur.fetchone()
    if row is None:
        return None
    return row[0] if isinstance(row[0], dict) else None


def set_batch_state(
    conn: psycopg.Connection, agent_id: str, state: dict
) -> None:
    """Upsert ``consolidation_state`` под ключом
    ``batch_consolidation:<agent_id>``. Не делает commit."""
    from psycopg.types.json import Jsonb
    sql = (
        "INSERT INTO consolidation_state (key, value, updated_at) "
        "VALUES (%s, %s, now()) "
        "ON CONFLICT (key) DO UPDATE "
        "SET value = EXCLUDED.value, updated_at = EXCLUDED.updated_at"
    )
    with conn.cursor() as cur:
        cur.execute(
            sql, (f"batch_consolidation:{agent_id}", Jsonb(state))
        )


def enqueue_llm_task(
    conn: psycopg.Connection,
    *,
    task_type: str,
    payload: dict,
    memory_id: uuid.UUID | None = None,
) -> Any:
    """INSERT в llm_tasks status='pending'. Возвращает ``id`` (UUID).
    Не делает commit.

    `memory_id` (волна 22) — опциональный: для reinterpret_merge id
    пишется в колонку для idx_llm_tasks_memory + ON DELETE CASCADE
    chain через reinterpret_applications.memory_id FK.
    """
    from psycopg.types.json import Jsonb
    if memory_id is None:
        sql = (
            "INSERT INTO llm_tasks (task_type, payload, status) "
            "VALUES (%s, %s, 'pending') RETURNING id"
        )
        params: tuple = (task_type, Jsonb(payload))
    else:
        sql = (
            "INSERT INTO llm_tasks (task_type, memory_id, payload, status) "
            "VALUES (%s, %s, %s, 'pending') RETURNING id"
        )
        params = (task_type, memory_id, Jsonb(payload))
    with conn.cursor() as cur:
        cur.execute(sql, params)
        return cur.fetchone()[0]


def parse_vector(value: Any) -> list[float] | None:
    """Обратное к _vector_literal — pgvector → list[float].

    psycopg по умолчанию возвращает vector как str ``"[0.1,0.2,...]"``,
    парсим вручную. None → None. Public с волны 22 — используется
    engine'ами reinterpret/memory_consolidation.
    """
    if value is None:
        return None
    if isinstance(value, list):
        return [float(x) for x in value]
    if isinstance(value, str):
        # Формат: "[0.1,0.2,0.3]" или "[0.1, 0.2, 0.3]".
        s = value.strip()
        if s.startswith("[") and s.endswith("]"):
            s = s[1:-1]
        if not s:
            return []
        return [float(x.strip()) for x in s.split(",")]
    raise TypeError(f"Не умею парсить vector из {type(value).__name__}: {value!r}")


# Backwards-compat alias — internal callers до волны 22 импортировали
# `_parse_vector`. Сохраняем чтобы не ломать существующие импорты.
_parse_vector = parse_vector


# ── Волна 22: memory_daily consolidation KV-state ─────────────────────


def get_memory_daily_state(
    conn: psycopg.Connection, agent_id: str
) -> dict | None:
    """``consolidation_state`` row под ключом ``memory_daily:<agent_id>``.
    None если не существует."""
    sql = "SELECT value FROM consolidation_state WHERE key = %s"
    with conn.cursor() as cur:
        cur.execute(sql, (f"memory_daily:{agent_id}",))
        row = cur.fetchone()
    if row is None:
        return None
    return row[0] if isinstance(row[0], dict) else None


def set_memory_daily_state(
    conn: psycopg.Connection, agent_id: str, state: dict
) -> None:
    """Upsert ``consolidation_state`` под ключом
    ``memory_daily:<agent_id>``. Не делает commit."""
    from psycopg.types.json import Jsonb
    sql = (
        "INSERT INTO consolidation_state (key, value, updated_at) "
        "VALUES (%s, %s, now()) "
        "ON CONFLICT (key) DO UPDATE "
        "SET value = EXCLUDED.value, updated_at = EXCLUDED.updated_at"
    )
    with conn.cursor() as cur:
        cur.execute(sql, (f"memory_daily:{agent_id}", Jsonb(state)))
